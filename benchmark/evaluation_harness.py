"""Official DIME-v1.0 benchmark evaluation harness."""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import requests

from agents.base_agent import BaseAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from agents.threshold_agent import ThresholdAgent
from benchmark.benchmark_config import BenchmarkConfig, DIME_V1_CONFIG
from benchmark.benchmark_registry import Split, TaskSpec, get_benchmark_task_specs
from benchmark.deterministic import set_global_seed
from benchmark.dime_index import compute_dime_index, select_latency_normalization
from benchmark.statistical_report import build_statistical_report, persist_statistical_report
from benchmark.utils import (
    BENCHMARK_RUNS_DIR,
    SEED_LOGS_DIR,
    STATISTICAL_REPORTS_DIR,
    action_to_dict,
    append_jsonl,
    atomic_write_json,
    ensure_result_dirs,
    observation_to_dict,
    to_plain_data,
    utc_run_id,
    write_csv,
)
from server.environment import DistributedInfraEnvironment
from server.models import InfraAction


class CallableAgent(BaseAgent):
    """Adapter for local callables that return InfraAction-compatible data."""

    def __init__(self, fn: Callable[[Any], Any]) -> None:
        self._fn = fn

    def act(self, observation: Any) -> Any:
        return self._fn(observation)


class APIAgent(BaseAgent):
    """Adapter for API agents that accept JSON observations and return actions."""

    def __init__(self, endpoint: str, timeout_s: float = 30.0) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def act(self, observation: Any) -> Any:
        response = requests.post(
            self.endpoint,
            json={"observation": observation_to_dict(observation)},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("action", payload)


class ReplayAgent(BaseAgent):
    """Replay pre-recorded actions for deterministic trajectory checks."""

    def __init__(self, actions: Iterable[Mapping[str, Any]]) -> None:
        self._actions = [dict(action) for action in actions]
        self._idx = 0

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        self._idx = 0

    def act(self, observation: Any) -> Any:
        if self._idx >= len(self._actions):
            return InfraAction(action_type="no_op")
        action = self._actions[self._idx]
        self._idx += 1
        return action


def _resolve_agent(agent: str | BaseAgent | Callable[[Any], Any]) -> BaseAgent:
    if isinstance(agent, BaseAgent):
        return agent
    if callable(agent) and not isinstance(agent, str):
        return CallableAgent(agent)
    if agent == "random":
        return RandomAgent()
    if agent == "heuristic":
        return HeuristicAgent()
    if agent == "threshold":
        return ThresholdAgent()
    if isinstance(agent, str) and agent.startswith("http"):
        return APIAgent(agent)
    raise ValueError(f"Unknown agent specifier: {agent!r}")


def _coerce_action(action: Any) -> InfraAction:
    if isinstance(action, InfraAction):
        return action
    if isinstance(action, Mapping):
        try:
            return InfraAction.model_validate(dict(action))
        except Exception:
            return InfraAction(action_type="no_op")
    return InfraAction(action_type="no_op")


def _reset_agent(agent: BaseAgent, seed: int, task_id: str) -> None:
    try:
        agent.reset(seed=seed, task_id=task_id)
    except TypeError:
        agent.reset()


@contextlib.contextmanager
def _inference_only(agent: BaseAgent):
    """Block common online-learning mutation entrypoints during evaluation."""
    patched: list[tuple[Any, str, Any]] = []

    def disabled(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("DIME benchmark evaluation is inference-only")

    names = (
        "backward",
        "learn",
        "optimize",
        "optimizer_step",
        "policy_update",
        "rollout",
        "train_step",
        "update",
        "update_policy",
    )
    for name in names:
        if hasattr(agent, name):
            patched.append((agent, name, getattr(agent, name)))
            setattr(agent, name, disabled)

    for owner_name in ("optimizer", "optim", "replay_buffer"):
        owner = getattr(agent, owner_name, None)
        if owner is None:
            continue
        for name in ("step", "add", "append", "extend", "push", "update"):
            if hasattr(owner, name):
                patched.append((owner, name, getattr(owner, name)))
                setattr(owner, name, disabled)

    if hasattr(agent, "eval"):
        try:
            agent.eval()
        except TypeError:
            pass
    if hasattr(agent, "train"):
        try:
            agent.train(False)
        except TypeError:
            pass

    try:
        try:
            import torch

            with torch.inference_mode():
                yield
        except ImportError:
            yield
    finally:
        for owner, name, original in reversed(patched):
            setattr(owner, name, original)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _mttr(uptime_history: list[float]) -> float:
    durations: list[int] = []
    current = 0
    for uptime in uptime_history:
        if uptime < 1.0:
            current += 1
        elif current:
            durations.append(current)
            current = 0
    if current:
        durations.append(current)
    return sum(durations) / len(durations) if durations else 0.0


def _episode_metrics(
    env: DistributedInfraEnvironment,
    rewards: list[float],
    task_score: float,
    initial_cloud_budget: int,
) -> dict[str, float]:
    sim = env.sim
    uptime = sum(sim.uptime_history) / len(sim.uptime_history) if sim.uptime_history else 0.0
    throughput = sim.total_requests_served / max(1, sim.total_requests_received)
    alive = sum(1 for node in sim.nodes if not node.is_failed)
    total = max(1, len(sim.nodes))
    resource_cost = max(0, initial_cloud_budget - sim.cloud_budget)
    return {
        "uptime": uptime,
        "p99_latency": _percentile(sim.latency_history, 99.0),
        "throughput": throughput,
        "throughput_ratio": throughput,
        "mttr": _mttr(sim.uptime_history),
        "resource_cost": float(resource_cost),
        "max_budget": float(max(1, initial_cloud_budget)),
        "initial_cloud_budget": float(initial_cloud_budget),
        "survival_rate": alive / total,
        "cumulative_reward": sum(rewards),
        "task_success": 1.0 if task_score >= 0.8 else 0.0,
        "task_score": task_score,
    }


def _config_snapshot(config: BenchmarkConfig, selected_latency_method: str | None = None) -> dict[str, Any]:
    snapshot = to_plain_data(config)
    if selected_latency_method is not None:
        snapshot["selected_latency_method"] = selected_latency_method
    return snapshot


def _run_episode(
    agent: BaseAgent,
    spec: TaskSpec,
    seed: int,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    set_global_seed(seed)
    _reset_agent(agent, seed=seed, task_id=spec.task_id)
    env = DistributedInfraEnvironment()
    obs = env.reset(seed=seed, episode_id=f"{spec.registry_id}:{seed}", **spec.reset_kwargs)
    initial_cloud_budget = env.sim.cloud_budget
    trajectory: list[dict[str, Any]] = [
        {"event": "reset", "seed": seed, "task_id": spec.task_id, "registry_id": spec.registry_id, "observation": observation_to_dict(obs)}
    ]
    rewards: list[float] = []
    task_score = float(getattr(obs, "task_score", 0.0) or 0.0)
    start = time.perf_counter()

    while True:
        action = _coerce_action(agent.act(obs))
        obs = env.step(action)
        obs_dict = observation_to_dict(obs)
        reward = float(obs_dict.get("reward", 0.0) or 0.0)
        rewards.append(reward)
        task_score = float(obs_dict.get("task_score", task_score) or task_score)
        trajectory.append(
            {
                "event": "step",
                "step": obs_dict.get("step"),
                "action": action_to_dict(action),
                "reward": reward,
                "done": bool(obs_dict.get("done", False)),
                "task_score": task_score,
                "observation": obs_dict,
            }
        )
        if bool(obs_dict.get("done", False)) or env.sim.step_count >= env.sim.max_steps:
            break

    elapsed_s = time.perf_counter() - start
    raw_path = run_dir / "trajectories" / spec.registry_id / f"seed_{seed:03d}.jsonl"
    append_jsonl(raw_path, trajectory)
    seed_log_path = SEED_LOGS_DIR / f"{run_dir.name}_{spec.registry_id}_seed_{seed:03d}.json"

    metrics = _episode_metrics(env, rewards, task_score, initial_cloud_budget)
    row = {
        "benchmark_version": DIME_V1_CONFIG.benchmark_version,
        "registry_id": spec.registry_id,
        "task_id": spec.task_id,
        "split": spec.split.value,
        "seed": seed,
        "topology_template": spec.topology_template,
        "trace_offset": spec.trace_offset,
        "steps": env.sim.step_count,
        "elapsed_s": round(elapsed_s, 6),
        "trajectory_path": str(raw_path),
        **metrics,
    }
    atomic_write_json(seed_log_path, row)
    return row


def run_benchmark(
    agent: str | BaseAgent | Callable[[Any], Any],
    benchmark_version: str = "DIME-v1.0",
    split: str = "hidden_eval",
) -> dict[str, Any]:
    """Run the official DIME benchmark and persist all artifacts."""
    if benchmark_version != DIME_V1_CONFIG.benchmark_version:
        raise ValueError(f"Unsupported benchmark version: {benchmark_version}")

    ensure_result_dirs()
    active_agent = _resolve_agent(agent)
    split_value = Split(split)
    specs = get_benchmark_task_specs(split_value)
    config = DIME_V1_CONFIG
    seeds = config.evaluation_protocol.seeds
    if len(seeds) != config.evaluation_protocol.episodes_per_task:
        raise RuntimeError("DIME-v1.0 requires exactly 100 seeds for 100 episodes per task")

    run_id = utc_run_id(f"{benchmark_version}_{split_value.value}")
    run_dir = BENCHMARK_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "benchmark_config.initial.json", _config_snapshot(config))

    episode_rows: list[dict[str, Any]] = []
    with _inference_only(active_agent):
        for spec in specs:
            for seed in seeds:
                episode_rows.append(_run_episode(active_agent, spec, seed, run_dir=run_dir))

    latency_selection = select_latency_normalization(episode_rows)
    selected_method = latency_selection["selected_method"]
    final_config_snapshot = _config_snapshot(config, selected_method)
    final_config_snapshot["latency_method_selection"] = latency_selection

    scored_rows: list[dict[str, Any]] = []
    for row in episode_rows:
        score_payload = compute_dime_index(row, final_config_snapshot)
        scored_rows.append({**row, **score_payload})

    report = build_statistical_report(scored_rows)
    summary = {
        "run_id": run_id,
        "benchmark_version": benchmark_version,
        "split": split_value.value,
        "episodes_per_task": config.evaluation_protocol.episodes_per_task,
        "num_tasks": len(specs),
        "num_episodes": len(scored_rows),
        "selected_latency_method": selected_method,
        "latency_method_selection": latency_selection,
        "mean_dime_index": report["episodes"]["dime_index"]["mean"],
        "artifact_dir": str(run_dir),
    }

    atomic_write_json(run_dir / "benchmark_config.snapshot.json", final_config_snapshot)
    atomic_write_json(run_dir / "benchmark_summary.json", summary)
    atomic_write_json(run_dir / "episode_metrics.json", scored_rows)
    write_csv(
        run_dir / "episode_metrics.csv",
        scored_rows,
        [
            "benchmark_version",
            "registry_id",
            "task_id",
            "split",
            "seed",
            "topology_template",
            "trace_offset",
            "steps",
            "dime_index",
            "uptime",
            "latency_score",
            "throughput",
            "recovery_speed",
            "cost_efficiency",
            "p99_latency",
            "mttr",
            "resource_cost",
            "cumulative_reward",
            "task_success",
            "survival_rate",
            "task_score",
        ],
    )
    persist_statistical_report(
        report,
        STATISTICAL_REPORTS_DIR / f"{run_id}.json",
        STATISTICAL_REPORTS_DIR / f"{run_id}.csv",
    )
    atomic_write_json(run_dir / "statistical_report.json", report)
    return {"summary": summary, "report": report, "run_dir": str(run_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official DIME-v1.0 benchmark.")
    parser.add_argument("--agent", default="heuristic", help="random, heuristic, threshold, or an HTTP endpoint")
    parser.add_argument("--split", default="hidden_eval", choices=[split.value for split in Split])
    parser.add_argument("--benchmark-version", default="DIME-v1.0")
    args = parser.parse_args()
    result = run_benchmark(args.agent, benchmark_version=args.benchmark_version, split=args.split)
    print(result["summary"])


if __name__ == "__main__":
    main()
