"""Deterministic replay controls and validation for DIME."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.base_agent import BaseAgent
from agents.heuristic_agent import HeuristicAgent
from benchmark.utils import action_to_dict, observation_to_dict
from server.environment import DistributedInfraEnvironment
from server.models import InfraAction


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch if installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        return


@dataclass(frozen=True)
class ReplayValidationResult:
    """Outcome from deterministic replay validation."""

    passed: bool
    task_id: str
    seed: int
    topology_template: str
    trace_offset: int
    steps: int


def _reset_agent(agent: Any, seed: int, task_id: str) -> None:
    reset = getattr(agent, "reset", None)
    if reset is None:
        return
    try:
        reset(seed=seed, task_id=task_id)
    except TypeError:
        reset()


def _coerce_action(action: Any) -> InfraAction:
    if isinstance(action, InfraAction):
        return action
    if isinstance(action, dict):
        try:
            return InfraAction.model_validate(action)
        except Exception:
            return InfraAction(action_type="no_op")
    return InfraAction(action_type="no_op")


def _run_replay(
    agent: BaseAgent,
    *,
    task_id: str,
    seed: int,
    topology_template: str,
    trace_offset: int,
) -> dict[str, Any]:
    set_global_seed(seed)
    _reset_agent(agent, seed, task_id)
    env = DistributedInfraEnvironment()
    obs = env.reset(
        seed=seed,
        task=task_id,
        topology_template=topology_template,
        trace_offset=trace_offset,
    )
    trajectory: list[dict[str, Any]] = []
    rewards: list[float] = []

    while True:
        action = _coerce_action(agent.act(obs))
        obs = env.step(action)
        obs_dict = observation_to_dict(obs)
        rewards.append(float(obs_dict.get("reward", 0.0) or 0.0))
        trajectory.append(
            {
                "action": action_to_dict(action),
                "reward": rewards[-1],
                "latency_ms": obs_dict.get("latency_ms"),
                "failed_nodes": obs_dict.get("failed_nodes", []),
                "step": obs_dict.get("step"),
            }
        )
        if bool(obs_dict.get("done", False)) or env.sim.step_count >= env.sim.max_steps:
            break

    return {
        "rewards": rewards,
        "latency_history": list(env.sim.latency_history),
        "failure_history": [row["failed_nodes"] for row in trajectory],
        "trajectory": trajectory,
    }


def validate_replay(
    agent: BaseAgent | None = None,
    task_id: str = "traffic_spike",
    seed: int = 42,
    topology_template: str = "default",
    trace_offset: int = 0,
) -> ReplayValidationResult:
    """Run identical seeds twice and fail if deterministic replay diverges."""
    active_agent = agent or HeuristicAgent()
    first = _run_replay(
        active_agent,
        task_id=task_id,
        seed=seed,
        topology_template=topology_template,
        trace_offset=trace_offset,
    )
    second = _run_replay(
        active_agent,
        task_id=task_id,
        seed=seed,
        topology_template=topology_template,
        trace_offset=trace_offset,
    )
    if first != second:
        raise AssertionError(
            "Deterministic replay diverged for "
            f"seed={seed}, task={task_id}, topology={topology_template}, trace_offset={trace_offset}"
        )
    return ReplayValidationResult(
        passed=True,
        task_id=task_id,
        seed=seed,
        topology_template=topology_template,
        trace_offset=trace_offset,
        steps=len(first["trajectory"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deterministic DIME replay.")
    parser.add_argument("--task", default="traffic_spike")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topology-template", default="default")
    parser.add_argument("--trace-offset", type=int, default=0)
    args = parser.parse_args()
    result = validate_replay(
        task_id=args.task,
        seed=args.seed,
        topology_template=args.topology_template,
        trace_offset=args.trace_offset,
    )
    print(result)


if __name__ == "__main__":
    main()
