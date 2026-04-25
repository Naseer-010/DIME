#!/usr/bin/env python3
"""
LLM Agent Inference Loop for Distributed Infrastructure Environment (DIME).

Runs the LLM agent against all graded tasks and reports model-specific
performance summaries to structured log files. Strictly adheres to the
Meta-PyTorch OpenEnv Hackathon STDOUT format.
"""

import json
import os
import sys
import time
import traceback
import math
from contextlib import contextmanager
from pathlib import Path
import requests
from typing import List, Optional, Dict
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:7860")

TASKS = [
    "traffic_spike",
    "node_failure",
    "cascading_failure",
    "flash_crowd",
    "level_5_alibaba_trace",
]
MAX_RETRIES = 3
BENCHMARK = "distributed_infra_env"

client: Optional[OpenAI] = None

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) managing a Kubernetes cluster.
You receive observations about the system state as JSON and must respond with a SINGLE kubectl command.

CLUSTER ARCHITECTURE:
- Node 0 (worker-0) is the DATABASE — all request processing requires a DB query.
  If the DB is overloaded or fails, ALL app servers stop processing.
- Nodes 1-7 (worker-1 through worker-7) are APP SERVERS — they receive external traffic.
- New nodes from scale_up have a 3-STEP COLD START (10% processing speed during boot).

Available commands:
- kubectl delete pod node-<ID>           → restart a failed node
- kubectl scale deployment frontend --replicas=<N>  → scale up capacity (costs 1 budget unit, 3-step cold start)
- kubectl exec -it istio-proxy -- traffic shift --from=<ID> --to=<ID>  → reroute traffic
- kubectl throttle ingress --rate=<float>  → throttle incoming requests (0.0-1.0)
- kubectl logs node-<ID>                 → investigate a node with telemetry timeout
- no_op                                  → do nothing

CONSTRAINTS:
- Cloud budget is LIMITED. Check cloud_budget before scaling.
- Restart has a 5-step cooldown per node.
- NEVER set throttle rate below 0.3 — dropping >70% of traffic causes a massive throughput penalty.
- Observations include 'prometheus_metrics' in production scrape format.

CRITICAL DECISION TREE (Follow strictly):
1. IF 'action_errors' contains "CRITICAL: Database node failed":
   IMMEDIATELY output: kubectl delete pod node-0
2. IF any node has telemetry "timeout" (cpu_load == -1):
   Output: kubectl logs node-<ID>
3. IF 'failed_nodes' is not empty:
   IMMEDIATELY output: kubectl delete pod node-<failed_node_index>
4. IF node-0 (DB) cpu_load > 0.75:
   RELIEVE DB PRESSURE: kubectl throttle ingress --rate=0.7
   (Do NOT reroute traffic away from DB — all servers depend on it)
5. IF any app server cpu_load > 0.85:
   Find the node with highest CPU and lowest CPU among app servers.
   Output: kubectl exec -it istio-proxy -- traffic shift --from=<high> --to=<low>
6. IF average cpu_loads > 0.70 AND cloud_budget > 0:
   Output: kubectl scale deployment frontend --replicas=10
7. IF 'latency_ms' > 45.0:
   Output: kubectl throttle ingress --rate=0.8
8. IF none of the above are true:
   Output: no_op

Respond with ONLY the kubectl command or "no_op". No markdown, no explanation."""

# ---------------------------------------------------------------------------
# Required Logging Functions
# ---------------------------------------------------------------------------


def get_log_path(model_name: str) -> Path:
    """Generate a dynamic, safe log file name based on the model being tested."""
    safe_name = model_name.replace("/", "_").replace(".", "_")
    return Path(__file__).resolve().parent / f"logs_{safe_name}.txt"


class TeeStream:
    """Write stream output to the terminal and a log file."""

    def __init__(self, primary, log_file) -> None:
        self.primary = primary
        self.log_file = log_file
        self.encoding = getattr(primary, "encoding", "utf-8")
        self.errors = getattr(primary, "errors", "replace")

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()


@contextmanager
def tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def log_start(task: str, env: str, model: str) -> None:
    print(f"\n[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------


def get_client() -> OpenAI:
    global client
    if not API_KEY:
        raise ValueError("API_KEY or HF_TOKEN environment variable is missing!")
    if client is None:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return client


def llm_decide(observation: dict) -> dict:
    obs_str = json.dumps(observation)
    user_prompt = f"Current system state:\n{obs_str}\nRespond with ONLY a kubectl command or 'no_op'."

    for attempt in range(MAX_RETRIES):
        try:
            response = get_client().chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,
                temperature=0.01,
            )
            content = response.choices[0].message.content.strip()
            # Strip any markdown code fences
            if "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # If the LLM returned a raw kubectl command, wrap it
            if content.startswith("kubectl") or content.startswith("aws"):
                return {"raw_command": content}

            # If it returned "no_op", handle that
            if "no_op" in content.lower() or "noop" in content.lower():
                return {"action_type": "no_op"}

            # Try parsing as JSON (backward compatibility)
            return json.loads(content)
        except Exception as e:
            print(
                f"[DEBUG] LLM call attempt {attempt + 1} failed: {str(e)}", flush=True
            )
            time.sleep(1)

    # If it fails all retries, return a no_op
    return {"action_type": "no_op"}


def env_reset(task_id: str) -> dict:
    response = requests.post(
        f"{ENV_SERVER_URL}/reset", json={"task": task_id}, timeout=10
    )
    response.raise_for_status()
    payload = response.json()
    data_block = payload.get("data", payload)
    if "observation" in data_block and isinstance(data_block["observation"], dict):
        return data_block["observation"]
    return data_block


def env_step(action: dict) -> dict:
    response = requests.post(
        f"{ENV_SERVER_URL}/step", json={"action": action}, timeout=10
    )
    response.raise_for_status()
    return response.json()


def run_task(task_id: str, model_name: str) -> dict:
    log_start(task=task_id, env=BENCHMARK, model=model_name)

    try:
        obs = env_reset(task_id)
    except Exception as e:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return {
            "task": task_id,
            "score": 0.0,
            "avg_latency": 0.0,
            "avg_uptime": 0.0,
            "total_steps": 0,
        }

    step = 0
    rewards_list = []
    latencies = []
    uptimes = []
    task_score = 0.0

    while True:
        step += 1

        # Track metrics if present in the observation
        lat = float(obs.get("latency_ms", 0.0))
        up = float(obs.get("uptime_pct", 0.0))
        if lat > 0:
            latencies.append(lat)
        if up > 0:
            uptimes.append(up)

        action = llm_decide(obs)

        # Format action strictly on one line without quotes that break bash/parsing
        action_str = json.dumps(action).replace('"', "'")

        error_msg = None
        reward = 0.0
        done = False

        try:
            result = env_step(action)
            data_block = result.get("data", result)

            if "observation" in data_block and isinstance(
                data_block["observation"], dict
            ):
                obs = data_block["observation"]
            else:
                obs = data_block

            reward = float(data_block.get("reward", obs.get("reward", 0.0)))
            done = bool(data_block.get("done", obs.get("done", False)))

            # Continuously update task_score
            task_score = float(obs.get("task_score", 0.0))

        except Exception as e:
            error_msg = str(e).replace("\n", " ")  # Prevent newline breaks in STDOUT
            done = True

        rewards_list.append(reward)
        log_step(
            step=step, action=action_str, reward=reward, done=done, error=error_msg
        )

        if done or step > 100:
            success = task_score >= 0.1
            log_end(success=success, steps=step, score=task_score, rewards=rewards_list)

            avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
            avg_up = sum(uptimes) / len(uptimes) if uptimes else 0.0

            return {
                "task": task_id,
                "score": task_score,
                "avg_latency": avg_lat,
                "avg_uptime": avg_up,
                "total_steps": step,
            }


def main():
    log_path = get_log_path(MODEL_NAME)
    all_task_stats = []

    with tee_output(log_path):
        print(f"==================================================")
        print(f"  STARTING DIME EVALUATION SESSION: {MODEL_NAME}")
        print(f"==================================================")

        for task_id in TASKS:
            stats = run_task(task_id, MODEL_NAME)
            all_task_stats.append(stats)

        # -----------------------------------------------------------
        # FINAL STRUCTURED SUMMARY FOR JUDGES
        # -----------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"FINAL PERFORMANCE SUMMARY: {MODEL_NAME}")
        print("=" * 80)
        print(
            f"{'Task Name':<25} | {'Task Score':<12} | {'Uptime %':<12} | {'Avg Latency (ms)':<15}"
        )
        print("-" * 80)

        for s in all_task_stats:
            print(
                f"{s['task']:<25} | {s['score']:<12.4f} | {s['avg_uptime']:<12.1f} | {s['avg_latency']:<15.1f}"
            )

        # Calculate Overarching Performance Metric (DIME Index)
        # Using a weighted calculation: (Avg Task Score * Avg Uptime) / log(Safe Latency)
        overall_score = sum(s["score"] for s in all_task_stats) / max(
            len(all_task_stats), 1
        )
        avg_system_uptime = sum(s["avg_uptime"] for s in all_task_stats) / max(
            len(all_task_stats), 1
        )

        # Ensure latency > 1 to avoid ZeroDivisionError or negative logs
        avg_system_latency = sum(s["avg_latency"] for s in all_task_stats) / max(
            len(all_task_stats), 1
        )
        safe_lat = max(avg_system_latency, 2.0)

        dime_index = (overall_score * avg_system_uptime) / math.log(safe_lat)

        print("-" * 80)
        print(f"Overall Task Completion Score : {overall_score:.4f}")
        print(f"Mean System Uptime            : {avg_system_uptime:.1f}%")
        print(f"Mean System Latency           : {avg_system_latency:.1f} ms")
        print("=" * 80)
        print(f"OVERALL DIME INDEX (Higher=Better): {dime_index:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
