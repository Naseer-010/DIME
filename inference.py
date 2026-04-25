##!/usr/bin/env python3
"""
LLM Agent Inference Loop for Distributed Infrastructure Environment (DIME).

Evaluates the autonomous reasoning capabilities of LLMs for SRE tasks.
Features CoT reasoning, exponential backoff retries, CSV metric logging,
and hybrid JSON/kubectl action outputs.
"""

import json
import os
import sys
import time
import traceback
import math
import csv
import re
from contextlib import contextmanager
from pathlib import Path
import requests
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = (
    os.environ.get("API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("HF_TOKEN")
)
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")

# Dynamic logging paths
safe_model_name = MODEL_NAME.replace("/", "_").replace(".", "_")
LOG_FILE = Path(__file__).resolve().parent / f"logs_{safe_model_name}.txt"
METRICS_FILE = Path(__file__).resolve().parent / f"metrics_{safe_model_name}.csv"

TASKS = [
    "traffic_spike",
    "node_failure",
    "cascading_failure",
    "flash_crowd",
    "level_5_alibaba_trace",
]
MAX_RETRIES = 4
BENCHMARK = "distributed_infra_env"

client: Optional[OpenAI] = None

# THE NEW PROMPT: No cheat-sheets. Just the environment rules and objectives.
SYSTEM_PROMPT = """You are an autonomous AI Site Reliability Engineer (SRE) managing a distributed cluster.
You receive observations about the system state as JSON and must decide the best action to maintain system health.

CLUSTER ARCHITECTURE:
- Node 0 is the DATABASE — a single point of failure. If it fails, all processing stops.
- Nodes 1-7 are APP SERVERS.
- Scaling up adds new nodes, but they have a 3-step cold start.

YOUR OBJECTIVES:
1. Maximize Uptime: Prevent node failures by reducing extreme CPU loads and restarting failed nodes immediately.
2. Minimize Latency: Keep average latency low by scaling up or throttling when necessary.
3. Conserve Budget: Do not scale up unnecessarily. Cloud budget is limited.
4. Maintain Throughput: Do not throttle traffic unnecessarily. Dropping too much traffic causes a massive penalty.

AVAILABLE ACTIONS (Must output exactly one of these JSON objects):
1. Restart a failed node: {"action_type": "restart_node", "target": <node_index>, "kubectl_command": "kubectl delete pod node-<node_index>"}
2. Scale up capacity: {"action_type": "scale_up", "kubectl_command": "kubectl scale deployment frontend --replicas=10"}
3. Reroute traffic: {"action_type": "reroute_traffic", "from_node": <index>, "to_node": <index>, "kubectl_command": "kubectl exec -it istio-proxy -- traffic shift --from=<from_node> --to=<to_node>"}
4. Throttle ingress: {"action_type": "throttle", "rate": <float between 0.3 and 1.0>, "kubectl_command": "kubectl throttle ingress --rate=<rate>"}
5. Do nothing: {"action_type": "no_op", "kubectl_command": "no_op"}

Respond using the following STRICT format. 
<reasoning>Analyze the current metrics, identify risks, and explain your strategy based on the objectives.</reasoning>
<action>
{"action_type": "...", "kubectl_command": "..."}
</action>"""


# ---------------------------------------------------------------------------
# CSV Logging Setup for Graphs
# ---------------------------------------------------------------------------
def init_metrics_file():
    if not METRICS_FILE.exists():
        with open(METRICS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "model",
                    "task_id",
                    "step",
                    "action_taken",
                    "reward",
                    "cumulative_score",
                    "done",
                    "error",
                ]
            )


init_metrics_file()


def log_to_csv(
    task_id: str,
    step: int,
    action: str,
    reward: float,
    score: float,
    done: bool,
    error: str,
):
    with open(METRICS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([MODEL_NAME, task_id, step, action, reward, score, done, error])


# ---------------------------------------------------------------------------
# Terminal Logging Functions
# ---------------------------------------------------------------------------
class TeeStream:
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
        raise ValueError("API_KEY environment variable is missing!")
    if client is None:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return client


def extract_json_action(text: str) -> dict:
    """Safely extracts the JSON object from the <action> tags."""
    try:
        match = re.search(r"<action>\s*({.*?})\s*</action>", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        match = re.search(r'({[^{}]*"action_type"[^{}]*})', text)
        if match:
            return json.loads(match.group(1))

    except json.JSONDecodeError:
        pass

    return {"action_type": "no_op", "kubectl_command": "no_op"}


def llm_decide(observation: dict) -> dict:
    obs_str = json.dumps(observation)
    user_prompt = f"Current system state:\n{obs_str}\nRespond with the required XML and JSON format."

    for attempt in range(MAX_RETRIES):
        try:
            response = get_client().chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=350,
                temperature=0.01,
            )
            content = response.choices[0].message.content.strip()
            parsed_action = extract_json_action(content)

            if "action_type" in parsed_action:
                return parsed_action

        except Exception as e:
            # Exponential Backoff (2s, 4s, 8s, 16s)
            wait_time = 2 ** (attempt + 1)
            print(
                f"[WARNING] API call failed (Attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}",
                flush=True,
            )
            print(f"[DEBUG] Retrying in {wait_time} seconds...", flush=True)
            time.sleep(wait_time)

    print("[ERROR] API repeatedly failed. Skipping turn with no_op.", flush=True)
    return {"action_type": "no_op", "kubectl_command": "no_op"}


def env_reset(task_id: str) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{ENV_SERVER_URL}/reset", json={"task": task_id}, timeout=15
            )
            response.raise_for_status()
            payload = response.json()
            data_block = payload.get("data", payload)
            if "observation" in data_block and isinstance(
                data_block["observation"], dict
            ):
                return data_block["observation"]
            return data_block
        except Exception as e:
            time.sleep(2**attempt)
    raise ConnectionError(f"Failed to reset environment after {MAX_RETRIES} attempts.")


def env_step(action: dict) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{ENV_SERVER_URL}/step", json={"action": action}, timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            time.sleep(2**attempt)
    raise ConnectionError(f"Failed to step environment after {MAX_RETRIES} attempts.")


def run_task(task_id: str) -> dict:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_reset(task_id)
    except Exception as e:
        print(
            f"[FATAL ERROR] Failed to connect to environment server for task '{task_id}': {e}",
            flush=True,
        )
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

        # Track metrics
        lat = float(obs.get("latency_ms", 0.0))
        up = float(obs.get("uptime_pct", 0.0))
        if lat > 0:
            latencies.append(lat)
        if up > 0:
            uptimes.append(up)

        # 1. Get the full combined dictionary from the LLM
        action = llm_decide(obs)

        # 2. Stringify the FULL action (including kubectl_command) for perfect logs
        action_str = json.dumps(action).replace('"', "'")

        # 3. Strip out 'kubectl_command' so Pydantic backend doesn't crash with a 422
        backend_action = {k: v for k, v in action.items() if k != "kubectl_command"}

        error_msg = None
        reward = 0.0
        done = False

        try:
            # Send the clean version to the server
            result = env_step(backend_action)
            data_block = result.get("data", result)

            if "observation" in data_block and isinstance(
                data_block["observation"], dict
            ):
                obs = data_block["observation"]
            else:
                obs = data_block

            reward = float(data_block.get("reward", obs.get("reward", 0.0)))
            done = bool(data_block.get("done", obs.get("done", False)))
            task_score = float(obs.get("task_score", 0.0))

        except Exception as e:
            error_msg = str(e).replace("\n", " ")
            done = True

        rewards_list.append(reward)

        # Write to outputs using the FULL action string
        log_step(
            step=step, action=action_str, reward=reward, done=done, error=error_msg
        )
        log_to_csv(task_id, step, action_str, reward, task_score, done, error_msg)

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
    print(f"API KEY LOADED:", API_KEY[:10] if API_KEY else "Missing!")

    all_task_stats = []

    with tee_output(LOG_FILE):
        print(f"==================================================")
        print(f"  STARTING DIME EVALUATION SESSION: {MODEL_NAME}")
        print(f"==================================================")

        for task_id in TASKS:
            stats = run_task(task_id)
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

        # Calculate DIME Index
        overall_score = sum(s["score"] for s in all_task_stats) / max(
            len(all_task_stats), 1
        )
        avg_system_uptime = sum(s["avg_uptime"] for s in all_task_stats) / max(
            len(all_task_stats), 1
        )
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
        print(f"Logs saved to: {LOG_FILE}")
        print(f"Metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
