#!/usr/bin/env python3
"""
LLM Agent Inference Loop for Distributed Infrastructure Environment (DIME).

Evaluates the autonomous reasoning capabilities of LLMs for SRE tasks.
Features robust CoT reasoning extraction, exponential backoff retries,
CSV metric logging, and a strict action reconstructor to guarantee 0 crashes.
"""

import json
import os
import sys
import time
import traceback
import math
import csv
import re
import ast
from contextlib import contextmanager
from pathlib import Path
import requests
from typing import List, Optional, Dict, Tuple

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
    "thundering_herd",
    "zombie_node",
    "memory_leak_slow_burn",
    "split_brain_io_bottleneck",
    "black_swan_az_failure",
    "retry_storm",
    "hot_shard_skew",
    "connection_pool_deadlock",
    "autoscaler_flapping_trap",
]
MAX_RETRIES = 4
BENCHMARK = "distributed_infra_env"

client: Optional[OpenAI] = None

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) managing a highly volatile Kubernetes cluster.
You receive telemetry as JSON and must respond with a SINGLE kubectl command to prevent cascading failure.

CLUSTER ARCHITECTURE & PHYSICS:
- Node 0 is the stateful DATABASE (SPOF). Nodes 1-7 are stateless APP WORKERS.
- NEW METRICS: You now see 'mem_utilizations' (RAM), 'io_wait' (Disk), 'p99_latency' (Tail risk), and 'error_budget'.
- MEMORY CLIFF: If ANY node hits > 0.98 mem_utilization, it suffers an instant OOM Kill. 
- COLD START: Scaling up takes 3 steps to boot.
- ERROR BUDGET: Throttling traffic saves the DB but burns your finite Error Budget.

Available commands:
- kubectl delete pod node-<ID>           → restart a node (clears memory leaks and deadlocks)
- kubectl scale deployment frontend --replicas=10  → scale up (takes 3 steps to boot)
- kubectl exec -it istio-proxy -- traffic shift --from=<ID> --to=<ID>  → reroute traffic away from a bad node
- kubectl throttle ingress --rate=<float>  → drop traffic (0.0 to 1.0). Burns error budget!
- kubectl logs node-<ID>                 → investigate telemetry timeout
- no_op                                  → do nothing

CRITICAL INCIDENT TRIAGE TREE (Follow strictly in order):
1. OOM IMMINENT (Memory Leak): IF ANY 'mem_utilizations' > 0.92:
   IMMEDIATELY output: kubectl delete pod node-5 (or whichever node is leaking. Scaling does NOT fix memory leaks!)
   
2. SPLIT-BRAIN (Disk I/O Bottleneck): IF node_0 'io_wait' > 0.80:
   Output: kubectl throttle ingress --rate=0.5 (Do NOT scale up; more workers will lock the DB disk further).

3. HOT SHARD (Load Balancer Skew): IF one worker's CPU > 0.90 but the cluster average is low:
   Output: kubectl exec -it istio-proxy -- traffic shift --from=<high_cpu_node> --to=<low_cpu_node>

4. RETRY STORM / THUNDERING HERD: IF 'p99_latency' > 100.0 AND traffic is spiking:
   Output: kubectl throttle ingress --rate=0.4 (Break the exponential retry loop).

5. CONNECTION DEADLOCK (Zombie Node): IF a worker's CPU is incredibly low (< 0.10) BUT 'p99_latency' is huge:
   Output: kubectl exec -it istio-proxy -- traffic shift --from=<zombie_node> --to=<healthy_node>

6. BLACK SWAN (Multi-Node Death): IF multiple nodes are in 'failed_nodes':
   Output: kubectl throttle ingress --rate=0.3 (Shed load to protect survivors while you recover).

7. DATABASE SURVIVAL: IF node-0 (DB) cpu_load > 0.80:
   Output: kubectl throttle ingress --rate=0.7

8. SAFE SCALING: IF avg worker CPU > 0.75 AND 'error_budget' > 20:
   Output: kubectl scale deployment frontend --replicas=10

9. HEALTHY / FLAPPING TRAP: If metrics are stable or oscillating slightly:
   Output: no_op

Respond using the following STRICT format. You must include the XML reasoning tags:
<reasoning>Diagnose the telemetry. Identify which of the 9 Triage rules applies.</reasoning>
<action>
{"command": "your_kubectl_command_or_no_op_here"}
</action>"""


# ---------------------------------------------------------------------------
# CSV Logging Setup for Graphs
# ---------------------------------------------------------------------------
def init_metrics_file():
    if not METRICS_FILE.exists():
        with open(METRICS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "model",
                    "task_id",
                    "step",
                    "action_taken",
                    "reasoning",
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
    reasoning: str,
    reward: float,
    score: float,
    done: bool,
    error: str,
):
    with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [MODEL_NAME, task_id, step, action, reasoning, reward, score, done, error]
        )


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


def parse_llm_response(text: str) -> Tuple[dict, str]:
    """Robustly extracts JSON action and reasoning, handling markdown and loose formatting."""
    reasoning = "No reasoning provided."
    action_dict = {"action_type": "no_op", "kubectl_command": "no_op"}

    # Extract reasoning safely
    res_match = re.search(
        r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.IGNORECASE | re.DOTALL
    )
    if res_match:
        reasoning = res_match.group(1).strip()

    # Extract Action block safely
    act_match = re.search(
        r"<action>\s*(.*?)\s*</action>", text, re.IGNORECASE | re.DOTALL
    )
    json_text = act_match.group(1) if act_match else text

    # Strip everything outside the outermost JSON brackets if tags failed
    start_idx = json_text.find("{")
    end_idx = json_text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        json_text = json_text[start_idx : end_idx + 1]
    else:
        json_text = '{"command": "no_op"}'  # Total failure fallback

    # Clean up markdown code blocks if the LLM added them
    json_text = re.sub(r"```[a-zA-Z]*", "", json_text)
    json_text = json_text.replace("```", "").strip()

    try:
        try:
            action_dict = json.loads(json_text)
        except json.JSONDecodeError:
            action_dict = ast.literal_eval(json_text)
    except Exception as e:
        print(f"[DEBUG] Parser failed to extract action: {str(e)}", flush=True)

    return action_dict, reasoning


def build_safe_backend_action(action_dict: dict) -> dict:
    """
    STRICT RECONSTRUCTOR: Prevents FastAPI 422 Errors.
    Guarantees the backend only receives valid keys specified in server/models.py.
    Prioritizes 'raw_command' syntax to allow the backend's regex parser to do the heavy lifting.
    """
    # 1. If the LLM successfully formatted the new prompt, it output a "command" key.
    # We must translate this to "raw_command" for the backend parser and satisfy Pydantic.
    if isinstance(action_dict, dict) and (
        "command" in action_dict or "raw_command" in action_dict
    ):
        cmd = str(
            action_dict.get("command") or action_dict.get("raw_command") or "no_op"
        ).strip()
        return {"action_type": "no_op", "raw_command": cmd}

    # 2. Fallback: If the LLM hallucinates an old schema (e.g., outputs {"action_type": "throttle", "rate": 0.5})
    # We map it safely.
    safe_action = {"action_type": action_dict.get("action_type", "no_op")}
    act_type = safe_action["action_type"]

    if act_type == "restart_node" or act_type == "query_logs":
        try:
            safe_action["target"] = int(action_dict.get("target", 0))
        except (ValueError, TypeError):
            safe_action["target"] = 0

    elif act_type == "reroute_traffic":
        try:
            safe_action["from_node"] = int(action_dict.get("from_node", 0))
            safe_action["to_node"] = int(action_dict.get("to_node", 0))
        except (ValueError, TypeError):
            safe_action["action_type"] = "no_op"

    elif act_type == "throttle":
        try:
            raw_rate = float(action_dict.get("rate", 1.0))
            safe_action["rate"] = max(0.0, min(1.0, raw_rate))
        except (ValueError, TypeError):
            safe_action["rate"] = 1.0

    elif act_type not in ["scale_up", "no_op"]:
        safe_action["action_type"] = "no_op"

    return safe_action


def llm_decide(observation: dict) -> Tuple[dict, str]:
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
                max_tokens=400,
                temperature=0.01,
            )
            content = response.choices[0].message.content.strip()
            action_dict, reasoning = parse_llm_response(content)

            if "action_type" in action_dict or "command" in action_dict:
                return action_dict, reasoning

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
    return {
        "action_type": "no_op",
        "kubectl_command": "no_op",
    }, "API Error - Fallback to no_op"


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

        # 1. Get the action dict and the reasoning string
        action_dict, reasoning = llm_decide(obs)

        # 2. Stringify full action for CSV & format clean string for terminal
        action_str = json.dumps(action_dict).replace('"', "'")
        reasoning_clean = reasoning.replace("\n", " ")
        terminal_action_str = f"act={action_str} | rsn='{reasoning_clean[:90]}...'"

        # 3. Use the strict reconstructor to get the perfect backend payload
        backend_action = build_safe_backend_action(action_dict)

        error_msg = None
        reward = 0.0
        done = False

        try:
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

        # Log to outputs
        log_step(
            step=step,
            action=terminal_action_str,
            reward=reward,
            done=done,
            error=error_msg,
        )
        log_to_csv(
            task_id, step, action_str, reasoning, reward, task_score, done, error_msg
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
