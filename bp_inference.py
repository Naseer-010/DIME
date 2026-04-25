#!/usr/bin/env python3
"""
LLM Agent Inference Loop for Distributed Infrastructure Environment.

Runs the LLM agent against all graded tasks and reports scores.
Strictly adheres to the Meta-PyTorch OpenEnv Hackathon STDOUT format.
"""

import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
import requests
from typing import List, Optional
from openai import OpenAI
import csv

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get(
    "API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
)
API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GEMINI_URL")
    or os.environ.get("OPENAI_API_KEY")
)
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash")

LOG_FILE = Path(__file__).resolve().parent / f"gemini_{MODEL_NAME}.txt"
METRICS_FILE = Path(__file__).resolve().parent / f"training_metrics_{MODEL_NAME}.csv"
ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")

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
You receive observations about the system state as JSON and must respond with a command.

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
5. IF any app server cpu_load > 0.85:
   Find the node with highest CPU and lowest CPU among app servers.
   Output: kubectl exec -it istio-proxy -- traffic shift --from=<high> --to=<low>
6. IF average cpu_loads > 0.70 AND cloud_budget > 0:
   Output: kubectl scale deployment frontend --replicas=10
7. IF 'latency_ms' > 45.0:
   Output: kubectl throttle ingress --rate=0.8
8. IF none of the above are true:
   Output: no_op

Respond using the following STRICT format. You must include the XML reasoning tags:
<reasoning>Briefly explain your diagnostic thought process here.</reasoning>
<action>
{"command": "your_kubectl_command_or_no_op_here"}
</action>"""

# ---------------------------------------------------------------------------
# CSV Logging Setup for Graphs
# ---------------------------------------------------------------------------
def init_metrics_file():
    if not METRICS_FILE.exists():
        with open(METRICS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'task_id', 'step', 'action_taken', 'reward', 'cumulative_score', 'done', 'error'])

init_metrics_file()

def log_to_csv(episode: int, task_id: str, step: int, action: str, reward: float, score: float, done: bool, error: str):
    with open(METRICS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, task_id, step, action, reward, score, done, error])

# ---------------------------------------------------------------------------
# Required OpenEnv Terminal Logging Functions
# ---------------------------------------------------------------------------

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
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    global client
    if not API_KEY:
        raise ValueError(
            "Missing API key. Set GEMINI_API_KEY, GOOGLE_API_KEY, GEMINI_URL, or OPENAI_API_KEY."
        )
    if client is None:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return client

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
                max_tokens=300, # Increased to allow space for <reasoning>
                temperature=0.01,
            )
            content = (response.choices[0].message.content or "").strip()
            
            # THE 422 FIX:
            # We send the ENTIRE raw string to the backend so `command_parser.py` can extract the <reasoning>.
            # We inject "action_type": "no_op" just to bypass the strict Pydantic schema validation.
            return {
                "action_type": "no_op",
                "raw_command": content
            }
            
        except Exception as e:
            print(f"[DEBUG] LLM call attempt {attempt + 1} failed: {str(e)}", flush=True)
            time.sleep(1)

    return {"action_type": "no_op"}

def env_reset(task_id: str) -> dict:
    response = requests.post(f"{ENV_SERVER_URL}/reset", json={"task": task_id}, timeout=10)
    response.raise_for_status()
    payload = response.json()
    data_block = payload.get("data", payload)
    if "observation" in data_block and isinstance(data_block["observation"], dict):
        return data_block["observation"]
    return data_block

def env_step(action: dict) -> dict:
    response = requests.post(f"{ENV_SERVER_URL}/step", json={"action": action}, timeout=10)
    response.raise_for_status()
    return response.json()

global_episode_count = 0

def run_task(task_id: str) -> float:
    global global_episode_count
    global_episode_count += 1
    
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_reset(task_id)
    except Exception as e:
        print(f"[FATAL ERROR] Failed to connect to environment server for task '{task_id}': {e}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    step = 0
    rewards_list = []
    task_score = 0.0

    while True:
        step += 1
        action = llm_decide(obs)
        action_str = json.dumps(action).replace('"', "'")

        error_msg = None
        reward = 0.0
        done = False

        try:
            result = env_step(action)
            data_block = result.get("data", result)

            if "observation" in data_block and isinstance(data_block["observation"], dict):
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
        
        log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
        log_to_csv(global_episode_count, task_id, step, action_str, reward, task_score, done, error_msg)

        if done or step > 100:
            success = task_score >= 0.1
            log_end(success=success, steps=step, score=task_score, rewards=rewards_list)
            return task_score

def main():
    for task_id in TASKS:
        run_task(task_id)

if __name__ == "__main__":
    with tee_output(LOG_FILE):
        try:
            main()
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
