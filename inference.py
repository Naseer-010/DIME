#!/usr/bin/env python3
"""
LLM Agent Inference Loop for Distributed Infrastructure Environment (DIME).

Evaluates the autonomous reasoning capabilities of LLMs for SRE tasks.
Features:
  - Dual-mode inference: local GPU (Transformers) and remote endpoint (OpenAI API)
  - Structured JSON-lines logging per episode + legacy console/CSV output
  - Robust CoT reasoning extraction with multi-format fallback
  - Strict action reconstructor to guarantee 0 backend crashes
  - CLI argument support for model, mode, tasks, log directory

Usage:
  python inference.py                                    # defaults
  python inference.py --mode local --model Qwen/Qwen3-8B
  python inference.py --mode endpoint --tasks traffic_spike node_failure
  python inference.py --log-dir /path/to/logs
"""

import argparse
import ast
import csv
import json
import math
import os
import re
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (env vars with CLI override)
# ---------------------------------------------------------------------------
_DEFAULT_API_BASE = "https://router.huggingface.co/v1"
_DEFAULT_MODEL = "Qwen/Qwen3-8B"
_DEFAULT_ENV_URL = "http://localhost:8000"

ALL_TASKS = [
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

# Global states for models and clients (lazy-loaded)
_tokenizer = None
_model = None
_client = None

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
# Structured JSON-lines Logger
# ---------------------------------------------------------------------------
class StructuredLogger:
    """Writes one JSON object per line to a .jsonl file for each episode."""

    def __init__(self, log_dir: Path, model_name: str):
        self.log_dir = log_dir
        self.model_name = model_name
        self._fh = None
        self._path = None

    def start_episode(self, task_id: str) -> Path:
        self.close()
        safe_model = self.model_name.replace("/", "_").replace(".", "_")
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        subdir = self.log_dir / safe_model
        subdir.mkdir(parents=True, exist_ok=True)
        self._path = subdir / f"{task_id}_{ts}.jsonl"
        self._fh = open(self._path, "w", encoding="utf-8", buffering=1)
        self._write_record(
            event="episode_start",
            task_id=task_id,
            model=self.model_name,
            timestamp=ts,
        )
        return self._path

    def log_step(
        self,
        step: int,
        raw_llm_output: str,
        parsed_action: dict,
        backend_action: dict,
        reward: float,
        done: bool,
        task_score: float,
        obs_snapshot: Optional[dict] = None,
        error: Optional[str] = None,
        reasoning: str = "",
    ):
        record = {
            "event": "step",
            "step": step,
            "reasoning": reasoning[:500],
            "raw_llm_output_length": len(raw_llm_output),
            "parsed_action": parsed_action,
            "backend_action": backend_action,
            "reward": round(reward, 4),
            "done": done,
            "task_score": round(task_score, 4),
        }
        if error:
            record["error"] = error
        if obs_snapshot:
            # Store compact summary of observation
            record["obs"] = {
                "latency_ms": obs_snapshot.get("latency_ms"),
                "failed_nodes": obs_snapshot.get("failed_nodes"),
                "request_rate": obs_snapshot.get("request_rate"),
                "error_budget": obs_snapshot.get("error_budget"),
            }
        self._write_record(**record)

    def end_episode(
        self,
        task_id: str,
        success: bool,
        total_steps: int,
        task_score: float,
        rewards: List[float],
    ):
        self._write_record(
            event="episode_end",
            task_id=task_id,
            success=success,
            total_steps=total_steps,
            task_score=round(task_score, 4),
            reward_mean=round(sum(rewards) / max(len(rewards), 1), 4),
            reward_min=round(min(rewards) if rewards else 0.0, 4),
            reward_max=round(max(rewards) if rewards else 0.0, 4),
            reward_std=round(_std(rewards), 4),
        )
        self.close()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _write_record(self, **kwargs):
        if self._fh is not None:
            self._fh.write(json.dumps(kwargs, default=str) + "\n")


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


# ---------------------------------------------------------------------------
# CSV Logging (legacy, kept for backward compatibility)
# ---------------------------------------------------------------------------
def _init_csv(path: Path):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
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


def _log_csv(
    path: Path,
    model: str,
    task_id: str,
    step: int,
    action: str,
    reasoning: str,
    reward: float,
    score: float,
    done: bool,
    error: Optional[str],
):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [model, task_id, step, action, reasoning, reward, score, done, error]
        )


# ---------------------------------------------------------------------------
# Terminal Logging (legacy console output + tee to file)
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


def log_start(task: str, env: str, model: str, mode: str) -> None:
    print(
        f"\n[START] task={task} env={env} model={model} mode={mode.upper()}",
        flush=True,
    )


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
# Model / Client Loading
# ---------------------------------------------------------------------------
def _get_api_key() -> Optional[str]:
    return (
        os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("HF_TOKEN")
    )


def get_client(api_base: str, api_key: str):
    """Initialize the OpenAI client for endpoint inference."""
    global _client
    if _client is None:
        from openai import OpenAI

        print(f"\n[INFO] Connecting to endpoint {api_base}...", flush=True)
        _client = OpenAI(base_url=api_base, api_key=api_key)
    return _client


def load_local_model(model_name: str):
    """Load local models via Hugging Face Transformers to GPU."""
    global _tokenizer, _model
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(
            f"\n[INFO] Loading local model {model_name} via Transformers to GPU...",
            flush=True,
        )
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
    return _tokenizer, _model


# ---------------------------------------------------------------------------
# LLM Response Parser
# ---------------------------------------------------------------------------
def parse_llm_response(text: str) -> Tuple[dict, str]:
    """
    Robustly extracts JSON action and reasoning from LLM output.

    Handles: <think>/<reasoning> blocks, markdown code fences, loose JSON,
    double-escaped strings, and missing XML tags.
    """
    reasoning = "No reasoning provided."
    action_dict: dict = {"action_type": "no_op", "kubectl_command": "no_op"}

    # 1. Extract <think> block as primary reasoning
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()

    # Strip think block before parsing action tags
    text_stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Extract <reasoning> tag as fallback
    if reasoning == "No reasoning provided.":
        res_match = re.search(
            r"<reasoning>\s*(.*?)\s*</reasoning>",
            text_stripped,
            re.IGNORECASE | re.DOTALL,
        )
        if res_match:
            reasoning = res_match.group(1).strip()

    # 3. Extract Action block
    act_match = re.search(
        r"<action>\s*(.*?)\s*</action>", text_stripped, re.IGNORECASE | re.DOTALL
    )
    json_text = act_match.group(1) if act_match else text_stripped

    # 4. Extract JSON from content (handles markdown code fences)
    # Strip markdown code fences
    json_text = re.sub(r"```[a-zA-Z]*", "", json_text)
    json_text = json_text.replace("```", "").strip()

    # Find outermost JSON braces
    start_idx = json_text.find("{")
    end_idx = json_text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        json_text = json_text[start_idx : end_idx + 1]
    else:
        # Try extracting a bare kubectl command
        cmd_match = re.search(r"(kubectl\s+\S+.*?)(?:\n|$)", json_text, re.IGNORECASE)
        if cmd_match:
            json_text = json.dumps({"command": cmd_match.group(1).strip()})
        else:
            json_text = '{"command": "no_op"}'

    # 5. Parse JSON with multiple fallbacks
    try:
        action_dict = json.loads(json_text)
    except json.JSONDecodeError:
        # Try fixing common LLM JSON errors
        try:
            # Handle single quotes
            fixed = json_text.replace("'", '"')
            action_dict = json.loads(fixed)
        except json.JSONDecodeError:
            try:
                action_dict = ast.literal_eval(json_text)
            except Exception:
                # Extract command from raw text as last resort
                cmd_match = re.search(
                    r"(kubectl\s+\S+.*?)(?:\"|'|$)", json_text, re.IGNORECASE
                )
                if cmd_match:
                    action_dict = {"command": cmd_match.group(1).strip()}
                else:
                    action_dict = {"action_type": "no_op", "kubectl_command": "no_op"}

    return action_dict, reasoning


# ---------------------------------------------------------------------------
# Safe Backend Action Builder
# ---------------------------------------------------------------------------
def build_safe_backend_action(action_dict: dict) -> dict:
    """
    STRICT RECONSTRUCTOR: Prevents FastAPI 422 Errors.
    Guarantees the backend only receives valid keys specified in server/models.py.
    """
    if isinstance(action_dict, dict) and (
        "command" in action_dict or "raw_command" in action_dict
    ):
        cmd = str(
            action_dict.get("command") or action_dict.get("raw_command") or "no_op"
        ).strip()
        return {"action_type": "no_op", "raw_command": cmd}

    safe_action = {"action_type": action_dict.get("action_type", "no_op")}
    act_type = safe_action["action_type"]

    if act_type in ("restart_node", "query_logs"):
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

    elif act_type not in ("scale_up", "no_op"):
        safe_action["action_type"] = "no_op"

    return safe_action


# ---------------------------------------------------------------------------
# LLM Decision Maker
# ---------------------------------------------------------------------------
def llm_decide(
    observation: dict,
    model_name: str,
    mode: str,
    api_base: str,
    api_key: Optional[str],
) -> Tuple[dict, str, str]:
    """
    Query the LLM for a decision based on the current observation.

    Returns:
        (action_dict, reasoning, raw_output)
    """
    obs_str = json.dumps(observation)
    user_prompt = f"Current system state:\n{obs_str}\nRespond with the required XML and JSON format."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            content = ""

            # --- LOCAL GPU INFERENCE ---
            if mode == "local":
                import torch

                tokenizer, model = load_local_model(model_name)

                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=4096,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                input_length = inputs.input_ids.shape[1]
                content = tokenizer.decode(
                    outputs[0][input_length:], skip_special_tokens=True
                ).strip()

            # --- REMOTE ENDPOINT INFERENCE ---
            elif mode == "endpoint":
                if not api_key:
                    raise ValueError(
                        "API_KEY is missing for endpoint inference! "
                        "Set API_KEY, OPENAI_API_KEY, or HF_TOKEN."
                    )
                api_client = get_client(api_base, api_key)
                response = api_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.7,
                    top_p=0.9,
                )
                content = response.choices[0].message.content.strip()

            else:
                raise ValueError(
                    f"Unknown inference mode: {mode}. Must be 'local' or 'endpoint'."
                )

            action_dict, reasoning = parse_llm_response(content)

            print("\n[DEBUG RAW OUTPUT]\n", content[:500], "\n", flush=True)

            if (
                "action_type" in action_dict
                or "command" in action_dict
                or "raw_command" in action_dict
            ):
                return action_dict, reasoning, content

        except Exception as e:
            wait_time = 2 ** (attempt + 1)
            print(
                f"[WARNING] Inference call failed (Attempt {attempt + 1}/{MAX_RETRIES}): {str(e)[:200]}",
                flush=True,
            )
            if mode == "endpoint":
                print(f"[DEBUG] Retrying in {wait_time} seconds...", flush=True)
                time.sleep(wait_time)

    print("[ERROR] Inference repeatedly failed. Skipping turn with no_op.", flush=True)
    return (
        {"action_type": "no_op", "kubectl_command": "no_op"},
        "Inference Error - Fallback to no_op",
        "",
    )


# ---------------------------------------------------------------------------
# Environment Communication
# ---------------------------------------------------------------------------
def env_reset(env_url: str, task_id: str) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{env_url}/reset", json={"task": task_id}, timeout=15
            )
            response.raise_for_status()
            payload = response.json()
            data_block = payload.get("data", payload)
            if "observation" in data_block and isinstance(
                data_block["observation"], dict
            ):
                return data_block["observation"]
            return data_block
        except Exception:
            time.sleep(2**attempt)
    raise ConnectionError(f"Failed to reset environment after {MAX_RETRIES} attempts.")


def env_step(env_url: str, action: dict) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{env_url}/step", json={"action": action}, timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            time.sleep(2**attempt)
    raise ConnectionError(f"Failed to step environment after {MAX_RETRIES} attempts.")


# ---------------------------------------------------------------------------
# Task Runner
# ---------------------------------------------------------------------------
def run_task(
    task_id: str,
    *,
    model_name: str,
    mode: str,
    api_base: str,
    api_key: Optional[str],
    env_url: str,
    csv_path: Path,
    structured_logger: StructuredLogger,
    max_episode_steps: int = 100,
) -> dict:
    log_start(task=task_id, env=BENCHMARK, model=model_name, mode=mode)
    episode_path = structured_logger.start_episode(task_id)
    print(f"[INFO] Structured log: {episode_path}", flush=True)

    try:
        obs = env_reset(env_url, task_id)
    except Exception as e:
        print(
            f"[FATAL ERROR] Failed to connect to environment server for task '{task_id}': {e}",
            flush=True,
        )
        log_end(success=False, steps=0, score=0.0, rewards=[])
        structured_logger.end_episode(task_id, False, 0, 0.0, [])
        return {
            "task": task_id,
            "score": 0.0,
            "avg_latency": 0.0,
            "avg_uptime": 0.0,
            "total_steps": 0,
        }

    step = 0
    rewards_list: List[float] = []
    latencies: List[float] = []
    uptimes: List[float] = []
    task_score = 0.0

    while True:
        step += 1

        # Track metrics from observation
        lat = float(obs.get("latency_ms", 0.0))
        up = float(obs.get("uptime_pct", 0.0))
        if lat > 0:
            latencies.append(lat)
        if up > 0:
            uptimes.append(up)

        # 1. Get the action dict, reasoning, and raw output
        action_dict, reasoning, raw_output = llm_decide(
            obs, model_name, mode, api_base, api_key
        )

        # 2. Stringify for logging
        action_str = json.dumps(action_dict).replace('"', "'")
        reasoning_clean = reasoning.replace("\n", " ")
        terminal_action_str = f"act={action_str} | rsn='{reasoning_clean[:90]}...'"

        # 3. Build safe backend payload
        backend_action = build_safe_backend_action(action_dict)

        error_msg = None
        reward = 0.0
        done = False

        try:
            result = env_step(env_url, backend_action)
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

        # Log to all outputs
        log_step(
            step=step,
            action=terminal_action_str,
            reward=reward,
            done=done,
            error=error_msg,
        )
        _log_csv(
            csv_path,
            model_name,
            task_id,
            step,
            action_str,
            reasoning,
            reward,
            task_score,
            done,
            error_msg,
        )
        structured_logger.log_step(
            step=step,
            raw_llm_output=raw_output,
            parsed_action=action_dict,
            backend_action=backend_action,
            reward=reward,
            done=done,
            task_score=task_score,
            obs_snapshot=obs if isinstance(obs, dict) else None,
            error=error_msg,
            reasoning=reasoning,
        )

        if done or step > max_episode_steps:
            success = task_score >= 0.1
            log_end(success=success, steps=step, score=task_score, rewards=rewards_list)
            structured_logger.end_episode(
                task_id, success, step, task_score, rewards_list
            )

            avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
            avg_up = sum(uptimes) / len(uptimes) if uptimes else 0.0

            return {
                "task": task_id,
                "score": task_score,
                "avg_latency": avg_lat,
                "avg_uptime": avg_up,
                "total_steps": step,
            }


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DIME LLM Agent Inference Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["local", "endpoint"],
        default=os.environ.get("INFERENCE_MODE", "endpoint").lower(),
        help="Inference mode: local GPU or remote endpoint (default: endpoint)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME", _DEFAULT_MODEL),
        help=f"Model name/path (default: {_DEFAULT_MODEL})",
    )
    p.add_argument(
        "--api-base",
        default=os.environ.get("API_BASE_URL", _DEFAULT_API_BASE),
        help="API base URL for endpoint mode",
    )
    p.add_argument(
        "--env-url",
        default=os.environ.get("ENV_SERVER_URL", _DEFAULT_ENV_URL),
        help="Environment server URL",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Tasks to run (default: all tasks)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Max steps per episode (default: 100)",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for structured JSON logs (default: ./logs/)",
    )
    return p


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    model_name = args.model
    mode = args.mode
    api_base = args.api_base
    api_key = _get_api_key()
    env_url = args.env_url
    tasks = args.tasks or ALL_TASKS
    max_steps = args.max_steps

    # Setup paths
    project_root = Path(__file__).resolve().parent
    safe_model_name = model_name.replace("/", "_").replace(".", "_")
    log_file = project_root / f"logs_{safe_model_name}.txt"
    csv_file = project_root / f"metrics_{safe_model_name}.csv"
    log_dir = Path(args.log_dir) if args.log_dir else project_root / "logs"

    _init_csv(csv_file)
    structured_logger = StructuredLogger(log_dir, model_name)

    print(f"API KEY LOADED: {api_key[:10] if api_key else 'Missing or Running Local!'}")

    all_task_stats = []

    with tee_output(log_file):
        print("==================================================")
        print(f"  STARTING DIME EVALUATION SESSION: {model_name}")
        print(f"  MODE: {mode.upper()}")
        print(f"  TASKS: {len(tasks)}")
        print(f"  STRUCTURED LOGS: {log_dir}")
        print("==================================================")

        for task_id in tasks:
            stats = run_task(
                task_id,
                model_name=model_name,
                mode=mode,
                api_base=api_base,
                api_key=api_key,
                env_url=env_url,
                csv_path=csv_file,
                structured_logger=structured_logger,
                max_episode_steps=max_steps,
            )
            all_task_stats.append(stats)

        # -----------------------------------------------------------
        # FINAL STRUCTURED SUMMARY FOR JUDGES
        # -----------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"FINAL PERFORMANCE SUMMARY: {model_name}")
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
        print(f"Logs saved to: {log_file}")
        print(f"Metrics saved to: {csv_file}")
        print(f"Structured logs: {log_dir}")

    structured_logger.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
