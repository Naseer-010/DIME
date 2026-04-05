#!/usr/bin/env python3
"""
LLM Agent Inference Loop for Distributed Infrastructure Environment.

Runs the LLM agent against all 3 graded tasks and reports scores.
Uses the OpenAI-compatible client pointing at HuggingFace Inference API.

Environment variables:
    API_BASE_URL — inference endpoint (default: HF Inference API)
    MODEL_NAME   — model ID (default: meta-llama/Llama-3.1-8B-Instruct)
    HF_TOKEN     — HuggingFace API token
"""

import json
import os
import time

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://router.huggingface.co/hf-inference/models",
)
MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "meta-llama/Llama-3.1-8B-Instruct",
)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Environment server URL
ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")

TASKS = ["traffic_spike", "node_failure", "cascading_failure"]
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# System prompt for the LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) managing a distributed compute cluster.

You receive observations about the system state as JSON and must respond with a single action as JSON.

Available actions:
1. {"action_type": "restart_node", "target": <node_index>}  — Restart a failed node (takes 2 steps to come online)
2. {"action_type": "reroute_traffic", "from_node": <source>, "to_node": <dest>}  — Shift 30% of load from source to destination
3. {"action_type": "scale_up"}  — Add a temporary capacity node for 10 steps
4. {"action_type": "throttle", "rate": <0.0-1.0>}  — Reduce incoming request acceptance rate
5. {"action_type": "no_op"}  — Do nothing this step

Key rules:
- A node at >90% CPU for 3 consecutive steps will FAIL — act before this happens
- Failed node load cascades to neighbors, potentially causing chain failures
- Restarting takes 2 steps — plan ahead
- Minimize unnecessary actions (over-intervention is penalized)
- Read the task_hint carefully for specific objectives

Respond with ONLY a valid JSON action object, no other text."""


# ---------------------------------------------------------------------------
# LLM Decision Function
# ---------------------------------------------------------------------------


def llm_decide(observation: dict) -> dict:
    """Send observation to LLM and get an action decision."""
    obs_str = json.dumps(observation, indent=2)

    user_prompt = f"""Current system state:
{obs_str}

Analyze the system state and decide the best action. Respond with ONLY a JSON action object."""

    url = f"{API_BASE_URL}/{MODEL_NAME}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.3,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"].strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            action = json.loads(content)
            return action

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"  [Attempt {attempt + 1}] LLM call failed: {e}")
            if attempt == MAX_RETRIES - 1:
                print("  Falling back to no_op")
                return {"action_type": "no_op"}
            time.sleep(1)

    return {"action_type": "no_op"}


# ---------------------------------------------------------------------------
# Environment Interaction
# ---------------------------------------------------------------------------


def env_reset(task_id: str) -> dict:
    """Reset the environment with a specific task."""
    response = requests.post(
        f"{ENV_SERVER_URL}/reset",
        json={"task": task_id},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("observation", data)


def env_step(action: dict) -> dict:
    """Execute an action in the environment."""
    response = requests.post(
        f"{ENV_SERVER_URL}/step",
        json={"action": action},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------


def run_task(task_id: str) -> float:
    """Run one task and return the final score."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}")

    obs = env_reset(task_id)
    print(f"  Initial observation: {len(obs.get('cpu_loads', []))} nodes")
    print(f"  Task hint: {obs.get('task_hint', 'N/A')[:80]}...")

    total_reward = 0.0
    step = 0

    while True:
        action = llm_decide(obs)
        action_type = action.get("action_type", "no_op")

        result = env_step(action)

        obs = result.get("observation", result)
        reward = result.get("reward", obs.get("reward", 0.0))
        done = result.get("done", obs.get("done", False))
        metadata = obs.get("metadata", {})

        total_reward += reward if reward else 0.0
        step += 1

        cpu_loads = obs.get("cpu_loads", [])
        avg_cpu = sum(cpu_loads) / len(cpu_loads) if cpu_loads else 0
        failed = obs.get("failed_nodes", [])
        latency = obs.get("latency_ms", 0)

        print(
            f"  Step {step:3d} | Action: {action_type:15s} | "
            f"Avg CPU: {avg_cpu:.2f} | Failed: {len(failed)} | "
            f"Latency: {latency:6.1f}ms | Reward: {reward:+.4f}"
        )

        if done:
            task_score = metadata.get("task_score", 0.0)
            print(f"\n  Episode complete!")
            print(f"  Total reward:  {total_reward:+.4f}")
            print(f"  Task score:    {task_score:.4f}")
            return task_score

        if step > 100:
            print("  [!] Safety limit reached, ending episode")
            return metadata.get("task_score", 0.0)


def main():
    """Run all tasks and report final scores."""
    print("\n" + "=" * 60)
    print("  Distributed Infrastructure Management — LLM Agent")
    print("=" * 60)
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Server: {ENV_SERVER_URL}")
    print(f"  LLM:    {API_BASE_URL}")

    if not HF_TOKEN:
        print("\n  [WARNING] HF_TOKEN not set. LLM calls may fail.")
        print("  Set: export HF_TOKEN=hf_xxxxx")

    scores = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"\n  [ERROR] Task {task_id} failed: {e}")
            scores[task_id] = 0.0

    print("\n" + "=" * 60)
    print("  FINAL SCORES")
    print("=" * 60)
    for task_id, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_id:25s}  {bar}  {score:.4f}")

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  Average score: {avg:.4f}")
    print("=" * 60)

    print("\n" + json.dumps(scores, indent=2))

    return scores


if __name__ == "__main__":
    main()
