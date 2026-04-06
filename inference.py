#!/usr/bin/env python3
"""
LLM Agent Inference Loop for Distributed Infrastructure Environment.

Runs the LLM agent against all 3 graded tasks and reports scores.
Uses the OpenAI Python Client to interact with the HuggingFace Inference API.

Environment variables:
    API_BASE_URL — inference endpoint (default: HF Inference API)
    MODEL_NAME   — model ID (default: meta-llama/Llama-3.1-8B-Instruct)
    HF_TOKEN     — HuggingFace API token
"""

import json
import os
import sys
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://router.huggingface.co/v1",  
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

# Initialize the OpenAI client for LLM calls
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "hf_placeholder"
)

# ---------------------------------------------------------------------------
# System prompt for the LLM agent (GOD-MODE)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE).
You receive observations about the system state as JSON and must respond with a single action as JSON.

Available actions:
- {"action_type": "restart_node", "target": <int>}
- {"action_type": "reroute_traffic", "from_node": <int>, "to_node": <int>}
- {"action_type": "scale_up"}
- {"action_type": "throttle", "rate": <float>}
- {"action_type": "no_op"}

CRITICAL DECISION TREE (Follow strictly):
1. IF 'failed_nodes' is not empty: 
   IMMEDIATELY output {"action_type": "restart_node", "target": <failed_node_index>}
2. IF any node in 'cpu_loads' is > 0.85:
   Find the node with the highest CPU (e.g., node 4) and the node with the lowest CPU (e.g., node 1). 
   Output {"action_type": "reroute_traffic", "from_node": 4, "to_node": 1}
3. IF average 'cpu_loads' > 0.70 AND no nodes are failing:
   Output {"action_type": "scale_up"}
4. IF 'latency_ms' > 45.0:
   Output {"action_type": "throttle", "rate": 0.8}
5. IF none of the above are true:
   Output {"action_type": "no_op"}

Respond with ONLY a valid JSON action object. No markdown formatting, and no other text."""


# ---------------------------------------------------------------------------
# LLM Decision Function
# ---------------------------------------------------------------------------

def llm_decide(observation: dict) -> dict:
    """Send observation to LLM and get an action decision."""
    obs_str = json.dumps(observation, indent=2)

    user_prompt = f"""Current system state:
{obs_str}

Analyze the system state and decide the best action. Respond with ONLY a JSON action object."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,
                temperature=0.01, # Extremely low temperature to prevent hallucination
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response (handle markdown code blocks if the LLM hallucinates them)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            action = json.loads(content)
            return action

        except Exception as e:
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
    payload = response.json()
    
    # ROBUST PAYLOAD UNWRAPPING
    data_block = payload.get("data", payload)
    if "observation" in data_block and isinstance(data_block["observation"], dict):
        return data_block["observation"]
    return data_block


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

    # --- REQUIRED TAG FOR AUTO-EVALUATOR ---
    print(f"[START] Task {task_id} initialized.")

    obs = env_reset(task_id)
    print(f"  Initial observation: {len(obs.get('cpu_loads', []))} nodes")
    print(f"  Task hint: {obs.get('task_hint', 'N/A')[:80]}...")

    total_reward = 0.0
    step = 0

    while True:
        # LLM decides action
        action = llm_decide(obs)
        action_type = action.get("action_type", "no_op")

        # --- REQUIRED TAG FOR AUTO-EVALUATOR ---
        print(f"[STEP] {json.dumps(action)}")

        # Execute action
        result = env_step(action)

        # ROBUST PAYLOAD UNWRAPPING
        data_block = result.get("data", result)
        
        if "observation" in data_block and isinstance(data_block["observation"], dict):
            obs = data_block["observation"]
        else:
            obs = data_block

        reward = data_block.get("reward", obs.get("reward", 0.0))
        done = data_block.get("done", obs.get("done", False))
        metadata = data_block.get("metadata", obs.get("metadata", {}))

        total_reward += reward if reward else 0.0
        step += 1

        # Progress logging (for human debugging)
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
            
            # --- REQUIRED TAG FOR AUTO-EVALUATOR ---
            print(f"[END] Task {task_id} completed. Final Score: {task_score:.4f}")
            return task_score

        if step > 100:  # Safety limit
            print("  [!] Safety limit reached, ending episode")
            task_score = metadata.get("task_score", 0.0)
            
            # --- REQUIRED TAG FOR AUTO-EVALUATOR ---
            print(f"[END] Task {task_id} completed (TIMEOUT). Final Score: {task_score:.4f}")
            return task_score


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
            print(f"[END] Task {task_id} failed with error. Final Score: 0.0000")

    # Final report
    print("\n" + "=" * 60)
    print("  FINAL SCORES")
    print("=" * 60)
    for task_id, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_id:25s}  {bar}  {score:.4f}")

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  Average score: {avg:.4f}")
    print("=" * 60)

    # Output structured JSON for automated evaluation
    print("\n" + json.dumps(scores, indent=2))

    return scores


if __name__ == "__main__":
    main()