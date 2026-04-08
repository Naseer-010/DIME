#!/usr/bin/env python3
"""
LLM Agent Inference Loop for Distributed Infrastructure Environment.

Runs the LLM agent against all 3 graded tasks and reports scores.
Strictly adheres to the Meta-PyTorch OpenEnv Hackathon STDOUT format.
"""

import json
import os
import time
import requests
from typing import List, Optional
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

# FIX: API_KEY must be checked FIRST!
API_KEY = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "hf_placeholder"))
ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")

TASKS = ["traffic_spike", "node_failure", "cascading_failure", "flash_crowd"]
MAX_RETRIES = 3
BENCHMARK = "distributed_infra_env"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

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
   Find the node with highest CPU and lowest CPU. 
   Output {"action_type": "reroute_traffic", "from_node": <high>, "to_node": <low>}
3. IF average 'cpu_loads' > 0.70 AND no nodes are failing:
   Output {"action_type": "scale_up"}
4. IF 'latency_ms' > 45.0:
   Output {"action_type": "throttle", "rate": 0.8}
5. IF none of the above are true:
   Output {"action_type": "no_op"}

Respond with ONLY a valid JSON action object. No markdown formatting, and no other text."""

# ---------------------------------------------------------------------------
# Required Logging Functions
# ---------------------------------------------------------------------------

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

def llm_decide(observation: dict) -> dict:
    obs_str = json.dumps(observation)
    user_prompt = f"Current system state:\n{obs_str}\nRespond with ONLY a JSON action object."

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,
                temperature=0.01,
            )
            content = response.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except Exception:
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

def run_task(task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        obs = env_reset(task_id)
    except Exception as e:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    step = 0
    rewards_list = []
    
    # Initialize task_score outside the loop so we always have a value 
    # even if the loop breaks early or errors out.
    task_score = 0.0

    while True:
        step += 1
        action = llm_decide(obs)
        
        # Format action strictly on one line without quotes that break bash/parsing
        action_str = json.dumps(action).replace('"', "'") 
        
        error_msg = None
        reward = 0.0
        done = False

        try:
            # ---> THE CHANGES YOU ASKED ABOUT ARE HERE <---
            result = env_step(action)
            data_block = result.get("data", result)
            
            if "observation" in data_block and isinstance(data_block["observation"], dict):
                obs = data_block["observation"]
            else:
                obs = data_block

            reward = float(data_block.get("reward", obs.get("reward", 0.0)))
            done = bool(data_block.get("done", obs.get("done", False)))
            
            # This continuously updates the task_score on every single step.
            task_score = float(obs.get("task_score", 0.0))

        except Exception as e:
            error_msg = str(e).replace("\n", " ") # Prevent newline breaks in STDOUT
            done = True

        rewards_list.append(reward)
        log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

        # Even if step > 100 hits (timeout failure), task_score has the partial credit from the last step!
        if done or step > 100:
            # Define success: Let's say getting more than 0.1 points counts as partial success
            success = task_score >= 0.1 
            log_end(success=success, steps=step, score=task_score, rewards=rewards_list)
            return task_score

def main():
    for task_id in TASKS:
        run_task(task_id)

if __name__ == "__main__":
    main()
