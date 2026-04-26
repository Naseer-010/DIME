#!/usr/bin/env python3
"""
GRPO fine-tuning of Qwen3-8B on DIME using Unsloth + TRL.

Install (run once before this script):
    pip install unsloth vllm
    # transformers 4.56.2 is pulled in automatically; no manual pinning needed.
    # Tested with: unsloth 2026.4.8, vllm 0.19.1, trl 0.24.0, transformers 4.56.2

Why Unsloth over plain HuggingFace (train_grpo.py):
  - FP8 weights + adamw_8bit halves optimizer memory
  - PatchFastRL patches TRL's GRPO trainer for Unsloth kernels
  - TRL GRPOTrainer handles rollout batching, advantage normalisation,
    clipped surrogate, and logging out of the box

-1000 reward trap fixes (same as train_grpo.py):
  1. Custom reward range [-2.5, +5.0] — never clips to -1000
  2. Format + validity bonuses guarantee within-group variance from step 0
  3. Triage oracle gives strong signal when the correct rule is unambiguous
  4. Rule 8 (DB RECOVERY) added to system prompt — missing from inference.py

Post-training benchmark:
    Set MODEL_NAME = "checkpoints/qwen3_grpo_unsloth/merged_16bit" in inference.py
    Then run python inference.py exactly as before.
"""

# UNSLOTH_VLLM_STANDBY must be set before unsloth is imported.
import os

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

# ---------------------------------------------------------------------------
# Install guard  (must run BEFORE trl imports — PatchFastRL patches them)
# ---------------------------------------------------------------------------
try:
    import unsloth  # noqa: F401
    import vllm  # noqa: F401
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}\n"
        "Run:\n"
        "  pip install unsloth vllm\n"
        "  # transformers 4.56.2 is installed automatically by vllm/unsloth"
    )

import gc
import json
import random
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL

# PatchFastRL MUST be called before importing GRPOTrainer/GRPOConfig — it
# patches TRL's vllm integration to handle the vllm API version difference.
PatchFastRL("grpo", FastLanguageModel)

from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer  # noqa: E402

from server.environment import DistributedInfraEnvironment
from server.models import InfraAction, InfraObservation
from server.command_parser import parse_command, CommandParseError
from server.rubrics import calculate_step_reward as _calculate_step_reward


def _probe_rubrics() -> bool:
    """Return True if rubrics returns bounded rewards (main branch), False if -1000 (nithish)."""
    try:
        env = DistributedInfraEnvironment()
        env.reset(task="traffic_spike")
        env.sim.nodes[0].is_failed = True
        r = _calculate_step_reward(env.sim)
        return r > -100  # main branch returns -5.0; nithish returns -1000.0
    except Exception:
        return False


_RUBRICS_BOUNDED = _probe_rubrics()
print(
    f"[GRPO] rubrics version: {'main (bounded [-5,+5])' if _RUBRICS_BOUNDED else 'nithish (WARNING: -1000 cliff — using fallback)'}"
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "unsloth/Qwen3-8B"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
OUTPUT_DIR = "checkpoints/qwen3_grpo_unsloth"

DATASET_EPISODES = 500  # env rollouts to build the training dataset
MAX_STEPS = 300  # GRPOTrainer update steps
NUM_GENERATIONS = 4  # G — completions per prompt; reward_env is CPU-bound, keep small
MAX_COMPLETION_LENGTH = (
    512  # Qwen3 no-think response is ~60 tokens; 512 is a safe ceiling
)
SAVE_STEPS = 100

ALL_TASKS = [
    "traffic_spike",
    "node_failure",
    "cascading_failure",
    "flash_crowd",
    "thundering_herd",
    "zombie_node",
    "hot_shard_skew",
    "memory_leak_slow_burn",
    "split_brain_io_bottleneck",
    "black_swan_az_failure",
    "retry_storm",
    "connection_pool_deadlock",
    "autoscaler_flapping_trap",
]

# ---------------------------------------------------------------------------
# System prompt — shared with inference.py
# ---------------------------------------------------------------------------

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
- kubectl exec -it istio-proxy -- traffic shift --from=<ID> --to=<ID>  → reroute traffic
- kubectl throttle ingress --rate=<float>  → drop traffic (0.0 to 1.0). Burns error budget!
- kubectl logs node-<ID>                 → investigate telemetry timeout
- no_op                                  → do nothing

CRITICAL INCIDENT TRIAGE TREE (Follow strictly in order):
1. OOM IMMINENT: IF ANY 'mem_utilizations' > 0.92 → kubectl delete pod node-<leaking_node>
2. DB RECOVERY: IF node-0 is in 'failed_nodes' → kubectl delete pod node-0
   (The DB is a SPOF. If it's dead, ALL other actions are futile until it restarts.)
3. SPLIT-BRAIN: IF node_0 'io_wait' > 0.80 → kubectl throttle ingress --rate=0.5
4. HOT SHARD: IF one worker CPU > 0.90 but cluster average is low
   → kubectl exec -it istio-proxy -- traffic shift --from=<hot> --to=<cold>
5. RETRY STORM: IF 'p99_latency' > 100.0 AND traffic spiking → kubectl throttle ingress --rate=0.4
6. ZOMBIE NODE: IF worker CPU < 0.10 BUT 'p99_latency' is huge
   → kubectl exec -it istio-proxy -- traffic shift --from=<zombie> --to=<healthy>
7. BLACK SWAN: IF multiple nodes in 'failed_nodes' (but DB is alive) → kubectl throttle ingress --rate=0.3
8. DATABASE SURVIVAL: IF node-0 cpu_load > 0.80 → kubectl throttle ingress --rate=0.7
9. SAFE SCALING: IF avg worker CPU > 0.75 AND 'error_budget' > 20
   → kubectl scale deployment frontend --replicas=10
10. HEALTHY: If metrics are stable → no_op

Respond in this exact format:
<reasoning>One sentence identifying the triage rule that applies.</reasoning>
<action>
{"command": "your_kubectl_command_or_no_op_here"}
</action>"""

# ---------------------------------------------------------------------------
# Triage oracle — deterministic expected action for any observation
# ---------------------------------------------------------------------------


def _get_expected_action(obs: dict) -> str:
    """Return the kubectl command the triage tree mandates, or 'no_op'.

    Rule ordering is critical for RL convergence:
      1. OOM — immediate life-or-death
      2. DB Recovery — SPOF must be restored before anything else
      3-6. Network/traffic rules
      7. Black Swan — only fires if DB is alive
      8-9. Proactive scaling
      10. Healthy
    """
    cpu = obs.get("cpu_loads", [0.3] * 8)
    mem = obs.get("mem_utilizations", [0.2] * 8)
    fail = set(obs.get("failed_nodes", []))
    io = float(obs.get("io_wait", 0.0))
    p99 = float(obs.get("p99_latency", 0.0))
    rr = float(obs.get("request_rate", 100.0))
    bud = float(obs.get("error_budget", 100.0))

    # Rule 1: OOM — instant kill prevention
    for i, m in enumerate(mem):
        if float(m) > 0.92:
            return f"kubectl delete pod node-{i}"

    # Rule 2: DB RECOVERY — the DB is a SPOF; if it's dead, nothing else matters
    if 0 in fail:
        return "kubectl delete pod node-0"

    # Rule 3: Split-brain
    if io > 0.80:
        return "kubectl throttle ingress --rate=0.5"

    # Rule 4: Hot shard
    workers = [(i, float(c)) for i, c in enumerate(cpu[1:], 1) if float(c) >= 0]
    if workers:
        avg = sum(c for _, c in workers) / len(workers)
        for i, c in workers:
            if c > 0.90 and avg < 0.60:
                dst = min(
                    (j for j, d in workers if j != i and j not in fail),
                    key=lambda j: float(cpu[j]),
                    default=None,
                )
                if dst is not None:
                    return f"kubectl exec -it istio-proxy -- traffic shift --from={i} --to={dst}"

    # Rule 5: Retry storm
    if p99 > 100.0 and rr > 150:
        return "kubectl throttle ingress --rate=0.4"

    # Rule 6: Zombie node
    for i, c in workers:
        if 0 <= c < 0.10 and p99 > 100.0:
            dst = next(
                (j for j, d in workers if d > 0.2 and j not in fail and j != i),
                None,
            )
            if dst is not None:
                return f"kubectl exec -it istio-proxy -- traffic shift --from={i} --to={dst}"

    # Rule 7: Black swan (only fires when DB is alive — DB recovery is above)
    if len(fail) >= 2:
        return "kubectl throttle ingress --rate=0.3"

    # Rule 8: DB survival (protect a living DB under load)
    db_cpu = float(cpu[0]) if cpu and float(cpu[0]) >= 0 else 0.0
    if db_cpu > 0.80:
        return "kubectl throttle ingress --rate=0.7"

    # Rule 9: Safe scaling
    if workers and sum(c for _, c in workers) / len(workers) > 0.75 and bud > 20:
        return "kubectl scale deployment frontend --replicas=10"

    return "no_op"


# ---------------------------------------------------------------------------
# Dataset collection
# ---------------------------------------------------------------------------


def _obs_to_dict(obs: InfraObservation) -> dict:
    keys = [
        "cpu_loads",
        "mem_utilizations",
        "queue_lengths",
        "failed_nodes",
        "latency_ms",
        "request_rate",
        "io_wait",
        "p99_latency",
        "error_budget",
        "step",
        "task_hint",
        "action_errors",
        "cloud_budget",
    ]
    return {k: getattr(obs, k) for k in keys if hasattr(obs, k)}


def _heuristic_action(obs: InfraObservation) -> InfraAction:
    """70% oracle, 30% random — ensures diverse state coverage in the dataset."""
    if random.random() < 0.30:
        atype = random.choice(["no_op", "restart_node", "throttle", "scale_up"])
        if atype == "restart_node":
            return InfraAction(
                action_type="restart_node",
                target=random.randint(0, min(7, len(obs.cpu_loads) - 1)),
            )
        if atype == "throttle":
            return InfraAction(
                action_type="throttle", rate=random.choice([0.3, 0.5, 0.7])
            )
        if atype == "scale_up":
            return InfraAction(action_type="scale_up")
        return InfraAction(action_type="no_op")

    cmd = _get_expected_action(_obs_to_dict(obs))
    if cmd == "no_op":
        return InfraAction(action_type="no_op")
    try:
        return parse_command(cmd)
    except CommandParseError:
        return InfraAction(action_type="no_op")


def collect_dataset(n_episodes: int, tasks: List[str]) -> Dataset:
    """
    Roll out the environment to build a static dataset for TRL GRPOTrainer.

    Each row = one env step observation.  Diverse states are ensured by mixing
    heuristic (rule-based) and random actions during data collection.

    Dataset columns:
      prompt   — chat messages list  (required by GRPOTrainer)
      obs_json — serialised obs dict  (passed to reward functions as kwarg)
      task     — task name            (passed to reward functions as kwarg)
    """
    rows: List[dict] = []
    env = DistributedInfraEnvironment()

    for ep in range(n_episodes):
        task = random.choice(tasks)
        obs = env.reset(task=task)

        for _ in range(20):
            d = _obs_to_dict(obs)
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "/no_think\n"  # suppress Qwen3 <think> block — response is ~60 tokens, not ~600
                                f"Current system state:\n{json.dumps(d)}\n"
                                "Respond with the required XML and JSON format."
                            ),
                        },
                    ],
                    "obs_json": json.dumps(d),
                    "task": task,
                }
            )

            action = _heuristic_action(obs)
            try:
                obs = env.step(action)
            except Exception:
                break
            if obs.done:
                break

        if (ep + 1) % 50 == 0:
            print(f"  [dataset] episode {ep + 1}/{n_episodes} → {len(rows)} rows")

    print(f"  [dataset] collected {len(rows)} total rows")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Helpers shared by reward functions
# ---------------------------------------------------------------------------


def _get_completion_text(comp) -> str:
    """
    Extract completion text from TRL GRPOTrainer's format.
    Handles both List[Dict] (messages) and plain str.
    """
    if isinstance(comp, list):
        return comp[0].get("content", "") if comp else ""
    return str(comp)


def _extract_command(text: str) -> Optional[str]:
    """Pull the kubectl command string out of a model completion."""
    # Strip think block first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"<action>\s*(.*?)\s*</action>", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    js = m.group(1).strip()
    s, e = js.find("{"), js.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        d = json.loads(js[s : e + 1])
        return str(d.get("command") or d.get("raw_command") or "no_op").strip()
    except json.JSONDecodeError:
        return None


def _restore_env_state(env: DistributedInfraEnvironment, obs: dict) -> None:
    """
    Inject observation values into env.sim for single-step reward evaluation.
    This is approximate (RNG state, adjacency, cooldowns are reset) but
    captures the signals that matter: CPU, memory, latency, failed nodes.
    """
    sim = env.sim
    cpu_l = obs.get("cpu_loads", [])
    mem_l = obs.get("mem_utilizations", [])
    q_l = obs.get("queue_lengths", [])

    for i in range(min(len(cpu_l), len(sim.nodes))):
        if float(cpu_l[i]) >= 0:
            sim.nodes[i].cpu_util = float(cpu_l[i])
        if i < len(mem_l) and float(mem_l[i]) >= 0:
            sim.nodes[i].memory_util = float(mem_l[i])
        if i < len(q_l) and int(q_l[i]) >= 0:
            sim.nodes[i].queue_length = int(q_l[i])

    for idx in obs.get("failed_nodes", []):
        if 0 <= idx < len(sim.nodes):
            sim.nodes[idx].is_failed = True

    sim.latency_ms = float(obs.get("latency_ms", 20.0))
    sim.error_budget = float(obs.get("error_budget", 100.0))
    sim.last_trace_p99_latency = float(obs.get("p99_latency", 0.0))
    sim.last_trace_node_0_io = float(obs.get("io_wait", 0.0))


# ---------------------------------------------------------------------------
# Reward functions (TRL GRPOTrainer signature)
#
# TRL calls each function as:
#   fn(completions, prompts=..., obs_json=..., task=..., **kwargs)
#
# With per_device_train_batch_size=1 and num_generations=G:
#   len(completions) == G
#   len(obs_json)    == G  (same value repeated G times by TRL)
# ---------------------------------------------------------------------------


def reward_format(completions: List, **kwargs) -> List[float]:
    """
    Reward XML structure compliance (mirrors Unsloth's match_format_exactly /
    match_format_approximately pattern).

    +3.0 — exactly one of each tag (perfect format)
     ±0.5 per tag otherwise (partial credit)
    -1.0 penalty for each duplicated or missing tag
    """
    scores = []
    for comp in completions:
        text = _get_completion_text(comp)
        n_re = text.count("<reasoning>")
        n_re_ = text.count("</reasoning>")
        n_ac = text.count("<action>")
        n_ac_ = text.count("</action>")

        if n_re == 1 and n_re_ == 1 and n_ac == 1 and n_ac_ == 1:
            scores.append(3.0)
        else:
            s = 0.0
            s += 0.5 if n_re_ == 1 else -1.0
            s += 0.5 if n_ac == 1 else -1.0
            s += 0.5 if n_ac_ == 1 else -1.0
            scores.append(s)
    return scores


def reward_validity(completions: List, **kwargs) -> List[float]:
    """
    Reward syntactically valid kubectl commands (mirrors Unsloth's check_answer).

    +2.0 — command parses without CommandParseError
    +1.0 — explicit no_op (valid choice)
    -1.0 — JSON found but command fails to parse
    -2.0 — no <action> block at all
    """
    scores = []
    for comp in completions:
        cmd = _extract_command(_get_completion_text(comp))
        if cmd is None:
            scores.append(-2.0)
        elif cmd == "no_op":
            scores.append(1.0)
        else:
            try:
                parse_command(cmd)
                scores.append(2.0)
            except CommandParseError:
                scores.append(-1.0)
    return scores


def reward_env(
    completions: List,
    obs_json: List[str] = None,
    task: List[str] = None,
    **kwargs,
) -> List[float]:
    """
    Environment simulation reward — the PRIMARY training signal.

    Uses calculate_step_reward() from server/rubrics.py:
      - 7 components: uptime, DB CPU, memory cliff, p99 latency, load shedding,
        action efficiency, temporal friction
      - Bounded to [-5.0, +5.0] — no -1000 cliff, gradients always flow

    Output is scaled by 2× so the environment physics dominates over the
    oracle (reward_triage) in the total reward signal.

    Range: [−10.0, +10.0]  (2× the raw [-5, +5])
    """
    scores = []
    for i, comp in enumerate(completions):
        try:
            obs_data = json.loads(obs_json[i]) if obs_json else {}
            task_name = task[i] if task else "traffic_spike"
        except (TypeError, IndexError, json.JSONDecodeError):
            scores.append(-10.0)
            continue

        env = DistributedInfraEnvironment()
        env.reset(task=task_name)
        _restore_env_state(env, obs_data)

        cmd = _extract_command(_get_completion_text(comp))
        if cmd and cmd != "no_op":
            try:
                action = parse_command(cmd)
            except CommandParseError:
                action = InfraAction(action_type="no_op")
        else:
            action = InfraAction(action_type="no_op")

        try:
            env.step(action)
        except Exception:
            pass

        if _RUBRICS_BOUNDED:
            # Main branch: 2× scaled to dominate over oracle reward
            scores.append(2.0 * _calculate_step_reward(env.sim))
        else:
            # Nithish branch fallback: simple 3-component formula [-5.0, +1.0]
            sim = env.sim
            nodes = sim.nodes
            alive = sum(1 for n in nodes if not n.is_failed)
            r_up = 0.5 * (alive / max(len(nodes), 1))
            r_lat = -0.5 * min((max(0.0, sim.latency_ms - 50.0) / 100.0) ** 2, 1.0)
            r_db = -2.0 if (nodes and nodes[0].is_failed) else 0.0
            scores.append(2.0 * (r_up + r_lat + r_db))

    return scores


def reward_triage(
    completions: List,
    obs_json: List[str] = None,
    **kwargs,
) -> List[float]:
    """
    Triage oracle reward — gentle guidance, NOT the primary teacher.

    Compares the model's action against the deterministic triage tree output.
    Kept intentionally weak so reward_env (physics) dominates learning.

    +1.0 — exact command match with expected action
    +0.5 — same action_type but different parameters
     0.0 — no_op when a specific action is expected, or healthy system
    -0.5 — completely wrong action type
    -0.5 — unnecessary action when system is healthy (expected no_op)
    """
    scores = []
    for i, comp in enumerate(completions):
        try:
            obs_data = json.loads(obs_json[i]) if obs_json else {}
        except (TypeError, IndexError, json.JSONDecodeError):
            scores.append(-0.5)
            continue

        expected = _get_expected_action(obs_data)
        predicted = _extract_command(_get_completion_text(comp))

        if predicted is None:
            scores.append(-0.5)
            continue

        if predicted.strip() == expected.strip():
            scores.append(1.0)
            continue

        if expected == "no_op":
            # Healthy system — mild penalty for unnecessary intervention
            scores.append(-0.5 if predicted != "no_op" else 0.0)
            continue

        if predicted == "no_op":
            # Missed a required action — mild penalty
            scores.append(0.0)
            continue

        # Same action type, wrong parameters?
        try:
            act_p = parse_command(predicted)
            act_e = parse_command(expected)
            scores.append(0.5 if act_p.action_type == act_e.action_type else -0.5)
        except CommandParseError:
            scores.append(-0.5)

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ---- Load model (Unsloth FastLanguageModel + LoRA + FP8) ----
    print(f"[GRPO] Loading {MODEL_NAME} ...")
    # fast_inference=True  → vLLM engine, much faster generation (~3-5x vs model.generate)
    # compilation_config=0 → basic CUDA graphs only; skips piecewise graph-split that
    #                        crashes on A100 SM 8.0 (vLLM bug in _decompose_size_nodes)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        load_in_fp8=False,  # FP8 requires compute capability 8.9+; A100 is 8.0
        compilation_config=0,  # avoid piecewise graph-split crash; still uses CUDA graphs
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,  # 2× alpha speeds up training
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",  # include MLP for DIME reasoning
        ],
        use_gradient_checkpointing="unsloth",  # 30% memory reduction
        random_state=3407,
    )

    # ---- Collect dataset ----
    print(
        f"\n[GRPO] Collecting dataset ({DATASET_EPISODES} episodes, {len(ALL_TASKS)} tasks)..."
    )
    dataset = collect_dataset(DATASET_EPISODES, ALL_TASKS)

    # Filter prompts that exceed 90th-percentile token length (avoids outlier OOM)
    print("[GRPO] Filtering dataset by prompt length...")
    prompt_lens = [
        len(
            tokenizer.apply_chat_template(
                row["prompt"], add_generation_prompt=True, tokenize=True
            )
        )
        for row in dataset
    ]
    max_prompt_len = int(np.quantile(prompt_lens, 0.90)) + 1
    max_comp_len = MAX_SEQ_LENGTH - max_prompt_len
    keep_idx = [i for i, L in enumerate(prompt_lens) if L <= max_prompt_len]
    dataset = dataset.select(keep_idx)
    print(
        f"[GRPO] Final dataset: {len(dataset)} rows | "
        f"max_prompt={max_prompt_len} max_completion={max_comp_len}"
    )

    # ---- Sleep vLLM engine if available (frees VRAM during training) ----
    if hasattr(model, "vllm_engine") and model.vllm_engine is not None:
        try:
            model.vllm_engine.sleep()
        except Exception:
            pass

    # ---- Training config ----
    # TRL 0.24.0 API: individual sampling params instead of vllm_sampling_params object.
    # PatchFastRL above also re-adds vllm_sampling_params as an alias, but the
    # native params are clearer and forward-compatible.
    training_args = TRLGRPOConfig(
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        min_p=0.1,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=5,
        per_device_train_batch_size=1,  # keep small: reward_env is CPU-bound, not GPU-bound
        gradient_accumulation_steps=4,  # effective batch = 4
        num_generations=NUM_GENERATIONS,
        vllm_gpu_memory_utilization=0.7,  # 70% of remaining VRAM for KV cache → faster generation
        max_prompt_length=max_prompt_len,
        max_completion_length=MAX_COMPLETION_LENGTH,  # hard cap: prevents Qwen3 think-block bloat
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        output_dir=OUTPUT_DIR,
        report_to="none",
    )

    # ---- Trainer ----
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_format,  # structural: <reasoning><action> tags          ← early signal
            reward_validity,  # syntactic: command parses without error        ← anti-hallucination
            reward_env,  # semantic: env simulation, uptime+latency       ← main SRE signal
            reward_triage,  # oracle: matches triage tree expected action    ← strong supervision
        ],
        args=training_args,
        train_dataset=dataset,
    )

    print("\n[GRPO] Training starts...")
    trainer.train()

    # ---- Save ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # LoRA adapter (for resuming / inspection)
    lora_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"\n[GRPO] LoRA adapter → {lora_dir}")

    # Merged 16-bit (drop-in replacement for inference.py)
    merged_dir = os.path.join(OUTPUT_DIR, "merged_16bit")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"[GRPO] Merged 16-bit → {merged_dir}")
    print(f"\n[GRPO] Benchmark: set MODEL_NAME = '{merged_dir}' in inference.py")

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
