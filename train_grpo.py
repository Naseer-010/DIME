#!/usr/bin/env python3
"""
GRPO fine-tuning of Qwen3-8B on DIME.

Why this avoids the -1000 reward trap:
  1. Custom reward (uptime + latency + db_survival) replaces ProductionSREReward.
     Range [-3.5, +1.0] with genuine within-group variance.
  2. Format/validity bonuses give signal even when env rewards are identical,
     preventing GRPO advantages from collapsing to zero in early training.
  3. Group std guard: skips the update when all G rewards are equal (std < 1e-4).
  4. New triage rule 8 (DB RECOVERY: restart node-0 when in failed_nodes) fixes
     the unrecoverable spiral that causes -1000 on every post-crash step.

Post-training compatibility with inference.py:
  - Saves a standard HuggingFace checkpoint (same architecture).
  - Set MODEL_NAME = "<save_path>/merged" in inference.py to benchmark.
"""

import copy
import csv
import json
import os
import random
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from server.environment import DistributedInfraEnvironment
from server.models import InfraAction, InfraObservation
from server.command_parser import parse_command, CommandParseError

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen3-8B"

    # GRPO
    G: int = 4                      # completions per prompt (group size)
    clip_eps: float = 0.2           # PPO-style surrogate clipping
    ent_coef: float = 0.001         # entropy bonus to prevent premature collapse

    # Generation
    max_new_tokens: int = 1024      # budget for <think> + <action> combined
    temperature: float = 0.9       # high enough for within-group diversity
    top_p: float = 0.95

    # Optimisation
    lr: float = 5e-7                # small — LLM parameters are fragile to large steps
    gradient_accumulation: int = 8  # update every 8 GRPO group-steps
    max_grad_norm: float = 1.0

    # Episode
    max_episodes: int = 500
    max_steps_per_episode: int = 15  # shorter than eval (30) keeps episodes fast
    save_every: int = 50
    log_every: int = 5

    # Paths
    save_path: str = "checkpoints/qwen3_grpo"
    metrics_file: str = "metrics_grpo_training.csv"

    # Curriculum
    curriculum_window: int = 30
    curriculum_threshold: float = -0.5  # advance when mean ep_reward > threshold

    # Memory
    gradient_checkpointing: bool = True
    use_lora: bool = False           # auto-enabled when peft is installed


CURRICULUM: List[List[str]] = [
    ["traffic_spike", "node_failure"],
    ["cascading_failure", "flash_crowd", "thundering_herd"],
    ["zombie_node", "hot_shard_skew", "memory_leak_slow_burn",
     "split_brain_io_bottleneck", "black_swan_az_failure",
     "retry_storm", "connection_pool_deadlock", "autoscaler_flapping_trap"],
]


# ---------------------------------------------------------------------------
# System prompt — identical to inference.py PLUS rule 8 (DB recovery)
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
- kubectl exec -it istio-proxy -- traffic shift --from=<ID> --to=<ID>  → reroute traffic away from a bad node
- kubectl throttle ingress --rate=<float>  → drop traffic (0.0 to 1.0). Burns error budget!
- kubectl logs node-<ID>                 → investigate telemetry timeout
- no_op                                  → do nothing

CRITICAL INCIDENT TRIAGE TREE (Follow strictly in order):
1. OOM IMMINENT (Memory Leak): IF ANY 'mem_utilizations' > 0.92:
   IMMEDIATELY output: kubectl delete pod node-5 (or whichever node is leaking)

2. SPLIT-BRAIN (Disk I/O Bottleneck): IF node_0 'io_wait' > 0.80:
   Output: kubectl throttle ingress --rate=0.5

3. HOT SHARD (Load Balancer Skew): IF one worker's CPU > 0.90 but cluster average is low:
   Output: kubectl exec -it istio-proxy -- traffic shift --from=<high_cpu_node> --to=<low_cpu_node>

4. RETRY STORM / THUNDERING HERD: IF 'p99_latency' > 100.0 AND traffic is spiking:
   Output: kubectl throttle ingress --rate=0.4

5. CONNECTION DEADLOCK (Zombie Node): IF a worker CPU < 0.10 BUT 'p99_latency' is huge:
   Output: kubectl exec -it istio-proxy -- traffic shift --from=<zombie_node> --to=<healthy_node>

6. BLACK SWAN (Multi-Node Death): IF multiple nodes are in 'failed_nodes':
   Output: kubectl throttle ingress --rate=0.3

7. DATABASE SURVIVAL: IF node-0 (DB) cpu_load > 0.80:
   Output: kubectl throttle ingress --rate=0.7

8. DB RECOVERY: IF node-0 is in 'failed_nodes':
   Output: kubectl delete pod node-0  ← restart the database immediately!

9. SAFE SCALING: IF avg worker CPU > 0.75 AND 'error_budget' > 20:
   Output: kubectl scale deployment frontend --replicas=10

10. HEALTHY / FLAPPING TRAP: If metrics are stable or oscillating slightly:
    Output: no_op

Be concise. Respond in this exact format:
<reasoning>One sentence identifying the triage rule that applies.</reasoning>
<action>
{"command": "your_kubectl_command_or_no_op_here"}
</action>"""


# ---------------------------------------------------------------------------
# Reward function — the core fix vs ProductionSREReward
# ---------------------------------------------------------------------------

def compute_step_reward(env: DistributedInfraEnvironment, completion: str) -> float:
    """
    Three-component reward with guaranteed within-group variance.

    Component 1 — env_reward (range [-3.0, +0.5]):
      +0.5 × uptime_fraction          positive signal for keeping nodes alive
      -0.5 × clamp(lat_excess^2, 1)   quadratic above 50ms SLA
      -2.0 if db failed               hard signal so policy learns to protect node-0

    Component 2 — format_reward (±0.3):
      +0.3 for valid <reasoning>+<action> XML structure
      -0.3 otherwise
      Ensures ~0.6 within-group range even before the policy learns any SRE skills.

    Component 3 — validity_reward (±0.2):
      +0.2 if the action command parses without CommandParseError
      -0.2 if hallucinated syntax
      Discourages the model from generating syntactically invalid commands.

    Total range: [-3.5, +1.0]
    Group std on random outputs: ~0.4 — sufficient for GRPO learning.
    """
    sim = env.sim
    nodes = sim.nodes

    # 1. Environment reward
    alive = sum(1 for n in nodes if not n.is_failed)
    uptime_r = 0.5 * (alive / max(len(nodes), 1))
    excess = max(0.0, sim.latency_ms - 50.0) / 100.0
    lat_r = -0.5 * min(excess ** 2, 1.0)
    db_r = -2.0 if (nodes and nodes[0].is_failed) else 0.0
    env_r = uptime_r + lat_r + db_r

    # 2. Format reward
    has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", completion, re.DOTALL))
    has_action = bool(re.search(r"<action>.*?</action>", completion, re.DOTALL))
    format_r = 0.3 if (has_reasoning and has_action) else -0.3

    # 3. Validity reward
    validity_r = -0.2
    act_match = re.search(r"<action>\s*(.*?)\s*</action>", completion, re.DOTALL)
    if act_match:
        json_str = act_match.group(1).strip()
        s, e = json_str.find("{"), json_str.rfind("}")
        if s != -1 and e != -1:
            try:
                action_dict = json.loads(json_str[s : e + 1])
                cmd = str(action_dict.get("command") or action_dict.get("raw_command") or "")
                if not cmd or cmd == "no_op":
                    validity_r = 0.2
                else:
                    parse_command(cmd)
                    validity_r = 0.2
            except (json.JSONDecodeError, CommandParseError, Exception):
                pass

    return env_r + format_r + validity_r


# ---------------------------------------------------------------------------
# Prompt + completion utilities
# ---------------------------------------------------------------------------

def build_prompt(obs: InfraObservation, tokenizer) -> str:
    """Build the chat prompt — matches inference.py exactly."""
    obs_fields = [
        "cpu_loads", "mem_utilizations", "queue_lengths", "failed_nodes",
        "latency_ms", "request_rate", "io_wait", "p99_latency", "error_budget",
        "request_rate_norm", "p99_latency_norm", "step", "task_hint",
        "action_errors", "cloud_budget", "telemetry_status",
    ]
    obs_dict = {f: getattr(obs, f) for f in obs_fields if hasattr(obs, f)}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Current system state:\n{json.dumps(obs_dict)}\n"
                "Respond with the required XML and JSON format."
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


def parse_completion_to_action(completion: str) -> InfraAction:
    """Extract InfraAction from a model completion, falling back to no_op."""
    text = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()
    act_match = re.search(r"<action>\s*(.*?)\s*</action>", text, re.DOTALL | re.IGNORECASE)
    if not act_match:
        return InfraAction(action_type="no_op")

    json_str = act_match.group(1).strip()
    s, e = json_str.find("{"), json_str.rfind("}")
    if s == -1 or e == -1:
        return InfraAction(action_type="no_op")

    try:
        action_dict = json.loads(json_str[s : e + 1])
    except json.JSONDecodeError:
        return InfraAction(action_type="no_op")

    cmd = str(action_dict.get("command") or action_dict.get("raw_command") or "no_op").strip()
    if cmd == "no_op":
        return InfraAction(action_type="no_op")
    try:
        return parse_command(cmd)
    except CommandParseError:
        return InfraAction(action_type="no_op")


# ---------------------------------------------------------------------------
# Log-prob utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_completion_log_prob(
    model,
    prompt_ids: torch.Tensor,      # [prompt_len]  (cpu)
    completion_ids: torch.Tensor,  # [comp_len]    (cpu)
    device: torch.device,
) -> float:
    """Sequence-level log probability of the completion given the prompt (no grad)."""
    full_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0).to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits[0].float()   # [seq_len, vocab] in fp32

    p_len, c_len = prompt_ids.shape[0], completion_ids.shape[0]
    # logits[t] predicts token[t+1]; completion starts at index p_len
    comp_logits = logits[p_len - 1 : p_len + c_len - 1]  # [c_len, vocab]
    log_probs = F.log_softmax(comp_logits, dim=-1)
    token_lps = log_probs.gather(1, completion_ids.unsqueeze(1).to(device)).squeeze(1)
    return float(token_lps.sum().item())


def compute_grpo_loss(
    model,
    prompt_ids: torch.Tensor,
    completions_ids: List[torch.Tensor],
    advantages: List[float],
    old_log_probs: List[float],
    clip_eps: float,
    ent_coef: float,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    GRPO loss for one group.

    For each completion i:
      ratio_i  = exp(log_pi_new(c_i) − log_pi_old(c_i))
      pg_loss  = −A_i × min(ratio_i, clip(ratio_i, 1±ε))
      ent_loss = −ent_coef × H(pi over completion tokens)

    Returns None when all advantages ≈ 0 (skips degenerate groups).
    """
    if max(abs(a) for a in advantages) < 1e-4:
        return None

    total_pg = torch.zeros(1, device=device)
    total_ent = torch.zeros(1, device=device)

    for comp_ids, adv, old_lp in zip(completions_ids, advantages, old_log_probs):
        full_ids = torch.cat([prompt_ids, comp_ids]).unsqueeze(0).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(full_ids).logits[0].float()  # [seq_len, vocab] fp32

        p_len, c_len = prompt_ids.shape[0], comp_ids.shape[0]
        comp_logits = logits[p_len - 1 : p_len + c_len - 1]
        log_probs = F.log_softmax(comp_logits, dim=-1)

        token_lps = log_probs.gather(1, comp_ids.unsqueeze(1).to(device)).squeeze(1)
        new_lp = token_lps.sum()

        entropy = -(torch.exp(log_probs) * log_probs).sum(-1).mean()

        ratio = torch.exp(new_lp - float(old_lp))
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        total_pg += -float(adv) * torch.min(ratio, clipped)
        total_ent += entropy

    G = len(completions_ids)
    return total_pg / G - ent_coef * (total_ent / G)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg: GRPOConfig):
    """Load tokenizer + model, applying LoRA if peft is available."""
    print(f"[GRPO] Loading {cfg.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Estimate GPU budget
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if HAS_PEFT and cfg.use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        print(f"[GRPO] LoRA enabled (r=16) — GPU: {gpu_gb:.0f}GB")
    elif gpu_gb < 50:
        raise RuntimeError(
            f"GPU has only {gpu_gb:.0f}GB. Full fine-tuning of Qwen3-8B needs ~50GB.\n"
            "Options:\n"
            "  1. Install peft and set cfg.use_lora=True: pip install peft\n"
            "  2. Use a larger GPU (A100 40/80GB recommended)"
        )
    else:
        print(f"[GRPO] Full fine-tuning ({gpu_gb:.0f}GB GPU — all parameters trainable)")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[GRPO] Trainable params: {total / 1e9:.2f}B")
    return tokenizer, model


# ---------------------------------------------------------------------------
# CSV metrics
# ---------------------------------------------------------------------------

def init_metrics_csv(path: str) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "task", "step",
            "reward", "ep_reward",
            "group_rewards", "group_std",
            "loss", "curriculum_level",
        ])


def append_metrics_csv(path: str, row: list) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = load_model(cfg)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, eps=1e-5)

        self._curriculum_level = 0
        self._ep_rewards: List[float] = []

        init_metrics_csv(cfg.metrics_file)
        os.makedirs(cfg.save_path, exist_ok=True)
        print(f"[GRPO] Training on device: {self.device}")

    # ---- Curriculum ----

    def _current_tasks(self) -> List[str]:
        return CURRICULUM[self._curriculum_level]

    def _try_advance(self) -> None:
        if (
            self._curriculum_level < len(CURRICULUM) - 1
            and len(self._ep_rewards) >= self.cfg.curriculum_window
        ):
            window = self._ep_rewards[-self.cfg.curriculum_window :]
            if float(np.mean(window)) >= self.cfg.curriculum_threshold:
                self._curriculum_level += 1
                print(f"\n[CURRICULUM] → Level {self._curriculum_level + 1}")
                print(f"[CURRICULUM]   Tasks: {self._current_tasks()}\n")

    # ---- Generation ----

    def _generate_group(
        self, obs: InfraObservation
    ) -> Tuple[str, List[str], List[torch.Tensor], List[float], torch.Tensor]:
        """
        Generate G completions from the same prompt.

        Returns:
          prompt_str        — the full prompt text
          completions_str   — G decoded completions
          completions_ids   — G completion token ID tensors (cpu)
          old_log_probs     — G scalar log probs under current policy (no_grad)
          prompt_ids        — prompt token ID tensor (cpu)
        """
        cfg = self.cfg
        prompt_str = build_prompt(obs, self.tokenizer)
        enc = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
        prompt_ids_gpu = enc.input_ids[0]  # [prompt_len]

        with torch.no_grad():
            out = self.model.generate(
                enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                num_return_sequences=cfg.G,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # out: [G, prompt_len + comp_len_padded]

        p_len = prompt_ids_gpu.shape[0]
        eos_id = self.tokenizer.eos_token_id
        completions_str, completions_ids, old_log_probs = [], [], []

        for i in range(cfg.G):
            comp_raw = out[i][p_len:]  # [comp_len_padded]

            # Trim trailing padding (pad_token == eos_token for Qwen)
            eos_pos = (comp_raw == eos_id).nonzero(as_tuple=True)[0]
            end = int(eos_pos[0].item()) + 1 if len(eos_pos) > 0 else comp_raw.shape[0]
            comp_ids = comp_raw[:end].cpu()

            comp_str = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
            completions_str.append(comp_str)
            completions_ids.append(comp_ids)

            lp = get_completion_log_prob(self.model, prompt_ids_gpu.cpu(), comp_ids, self.device)
            old_log_probs.append(lp)

        return prompt_str, completions_str, completions_ids, old_log_probs, prompt_ids_gpu.cpu()

    # ---- GRPO env step ----

    def _grpo_env_step(
        self,
        env: DistributedInfraEnvironment,
        obs: InfraObservation,
    ) -> Tuple[InfraObservation, float, List[float], float, Optional[torch.Tensor]]:
        """
        One GRPO step.

        1. Generate G completions for current obs.
        2. Deep-copy the env G times, step each copy with its completion, score.
        3. Compute group-relative advantages (mean=0, std=1).
        4. Compute GRPO loss.
        5. Advance the REAL env with completion[0] (on-policy).

        Returns: (new_obs, reward_0, all_rewards, group_std, grpo_loss)
        """
        _, completions_str, completions_ids, old_lps, prompt_ids = self._generate_group(obs)

        # Score each completion in an independent env copy
        rewards: List[float] = []
        for comp_str in completions_str:
            env_copy = deepcopy(env)
            action = parse_completion_to_action(comp_str)
            try:
                env_copy.step(action)
            except Exception:
                pass
            rewards.append(compute_step_reward(env_copy, comp_str))

        group_std = float(np.std(rewards)) + 1e-8
        mean_r = float(np.mean(rewards))
        advantages = [(r - mean_r) / group_std for r in rewards]

        loss = compute_grpo_loss(
            self.model,
            prompt_ids,
            completions_ids,
            advantages,
            old_lps,
            self.cfg.clip_eps,
            self.cfg.ent_coef,
            self.device,
        )

        # Advance real env on-policy (completion[0] = first sample)
        action_0 = parse_completion_to_action(completions_str[0])
        new_obs = env.step(action_0)

        return new_obs, rewards[0], rewards, group_std, loss

    # ---- Checkpoint ----

    def _save_checkpoint(self, tag: str, final: bool = False) -> None:
        save_dir = Path(self.cfg.save_path)

        if HAS_PEFT and hasattr(self.model, "peft_config"):
            # Intermediate: save LoRA adapter only (non-destructive)
            adapter_dir = save_dir / f"adapter_{tag}"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(adapter_dir))
            self.tokenizer.save_pretrained(str(adapter_dir))
            print(f"[GRPO] LoRA adapter → {adapter_dir}")

            if final:
                # Merge for inference.py compatibility
                print("[GRPO] Merging LoRA weights into base model...")
                merged = copy.deepcopy(self.model).merge_and_unload()
                merged_dir = save_dir / "merged"
                merged_dir.mkdir(parents=True, exist_ok=True)
                merged.save_pretrained(str(merged_dir))
                self.tokenizer.save_pretrained(str(merged_dir))
                print(f"[GRPO] Merged model → {merged_dir}")
                print(f"[GRPO] Benchmark: set MODEL_NAME = '{merged_dir}' in inference.py")
                del merged
        else:
            # Full fine-tuning: save full model
            ckpt_dir = save_dir / tag
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(ckpt_dir))
            self.tokenizer.save_pretrained(str(ckpt_dir))
            print(f"[GRPO] Checkpoint → {ckpt_dir}")
            if final:
                # Symlink latest as 'merged' so inference.py command is stable
                merged_link = save_dir / "merged"
                if merged_link.exists() or merged_link.is_symlink():
                    merged_link.unlink()
                merged_link.symlink_to(ckpt_dir.resolve())
                print(f"[GRPO] Benchmark: set MODEL_NAME = '{merged_link}' in inference.py")

    # ---- Main training loop ----

    def train(self) -> None:
        cfg = self.cfg
        self.optimizer.zero_grad()
        accum_count = 0
        running_loss = 0.0

        print(f"\n[GRPO] Starting | {cfg.max_episodes} episodes | "
              f"G={cfg.G} | LR={cfg.lr} | accum={cfg.gradient_accumulation}")
        print(f"[GRPO] Level 1 tasks: {self._current_tasks()}\n")

        for episode in range(cfg.max_episodes):
            task = random.choice(self._current_tasks())
            env = DistributedInfraEnvironment()
            obs = env.reset(task=task)

            ep_reward = 0.0
            ep_losses: List[float] = []
            step = 0

            while step < cfg.max_steps_per_episode:
                new_obs, step_r, all_r, grp_std, loss = self._grpo_env_step(env, obs)
                obs = new_obs
                ep_reward += step_r

                if loss is not None:
                    (loss / cfg.gradient_accumulation).backward()
                    running_loss += loss.item()
                    ep_losses.append(loss.item())
                    accum_count += 1

                    if accum_count >= cfg.gradient_accumulation:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            cfg.max_grad_norm,
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        running_loss = 0.0
                        accum_count = 0

                append_metrics_csv(cfg.metrics_file, [
                    episode, task, step,
                    round(step_r, 4), round(ep_reward, 4),
                    json.dumps([round(r, 4) for r in all_r]),
                    round(grp_std, 4),
                    round(loss.item(), 6) if loss is not None else "",
                    self._curriculum_level + 1,
                ])

                step += 1
                if obs.done:
                    break

            self._ep_rewards.append(ep_reward)
            self._try_advance()

            if episode % cfg.log_every == 0:
                window = self._ep_rewards[-cfg.curriculum_window :] or [ep_reward]
                mean_ep = float(np.mean(window))
                mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
                print(
                    f"[{episode:>4d}] task={task:<35} "
                    f"ep_rew={ep_reward:+.3f}  mean={mean_ep:+.3f}  "
                    f"loss={mean_loss:.5f}  curriculum={self._curriculum_level + 1}"
                )

            if (episode + 1) % cfg.save_every == 0:
                self._save_checkpoint(f"ep{episode + 1}")

        # Flush any remaining gradients
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        self._save_checkpoint("final", final=True)
        print(f"\n[GRPO] Training complete. Metrics → {cfg.metrics_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = GRPOConfig(
        model_name="Qwen/Qwen3-8B",
        G=4,
        clip_eps=0.2,
        ent_coef=0.001,
        max_new_tokens=1024,
        temperature=0.9,
        top_p=0.95,
        lr=5e-7,
        gradient_accumulation=8,
        max_grad_norm=1.0,
        max_episodes=500,
        max_steps_per_episode=15,
        save_every=50,
        log_every=5,
        save_path="checkpoints/qwen3_grpo",
        metrics_file="metrics_grpo_training.csv",
        curriculum_window=30,
        curriculum_threshold=-0.5,
        gradient_checkpointing=True,
        use_lora=HAS_PEFT,      # auto-use LoRA if peft is installed
    )

    trainer = GRPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
