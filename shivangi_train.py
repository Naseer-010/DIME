#!/usr/bin/env python3
"""
Offline training script for DIME log data.

Key guarantee:
- Samples with reward <= invalid_reward_sentinel (default: -1000.0) never
  influence optimization or model-selection metrics.

This script trains a causal LM on assistant responses captured in DIME logs
using reward-weighted supervised fine-tuning (AWR-style weighting).
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import random
import re
import sys
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger("dime_train")


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

START_RE = re.compile(r"\[START\]\s+task=([^\s]+)")
BLOCK_RE = re.compile(
    r"\[DEBUG RAW OUTPUT\]\s*(.*?)\n"
    r"\[STEP\]\s+step=(\d+)\s+action=(.*?)\s+reward=([-+]?\d+(?:\.\d+)?)\s+done=(true|false)\s+error=(.*?)(?:\n|$)",
    re.DOTALL,
)
STEP_RE = re.compile(
    r"^\[STEP\]\s+step=(\d+)\s+action=(.*?)\s+reward=([-+]?\d+(?:\.\d+)?)\s+done=(true|false)\s+error=(.*)$"
)

ACTION_JSON_RE = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.DOTALL | re.IGNORECASE)


@dataclass
class LoggedSample:
    task: str
    step: int
    reward: float
    done: bool
    error: str
    action_text: str
    command: str
    response_text: str


@dataclass
class WeightedSample(LoggedSample):
    valid_for_training: bool
    weight: float


def _safe_task_at(position: int, start_positions: Sequence[int], task_names: Sequence[str]) -> str:
    idx = bisect_right(start_positions, position) - 1
    if idx < 0:
        return "unknown"
    return task_names[idx]


def _extract_command_from_response(response_text: str) -> Optional[str]:
    m = ACTION_JSON_RE.search(response_text)
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    cmd = payload.get("command") if isinstance(payload, dict) else None
    return cmd if isinstance(cmd, str) else None


def _extract_command_from_action_str(action_text: str) -> str:
    # Format example: act={'command': 'no_op'} | rsn='...'
    part = action_text
    if part.startswith("act="):
        part = part[4:]
    if "|" in part:
        part = part.split("|", 1)[0].strip()

    try:
        payload = ast.literal_eval(part)
        if isinstance(payload, dict) and isinstance(payload.get("command"), str):
            return payload["command"]
    except Exception:
        pass

    # Last-resort fallback
    if "kubectl" in action_text:
        idx = action_text.find("kubectl")
        return action_text[idx:].strip(" '\"")
    if "no_op" in action_text:
        return "no_op"
    return "unknown"


def parse_log_file(log_path: Path) -> List[LoggedSample]:
    text = log_path.read_text(encoding="utf-8", errors="replace")

    start_positions: List[int] = []
    task_names: List[str] = []
    for m in START_RE.finditer(text):
        start_positions.append(m.start())
        task_names.append(m.group(1))

    samples: List[LoggedSample] = []
    consumed_step_line_positions: set[int] = set()
    for m in BLOCK_RE.finditer(text):
        pos = m.start()
        task = _safe_task_at(pos, start_positions, task_names)

        response_text = m.group(1).strip()
        step = int(m.group(2))
        action_text = m.group(3).strip()
        reward = float(m.group(4))
        done = m.group(5).strip().lower() == "true"
        error = m.group(6).strip()

        command = _extract_command_from_response(response_text)
        if not command:
            command = _extract_command_from_action_str(action_text)

        samples.append(
            LoggedSample(
                task=task,
                step=step,
                reward=reward,
                done=done,
                error=error,
                action_text=action_text,
                command=command,
                response_text=response_text,
            )
        )
        # Mark the concrete [STEP] line location consumed by this block parse.
        step_anchor = text.find("[STEP]", m.start(), m.end() + 256)
        if step_anchor >= 0:
            consumed_step_line_positions.add(step_anchor)

    # Fallback path: parse standalone [STEP] lines (logs without [DEBUG RAW OUTPUT]).
    current_task = "unknown"
    for line_match in re.finditer(r"^.*$", text, re.MULTILINE):
        line = line_match.group(0)
        pos = line_match.start()

        m_start = START_RE.search(line)
        if m_start:
            current_task = m_start.group(1)
            continue

        m_step = STEP_RE.match(line.strip())
        if not m_step:
            continue

        if pos in consumed_step_line_positions:
            continue

        step = int(m_step.group(1))
        action_text = m_step.group(2).strip()
        reward = float(m_step.group(3))
        done = m_step.group(4).strip().lower() == "true"
        error = m_step.group(5).strip()
        command = _extract_command_from_action_str(action_text)

        response_text = (
            "<reasoning>Recovered from [STEP] line (no raw debug output found).</reasoning>\\n"
            "<action>\\n"
            + json.dumps({"command": command})
            + "\\n</action>"
        )

        samples.append(
            LoggedSample(
                task=current_task,
                step=step,
                reward=reward,
                done=done,
                error=error,
                action_text=action_text,
                command=command,
                response_text=response_text,
            )
        )

    return samples


# -----------------------------------------------------------------------------
# Weighting
# -----------------------------------------------------------------------------


def _robust_stats(values: Sequence[float]) -> Tuple[float, float]:
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0, 1.0
    median = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])

    abs_dev = sorted(abs(v - median) for v in vals)
    mad = abs_dev[n // 2] if n % 2 == 1 else 0.5 * (abs_dev[n // 2 - 1] + abs_dev[n // 2])
    scale = max(1e-6, 1.4826 * mad)
    return median, scale


def compute_weights(
    samples: Sequence[LoggedSample],
    *,
    invalid_reward_sentinel: float,
    min_weight: float,
    max_weight: float,
    temperature: float,
    include_invalid_in_dataset: bool,
) -> List[WeightedSample]:
    valid_rewards = [
        s.reward
        for s in samples
        if s.reward > invalid_reward_sentinel + 1e-12
    ]
    center, scale = _robust_stats(valid_rewards)

    weighted: List[WeightedSample] = []
    for s in samples:
        valid = s.reward > invalid_reward_sentinel + 1e-12

        if valid:
            z = (s.reward - center) / max(1e-6, scale)
            # AWR-style positive weighting
            w = math.exp(z / max(1e-6, temperature))
            w = max(min_weight, min(max_weight, w))
        else:
            # Hard guarantee: invalid samples get zero training influence.
            w = 0.0

        if include_invalid_in_dataset or valid:
            weighted.append(
                WeightedSample(
                    **asdict(s),
                    valid_for_training=valid,
                    weight=w,
                )
            )

    return weighted


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    model_name: str
    output_dir: Path
    max_length: int
    train_split: float
    batch_size: int
    grad_accum_steps: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    seed: int
    fp16: bool
    bf16: bool
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    eval_every_steps: int
    save_every_steps: int


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


class _ResponseDataset:
    def __init__(self, encodings: List[Dict[str, Any]], weights: List[float], valid_mask: List[bool]):
        self.encodings = encodings
        self.weights = weights
        self.valid_mask = valid_mask

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.encodings[idx])
        item["sample_weight"] = self.weights[idx]
        item["valid_for_training"] = 1 if self.valid_mask[idx] else 0
        return item


def _build_optimizer(model: Any, lr: float, weight_decay: float) -> Any:
    import torch

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(grouped, lr=lr)


def _collate_batch(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    import torch

    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    sample_weights = []
    valid_flags = []

    for x in batch:
        ids = x["input_ids"]
        attn = x["attention_mask"]
        pad = max_len - len(ids)

        input_ids.append(ids + [pad_token_id] * pad)
        attention_mask.append(attn + [0] * pad)
        sample_weights.append(float(x["sample_weight"]))
        valid_flags.append(int(x["valid_for_training"]))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "sample_weights": torch.tensor(sample_weights, dtype=torch.float32),
        "valid_flags": torch.tensor(valid_flags, dtype=torch.long),
    }


def _compute_weighted_loss(model: Any, batch: Dict[str, Any], device: Any) -> Tuple[Any, Dict[str, float]]:
    import torch

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    sample_weights = batch["sample_weights"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    label_mask = attention_mask[:, 1:].contiguous()

    # Ignore padded tokens
    labels = labels.masked_fill(label_mask == 0, -100)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).view(labels.size())

    token_valid = (labels != -100).float()
    per_example_den = torch.clamp(token_valid.sum(dim=1), min=1.0)
    per_example_loss = (token_loss * token_valid).sum(dim=1) / per_example_den

    # Hard guarantee: if sample weight is 0.0 (e.g. reward -1000), no gradient contribution.
    weighted_num = (per_example_loss * sample_weights).sum()
    weighted_den = torch.clamp(sample_weights.sum(), min=1e-12)
    loss = weighted_num / weighted_den

    stats = {
        "batch_loss": float(loss.detach().cpu().item()),
        "avg_example_loss": float(per_example_loss.mean().detach().cpu().item()),
        "active_weight_sum": float(sample_weights.sum().detach().cpu().item()),
        "active_examples": int((sample_weights > 0).sum().detach().cpu().item()),
    }
    return loss, stats


def _evaluate(model: Any, dataloader: Any, device: Any) -> Dict[str, float]:
    import torch

    model.eval()
    losses: List[float] = []
    total_weight = 0.0

    with torch.no_grad():
        for batch in dataloader:
            _, stats = _compute_weighted_loss(model, batch, device)
            w = stats["active_weight_sum"]
            if w <= 0:
                continue
            losses.append(stats["batch_loss"] * w)
            total_weight += w

    if total_weight <= 0:
        return {"val_loss": float("nan"), "val_ppl": float("nan")}

    val_loss = sum(losses) / total_weight
    val_ppl = math.exp(min(20.0, val_loss))
    return {"val_loss": val_loss, "val_ppl": val_ppl}


def run_training(weighted_samples: Sequence[WeightedSample], cfg: TrainConfig, dry_run: bool = False) -> None:
    if not weighted_samples:
        raise RuntimeError("No samples available after parsing/filtering.")

    if dry_run:
        LOGGER.info("Dry-run enabled. Skipping framework imports and optimizer steps.")
        return

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    except Exception as exc:
        raise RuntimeError(
            "Missing training dependencies. Install: torch transformers peft (optional for LoRA)."
        ) from exc

    _seed_everything(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    LOGGER.info("Loading tokenizer/model: %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    if cfg.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
        except Exception as exc:
            raise RuntimeError(
                "--use-lora was set but PEFT is not available or incompatible."
            ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    texts = [s.response_text for s in weighted_samples]
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=cfg.max_length,
        add_special_tokens=True,
    )

    indices = list(range(len(weighted_samples)))
    random.Random(cfg.seed).shuffle(indices)

    split_idx = max(1, int(len(indices) * cfg.train_split))
    if split_idx >= len(indices):
        split_idx = len(indices) - 1

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    if not val_idx:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    def pack(idxs: Sequence[int]) -> _ResponseDataset:
        return _ResponseDataset(
            encodings=[
                {
                    "input_ids": encodings["input_ids"][i],
                    "attention_mask": encodings["attention_mask"][i],
                }
                for i in idxs
            ],
            weights=[weighted_samples[i].weight for i in idxs],
            valid_mask=[weighted_samples[i].valid_for_training for i in idxs],
        )

    train_ds = pack(train_idx)
    val_ds = pack(val_idx)

    collate = lambda batch: _collate_batch(batch, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    optimizer = _build_optimizer(model, cfg.lr, cfg.weight_decay)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, cfg.grad_accum_steps)))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    global_step = 0
    best_val = float("inf")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        running_loss = 0.0
        seen_updates = 0
        optimizer.zero_grad(set_to_none=True)

        for step_i, batch in enumerate(train_loader, start=1):
            loss, stats = _compute_weighted_loss(model, batch, device)

            # If entire batch is invalid (e.g. all -1000), skip update safely.
            if stats["active_weight_sum"] <= 0:
                continue

            loss = loss / max(1, cfg.grad_accum_steps)
            loss.backward()

            should_step = (step_i % cfg.grad_accum_steps == 0) or (step_i == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                seen_updates += 1
                running_loss += stats["batch_loss"]

                if cfg.eval_every_steps > 0 and global_step % cfg.eval_every_steps == 0:
                    val = _evaluate(model, val_loader, device)
                    LOGGER.info(
                        "step=%d epoch=%d train_loss=%.4f val_loss=%.4f val_ppl=%.4f",
                        global_step,
                        epoch + 1,
                        stats["batch_loss"],
                        val["val_loss"],
                        val["val_ppl"],
                    )
                    if val["val_loss"] < best_val:
                        best_val = val["val_loss"]
                        best_path = cfg.output_dir / "best"
                        best_path.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)

                if cfg.save_every_steps > 0 and global_step % cfg.save_every_steps == 0:
                    ckpt = cfg.output_dir / f"checkpoint-step-{global_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)

        epoch_loss = running_loss / max(1, seen_updates)
        LOGGER.info("epoch=%d/%d avg_train_loss=%.4f updates=%d", epoch + 1, cfg.epochs, epoch_loss, seen_updates)

    final_path = cfg.output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DIME policy model from inference logs.")

    p.add_argument("--log-file", type=Path, default=Path("logs_Qwen_Qwen3-8B.txt"), help="Primary log file to parse.")
    p.add_argument(
        "--extra-log-file",
        type=Path,
        action="append",
        default=[],
        help="Additional logs to include.",
    )
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B", help="HF model ID or local model path.")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen3_dime_train"))

    p.add_argument("--invalid-reward-sentinel", type=float, default=-1000.0)
    p.add_argument("--include-invalid-in-dataset", action="store_true", help="Keep invalid rows with zero weight.")
    p.add_argument("--min-weight", type=float, default=0.05)
    p.add_argument("--max-weight", type=float, default=5.0)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")

    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--eval-every-steps", type=int, default=20)
    p.add_argument("--save-every-steps", type=int, default=100)

    p.add_argument("--dry-run", action="store_true", help="Parse and validate only; no model training.")
    p.add_argument("--log-level", type=str, default="INFO")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    log_paths = [args.log_file, *args.extra_log_file]
    all_samples: List[LoggedSample] = []

    for lp in log_paths:
        if not lp.exists():
            raise FileNotFoundError(f"Log file not found: {lp}")
        parsed = parse_log_file(lp)
        LOGGER.info("Parsed %d samples from %s", len(parsed), lp)
        all_samples.extend(parsed)

    if not all_samples:
        raise RuntimeError("No trainable blocks were parsed from log files.")

    weighted = compute_weights(
        all_samples,
        invalid_reward_sentinel=args.invalid_reward_sentinel,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        temperature=args.temperature,
        include_invalid_in_dataset=args.include_invalid_in_dataset,
    )

    parsed_total = len(all_samples)
    parsed_valid = sum(
        1 for s in all_samples if s.reward > args.invalid_reward_sentinel + 1e-12
    )
    parsed_invalid = parsed_total - parsed_valid

    kept_total = len(weighted)
    kept_valid = sum(1 for s in weighted if s.valid_for_training)
    kept_invalid = kept_total - kept_valid
    dropped_invalid = parsed_invalid - kept_invalid

    LOGGER.info(
        "Dataset summary: parsed_total=%d parsed_valid=%d parsed_invalid=%d "
        "kept_total=%d kept_valid=%d kept_invalid=%d dropped_invalid=%d",
        parsed_total,
        parsed_valid,
        parsed_invalid,
        kept_total,
        kept_valid,
        kept_invalid,
        dropped_invalid,
    )

    if kept_valid == 0:
        raise RuntimeError(
            "No valid training samples remain after masking reward <= sentinel. "
            "This is expected if your log only contains -1000 rewards. "
            "Add more logs with non -1000 rewards, or regenerate trajectories."
        )

    # Ensure zero influence from invalid rows by explicit assertion.
    leaked = [s for s in weighted if not s.valid_for_training and abs(s.weight) > 0.0]
    if leaked:
        raise AssertionError("Invalid rows detected with non-zero weight; aborting for safety.")

    manifest = {
        "log_files": [str(p) for p in log_paths],
        "parsed_total_samples": parsed_total,
        "parsed_valid_samples": parsed_valid,
        "parsed_invalid_samples": parsed_invalid,
        "kept_total_samples": kept_total,
        "kept_valid_samples": kept_valid,
        "kept_invalid_samples": kept_invalid,
        "dropped_invalid_samples": dropped_invalid,
        "invalid_reward_sentinel": args.invalid_reward_sentinel,
        "weight_stats": {
            "min": min(s.weight for s in weighted),
            "max": max(s.weight for s in weighted),
            "mean": sum(s.weight for s in weighted) / len(weighted),
        },
        "dry_run": bool(args.dry_run),
    }
    write_manifest(args.output_dir / "dataset_manifest.json", manifest)

    cfg = TrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        train_split=args.train_split,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        eval_every_steps=args.eval_every_steps,
        save_every_steps=args.save_every_steps,
    )

    write_manifest(args.output_dir / "train_config.json", asdict(cfg))

    run_training(weighted, cfg, dry_run=args.dry_run)

    LOGGER.info("Training pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOGGER.error("Training failed: %s", exc)
        raise
