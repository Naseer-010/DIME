#!/usr/bin/env python3
"""
Parse and analyse metrics_Qwen_Qwen3-8B.csv produced by inference.py.

Usage:
    python nithish_data_parser.py                          # full report
    python nithish_data_parser.py --task traffic_spike     # one task
    python nithish_data_parser.py --show-thinking 3        # print N think blocks
    python nithish_data_parser.py --wrong-actions          # show misdiagnosed steps
"""

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).parent / "metrics_Qwen_Qwen3-8B.csv"

COLUMNS = [
    "model", "task_id", "step", "action_taken",
    "reasoning", "reward", "cumulative_score", "done", "error",
]

# ── helpers ──────────────────────────────────────────────────────────────────

def load(path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS, on_bad_lines="skip")
    df["step"]             = pd.to_numeric(df["step"],             errors="coerce")
    df["reward"]           = pd.to_numeric(df["reward"],           errors="coerce")
    df["cumulative_score"] = pd.to_numeric(df["cumulative_score"], errors="coerce")
    df["done"]             = df["done"].astype(str).str.strip().str.lower() == "true"
    return df


def extract_command(action_str: str) -> str:
    """Pull the kubectl command string from the action_taken column."""
    s = str(action_str).strip()
    # Try ast.literal_eval first (it's a Python dict repr)
    try:
        d = ast.literal_eval(s)
        return str(d.get("command") or d.get("raw_command") or "no_op").strip()
    except Exception:
        pass
    # Fallback: JSON-like
    m = re.search(r'"command"\s*:\s*"([^"]+)"', s)
    if m:
        return m.group(1).strip()
    return s


def extract_think(reasoning: str):
    """Return (think_block, structured_output) from a raw reasoning string."""
    think_m = re.search(r"<think>(.*?)</think>", reasoning, re.DOTALL | re.IGNORECASE)
    think   = think_m.group(1).strip() if think_m else ""

    # Everything after </think> (the <reasoning><action> part)
    after = reasoning[think_m.end():].strip() if think_m else reasoning.strip()
    return think, after


def classify_action(cmd: str) -> str:
    cmd = cmd.strip()
    if cmd == "no_op":                          return "no_op"
    if "delete pod"       in cmd:               return "restart_node"
    if "traffic shift"    in cmd:               return "reroute"
    if "throttle ingress" in cmd:               return "throttle"
    if "scale deployment" in cmd:               return "scale_up"
    if "kubectl logs"     in cmd:               return "query_logs"
    return "unknown"


# ── report sections ──────────────────────────────────────────────────────────

def print_separator(title=""):
    w = 72
    if title:
        print(f"\n{'─'*3} {title} {'─'*(w - len(title) - 5)}")
    else:
        print("─" * w)


def task_summary(df: pd.DataFrame):
    print_separator("PER-TASK SUMMARY")
    grp = df.groupby("task_id")

    rows = []
    for task, g in grp:
        total_steps  = len(g)
        pct_neg1000  = (g["reward"] == -1000).mean() * 100
        mean_reward  = g["reward"].mean()
        final_score  = g["cumulative_score"].dropna().iloc[-1] if len(g) > 0 else float("nan")
        n_done       = g["done"].sum()
        rows.append(dict(
            task=task,
            steps=total_steps,
            pct_neg1000=pct_neg1000,
            mean_reward=mean_reward,
            final_cum_score=final_score,
            episodes_done=n_done,
        ))

    summary = pd.DataFrame(rows).sort_values("task")
    summary["pct_neg1000"]    = summary["pct_neg1000"].map("{:.1f}%".format)
    summary["mean_reward"]    = summary["mean_reward"].map("{:+.1f}".format)
    summary["final_cum_score"]= summary["final_cum_score"].map("{:.3f}".format)
    print(summary.to_string(index=False))


def action_distribution(df: pd.DataFrame):
    print_separator("ACTION DISTRIBUTION  (all tasks)")
    df = df.copy()
    df["cmd"]  = df["action_taken"].apply(extract_command)
    df["type"] = df["cmd"].apply(classify_action)

    freq = df["type"].value_counts()
    total = len(df)
    for atype, cnt in freq.items():
        bar = "█" * int(cnt / total * 40)
        print(f"  {atype:<15} {cnt:>5}  ({cnt/total*100:5.1f}%)  {bar}")

    print()
    # Top-10 unique commands
    print("  Top-10 unique commands:")
    top_cmds = df["cmd"].value_counts().head(10)
    for cmd, cnt in top_cmds.items():
        print(f"    {cnt:>5}×  {cmd[:70]}")


def reward_breakdown(df: pd.DataFrame):
    print_separator("REWARD BREAKDOWN")
    total = len(df)
    neg1000  = (df["reward"] == -1000).sum()
    negative = ((df["reward"] < 0) & (df["reward"] > -1000)).sum()
    zero     = (df["reward"] == 0).sum()
    positive = (df["reward"] > 0).sum()

    print(f"  Total steps  : {total}")
    print(f"  -1000 (DB dead / reward trap) : {neg1000:>5}  ({neg1000/total*100:.1f}%)")
    print(f"  Other negative                : {negative:>5}  ({negative/total*100:.1f}%)")
    print(f"  Zero                          : {zero:>5}  ({zero/total*100:.1f}%)")
    print(f"  Positive                      : {positive:>5}  ({positive/total*100:.1f}%)")

    print()
    print("  WHY -1000?  calculate_step_reward() returns -1000 whenever")
    print("  node-0 (the DB SPOF) is in failed_nodes — even if the action")
    print("  is correct. Rule 8 (kubectl delete pod node-0) was MISSING")
    print("  from inference.py's system prompt, so the model never learned")
    print("  to restart the DB. Once the DB crashes, every subsequent step")
    print("  scores -1000 regardless of what command is issued.")


def thinking_analysis(df: pd.DataFrame, n: int = 3, task_filter: str = None):
    print_separator(f"THINKING SAMPLES  (n={n})")
    sample = df if task_filter is None else df[df["task_id"] == task_filter]
    sample = sample.dropna(subset=["reasoning"]).sample(min(n, len(sample)), random_state=42)

    for _, row in sample.iterrows():
        think, structured = extract_think(str(row["reasoning"]))
        cmd = extract_command(str(row["action_taken"]))

        print(f"\n  ┌─ task={row['task_id']}  step={row['step']}  reward={row['reward']}")
        print(f"  │  command : {cmd}")

        # Print first 400 chars of thinking
        short_think = think[:400].replace("\n", " ").strip()
        if len(think) > 400:
            short_think += " …"
        print(f"  │  thinking: {short_think}")

        # Show structured output after </think>
        short_struct = structured[:200].strip().replace("\n", " ")
        print(f"  │  output  : {short_struct}")
        print("  └" + "─" * 68)


def wrong_action_analysis(df: pd.DataFrame, n: int = 8):
    """Show steps where model chose a clearly wrong action despite correct reasoning."""
    print_separator("MISDIAGNOSED STEPS  (correct thinking → wrong command)")
    df = df.copy()
    df["cmd"]  = df["action_taken"].apply(extract_command)
    df["type"] = df["cmd"].apply(classify_action)

    # Steps where reward=-1000 but action was NOT throttle (model tried something else)
    # These highlight cases where the model diagnosed correctly but still lost
    interesting = df[
        (df["reward"] == -1000) &
        (df["type"].isin(["restart_node", "reroute", "scale_up"]))
    ].head(n)

    if interesting.empty:
        print("  (none found with current filters)")
        return

    for _, row in interesting.iterrows():
        think, _ = extract_think(str(row["reasoning"]))
        # What rule did it think applied?
        rule_m = re.search(r"rule\s*\d+|black swan|hot shard|oom|retry storm|zombie|split.brain|db recov",
                           think[:500], re.IGNORECASE)
        rule_hint = rule_m.group(0) if rule_m else "unknown rule"
        print(f"  task={row['task_id']:<30} step={int(row['step'])}  "
              f"rule≈{rule_hint:<20}  cmd={row['cmd'][:50]}")


def per_task_reward_curve(df: pd.DataFrame, task_filter: str = None):
    print_separator("CUMULATIVE SCORE PER TASK  (last value = final score)")
    tasks = [task_filter] if task_filter else sorted(df["task_id"].unique())
    for task in tasks:
        g = df[df["task_id"] == task].sort_values("step")
        scores = g["cumulative_score"].dropna().values
        if len(scores) == 0:
            continue
        # Mini sparkline
        lo, hi = scores.min(), scores.max()
        rng = hi - lo or 1
        bars = "▁▂▃▄▅▆▇█"
        spark = "".join(bars[min(int((s - lo) / rng * 7), 7)] for s in scores[-40:])
        print(f"  {task:<35}  final={scores[-1]:+.3f}  [{spark}]")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DIME inference metrics parser")
    parser.add_argument("--task",          default=None,  help="Filter to one task_id")
    parser.add_argument("--show-thinking", type=int, default=0, metavar="N",
                        help="Print N thinking block samples")
    parser.add_argument("--wrong-actions", action="store_true",
                        help="Show misdiagnosed steps")
    parser.add_argument("--csv",           default=str(CSV_PATH), help="Path to CSV")
    args = parser.parse_args()

    print(f"Loading {args.csv} …")
    df = load(Path(args.csv))
    if args.task:
        df_view = df[df["task_id"] == args.task]
        if df_view.empty:
            print(f"No rows found for task '{args.task}'. Available tasks:")
            print(" ", sorted(df["task_id"].unique()))
            sys.exit(1)
    else:
        df_view = df

    print(f"Loaded {len(df_view)} steps  |  "
          f"tasks: {sorted(df_view['task_id'].unique())}")

    task_summary(df_view)
    reward_breakdown(df_view)
    action_distribution(df_view)
    per_task_reward_curve(df_view, args.task)

    if args.show_thinking:
        thinking_analysis(df_view, n=args.show_thinking, task_filter=args.task)

    if args.wrong_actions:
        wrong_action_analysis(df_view)

    print_separator()
    print("Done.")


if __name__ == "__main__":
    main()
