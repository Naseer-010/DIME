#!/usr/bin/env python3
"""Run an LLM-backed DIME agent through the official benchmark harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any

from agents.llm_agent import LLMResearchAgent
from benchmark.benchmark_registry import Split
from benchmark.evaluation_harness import run_benchmark


DEFAULT_API_BASE = "http://localhost:11434/v1"
DEFAULT_MODEL = "Qwen/Qwen3-8B"


def _env_api_key() -> str | None:
    return (
        os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("HF_TOKEN")
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM SRE agent with the official DIME benchmark harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument(
        "--mode",
        choices=["local", "endpoint"],
        default=os.environ.get("INFERENCE_MODE", "endpoint").lower(),
        help="Use local Transformers inference or an OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("API_BASE", DEFAULT_API_BASE),
        help="OpenAI-compatible base URL for endpoint mode.",
    )
    parser.add_argument(
        "--api-key",
        default=_env_api_key() or "dummy_key",
        help="Endpoint API key. Defaults to API_KEY, OPENAI_API_KEY, HF_TOKEN, then dummy_key.",
    )
    parser.add_argument(
        "--split",
        choices=[split.value for split in Split],
        default=Split.HIDDEN_EVAL.value,
        help="Benchmark split to evaluate.",
    )
    parser.add_argument("--benchmark-version", default="DIME-v1.0")
    parser.add_argument(
        "--reward-ablation",
        action="append",
        default=[],
        help="Reward verifier to ablate in telemetry reports. Repeat for multiple.",
    )
    parser.add_argument(
        "--verbose-agent",
        action="store_true",
        help="Print each validated LLM action and a short reasoning preview.",
    )
    return parser


def _print_summary(result: dict[str, Any]) -> None:
    summary = dict(result.get("summary", {}))
    print("\n=== DIME LLM BENCHMARK COMPLETE ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nArtifacts: {result.get('run_dir', summary.get('artifact_dir', 'unknown'))}")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    agent = LLMResearchAgent(
        model_name=args.model,
        mode=args.mode,
        api_base=args.api_base,
        api_key=args.api_key,
        verbose=args.verbose_agent,
    )

    print(
        "Starting DIME benchmark "
        f"model={args.model!r} mode={args.mode!r} split={args.split!r}"
    )
    if args.mode == "endpoint":
        print(f"Endpoint: {args.api_base}")

    result = run_benchmark(
        agent=agent,
        benchmark_version=args.benchmark_version,
        split=args.split,
        reward_ablations=args.reward_ablation,
    )
    _print_summary(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
