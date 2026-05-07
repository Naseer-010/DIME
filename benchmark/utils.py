"""Utility helpers for canonical DIME benchmark runs."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
BENCHMARK_RUNS_DIR = RESULTS_ROOT / "benchmark_runs"
SEED_LOGS_DIR = RESULTS_ROOT / "seed_logs"
STATISTICAL_REPORTS_DIR = RESULTS_ROOT / "statistical_reports"


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Return ``value`` clipped to the closed interval [lower, upper]."""
    return max(lower, min(upper, float(value)))


def ensure_result_dirs() -> None:
    """Create benchmark artifact directories if they are missing."""
    for path in (BENCHMARK_RUNS_DIR, SEED_LOGS_DIR, STATISTICAL_REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def utc_run_id(prefix: str = "dime") -> str:
    """Stable UTC run identifier with second-level precision."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def to_plain_data(value: Any) -> Any:
    """Convert dataclasses, Pydantic models, paths, and tuples to JSON data."""
    if is_dataclass(value):
        return {field.name: to_plain_data(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "model_dump"):
        return to_plain_data(value.model_dump())
    if isinstance(value, Mapping):
        return {str(k): to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON so interrupted runs do not corrupt artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(to_plain_data(payload), fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        finally:
            raise


def append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Append JSONL records to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(to_plain_data(record), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    """Write a small CSV without bringing in pandas as a runtime dependency."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_plain_data(v) for k, v in row.items()})


def observation_to_dict(observation: Any) -> dict[str, Any]:
    """Normalize a DIME observation model or mapping to a plain dict."""
    if isinstance(observation, Mapping):
        return dict(observation)
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
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
        "task_score",
        "done",
        "reward",
        "cloud_budget",
        "action_errors",
    ]
    return {key: getattr(observation, key) for key in keys if hasattr(observation, key)}


def action_to_dict(action: Any) -> dict[str, Any]:
    """Normalize an InfraAction-like object or mapping to a plain dict."""
    if isinstance(action, Mapping):
        return dict(action)
    if hasattr(action, "model_dump"):
        return action.model_dump(exclude_none=True)
    return {
        key: getattr(action, key)
        for key in ("action_type", "target", "from_node", "to_node", "rate", "raw_command")
        if hasattr(action, key) and getattr(action, key) is not None
    }
