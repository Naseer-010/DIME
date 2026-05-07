"""Statistical reporting for DIME benchmark runs."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, variance
from typing import Any, Iterable, Mapping

from benchmark.utils import atomic_write_json, write_csv


DEFAULT_REPORT_METRICS = (
    "dime_index",
    "uptime",
    "latency_score",
    "throughput",
    "recovery_speed",
    "cost_efficiency",
    "p99_latency",
    "mttr",
    "cumulative_reward",
    "task_success",
    "survival_rate",
)


def summarize_values(values: Iterable[float]) -> dict[str, float]:
    """Compute benchmark-grade summary statistics."""
    data = [float(v) for v in values]
    if not data:
        return {"n": 0, "mean": 0.0, "std": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mu = mean(data)
    var = variance(data) if len(data) > 1 else 0.0
    std = math.sqrt(var)
    half_width = 1.96 * std / math.sqrt(len(data)) if len(data) > 1 else 0.0
    return {
        "n": len(data),
        "mean": round(mu, 6),
        "std": round(std, 6),
        "variance": round(var, 6),
        "min": round(min(data), 6),
        "max": round(max(data), 6),
        "ci95_low": round(mu - half_width, 6),
        "ci95_high": round(mu + half_width, 6),
    }


def _summarize_records(records: list[Mapping[str, Any]], metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in records if metric in row and row[metric] is not None]
        report[metric] = summarize_values(values)
    return report


def build_statistical_report(
    records: Iterable[Mapping[str, Any]],
    *,
    metrics: tuple[str, ...] = DEFAULT_REPORT_METRICS,
) -> dict[str, Any]:
    """Build summaries across episodes, tasks, and seeds."""
    data = list(records)
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_seed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

    for row in data:
        by_task[str(row.get("task_id", row.get("task", "unknown")))].append(row)
        by_seed[str(row.get("seed", "unknown"))].append(row)

    return {
        "episodes": _summarize_records(data, metrics),
        "tasks": {task: _summarize_records(rows, metrics) for task, rows in sorted(by_task.items())},
        "seeds": {seed: _summarize_records(rows, metrics) for seed, rows in sorted(by_seed.items())},
    }


def persist_statistical_report(report: Mapping[str, Any], json_path: Path, csv_path: Path) -> None:
    """Persist statistical report as JSON and long-form CSV."""
    atomic_write_json(json_path, report)
    rows: list[dict[str, Any]] = []
    for group, group_payload in report.items():
        if group == "episodes":
            for metric, stats in group_payload.items():
                rows.append({"group": group, "key": "all", "metric": metric, **stats})
        else:
            for key, metric_payload in group_payload.items():
                for metric, stats in metric_payload.items():
                    rows.append({"group": group, "key": key, "metric": metric, **stats})
    write_csv(
        csv_path,
        rows,
        ["group", "key", "metric", "n", "mean", "std", "variance", "min", "max", "ci95_low", "ci95_high"],
    )
