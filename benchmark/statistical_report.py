"""Statistical reporting for DIME benchmark runs."""

from __future__ import annotations

import math
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, variance
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
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "variance": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    mu = mean(data)
    var = variance(data) if len(data) > 1 else 0.0
    std = math.sqrt(var)
    half_width = 1.96 * std / math.sqrt(len(data)) if len(data) > 1 else 0.0
    ordered = sorted(data)
    return {
        "n": len(data),
        "mean": round(mu, 6),
        "std": round(std, 6),
        "variance": round(var, 6),
        "min": round(min(data), 6),
        "max": round(max(data), 6),
        "median": round(median(data), 6),
        "p95": round(_percentile_sorted(ordered, 95.0), 6),
        "p99": round(_percentile_sorted(ordered, 99.0), 6),
        "ci95_low": round(mu - half_width, 6),
        "ci95_high": round(mu + half_width, 6),
    }


def _percentile_sorted(ordered: list[float], pct: float) -> float:
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


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
    by_version: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

    for row in data:
        by_task[str(row.get("task_id", row.get("task", "unknown")))].append(row)
        by_seed[str(row.get("seed", "unknown"))].append(row)
        by_version[str(row.get("benchmark_version", "unknown"))].append(row)

    return {
        "episodes": _summarize_records(data, metrics),
        "tasks": {task: _summarize_records(rows, metrics) for task, rows in sorted(by_task.items())},
        "seeds": {seed: _summarize_records(rows, metrics) for seed, rows in sorted(by_seed.items())},
        "benchmark_versions": {
            version: _summarize_records(rows, metrics)
            for version, rows in sorted(by_version.items())
        },
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
        [
            "group",
            "key",
            "metric",
            "n",
            "mean",
            "std",
            "variance",
            "min",
            "max",
            "median",
            "p95",
            "p99",
            "ci95_low",
            "ci95_high",
        ],
    )


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _skew_abs(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    mu = mean(values)
    std = math.sqrt(variance(values))
    if std == 0.0:
        return 0.0
    return abs(sum(((value - mu) / std) ** 3 for value in values) / len(values))


def _welch_t_test(left: list[float], right: list[float]) -> dict[str, float | str]:
    if len(left) < 2 or len(right) < 2:
        return {"test": "welch_t", "statistic": 0.0, "p_value": 1.0}
    mean_l = mean(left)
    mean_r = mean(right)
    var_l = variance(left)
    var_r = variance(right)
    denom = math.sqrt(var_l / len(left) + var_r / len(right))
    if denom == 0.0:
        return {"test": "welch_t", "statistic": 0.0, "p_value": 1.0}
    t_stat = (mean_l - mean_r) / denom
    # Normal approximation avoids adding SciPy as a hard dependency.
    p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    return {"test": "welch_t", "statistic": round(t_stat, 6), "p_value": round(p_value, 6)}


def _mann_whitney_u(left: list[float], right: list[float]) -> dict[str, float | str]:
    if not left or not right:
        return {"test": "mann_whitney_u", "statistic": 0.0, "p_value": 1.0}
    combined = [(value, "l") for value in left] + [(value, "r") for value in right]
    combined.sort(key=lambda item: item[0])
    ranks: list[tuple[str, float]] = []
    idx = 0
    while idx < len(combined):
        end = idx + 1
        while end < len(combined) and combined[end][0] == combined[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for tie_idx in range(idx, end):
            ranks.append((combined[tie_idx][1], avg_rank))
        idx = end
    rank_sum_l = sum(rank for group, rank in ranks if group == "l")
    n_l = len(left)
    n_r = len(right)
    u_l = rank_sum_l - (n_l * (n_l + 1) / 2.0)
    mean_u = n_l * n_r / 2.0
    std_u = math.sqrt(n_l * n_r * (n_l + n_r + 1) / 12.0)
    z_score = 0.0 if std_u == 0.0 else (u_l - mean_u) / std_u
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z_score)))
    return {"test": "mann_whitney_u", "statistic": round(u_l, 6), "p_value": round(p_value, 6)}


def select_significance_test(left: Iterable[float], right: Iterable[float]) -> dict[str, Any]:
    """Select and run a safer significance test for two benchmark samples."""
    left_values = [float(v) for v in left]
    right_values = [float(v) for v in right]
    if len(left_values) < 2 or len(right_values) < 2:
        return {
            "selected_test": "insufficient_data",
            "reason": "Need at least two samples per run.",
            "result": {"p_value": 1.0, "statistic": 0.0},
        }
    var_l = variance(left_values)
    var_r = variance(right_values)
    unequal_variance = max(var_l, var_r) / max(min(var_l, var_r), 1e-9) > 4.0
    non_normal = _skew_abs(left_values) > 1.0 or _skew_abs(right_values) > 1.0
    if unequal_variance or non_normal:
        result = _mann_whitney_u(left_values, right_values)
        reason = "Selected Mann-Whitney U because samples appear non-normal or unequal-variance."
    else:
        result = _welch_t_test(left_values, right_values)
        reason = "Selected Welch t-test because samples are approximately symmetric with comparable variance."
    return {"selected_test": result["test"], "reason": reason, "result": result}


def build_leaderboard_rows(run_summaries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Generate benchmark-ready leaderboard rows."""
    rows: list[dict[str, Any]] = []
    for summary in run_summaries:
        rows.append(
            {
                "Agent Name": summary.get("agent_name", summary.get("agent", "unknown")),
                "Benchmark Version": summary.get("benchmark_version", "unknown"),
                "DIME Index": summary.get("mean_dime_index", 0.0),
                "Uptime": summary.get("mean_uptime", 0.0),
                "Latency Score": summary.get("mean_latency_score", 0.0),
                "Throughput": summary.get("mean_throughput", 0.0),
                "Recovery Speed": summary.get("mean_recovery_speed", 0.0),
                "Cost Efficiency": summary.get("mean_cost_efficiency", 0.0),
                "Mean Reward": summary.get("mean_reward", 0.0),
                "Std Reward": summary.get("std_reward", 0.0),
                "Success Rate": summary.get("success_rate", 0.0),
            }
        )
    return rows


def _load_run_metrics(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "episode_metrics.json").open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return list(payload if isinstance(payload, list) else [])


def summarize_run_for_leaderboard(run_dir: Path) -> dict[str, Any]:
    """Create one compact run summary from a benchmark artifact directory."""
    metrics = _load_run_metrics(run_dir)
    summary_path = run_dir / "benchmark_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
    report = build_statistical_report(metrics)
    episodes = report["episodes"]
    return {
        "agent_name": summary.get("agent_name", run_dir.name),
        "benchmark_version": summary.get("benchmark_version", "unknown"),
        "mean_dime_index": episodes["dime_index"]["mean"],
        "mean_uptime": episodes["uptime"]["mean"],
        "mean_latency_score": episodes["latency_score"]["mean"],
        "mean_throughput": episodes["throughput"]["mean"],
        "mean_recovery_speed": episodes["recovery_speed"]["mean"],
        "mean_cost_efficiency": episodes["cost_efficiency"]["mean"],
        "mean_reward": episodes["cumulative_reward"]["mean"],
        "std_reward": episodes["cumulative_reward"]["std"],
        "success_rate": episodes["task_success"]["mean"],
    }


def compare_agent_runs(run_paths: Iterable[str | Path]) -> dict[str, Any]:
    """Build side-by-side comparison and significance metadata for benchmark runs."""
    dirs = [Path(path) for path in run_paths]
    run_summaries = [summarize_run_for_leaderboard(path) for path in dirs]
    leaderboard = build_leaderboard_rows(run_summaries)
    comparisons: list[dict[str, Any]] = []
    if len(dirs) >= 2:
        base_metrics = _load_run_metrics(dirs[0])
        base_scores = [float(row.get("dime_index", 0.0)) for row in base_metrics]
        for other_dir in dirs[1:]:
            other_metrics = _load_run_metrics(other_dir)
            other_scores = [float(row.get("dime_index", 0.0)) for row in other_metrics]
            comparisons.append(
                {
                    "left": dirs[0].name,
                    "right": other_dir.name,
                    "metric": "dime_index",
                    "significance": select_significance_test(base_scores, other_scores),
                }
            )
    return {"leaderboard": leaderboard, "comparisons": comparisons}
