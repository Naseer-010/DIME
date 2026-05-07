"""Official DIME Index calculation."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, variance
from typing import Any, Iterable, Mapping

from benchmark.benchmark_config import DIME_V1_CONFIG
from benchmark.utils import clamp


LATENCY_INVERSE_MINMAX = "inverse_minmax"
LATENCY_SMOOTH_EXPONENTIAL = "smooth_exponential"


def latency_inverse_minmax(
    latency_ms: float,
    *,
    target_latency: float = 50.0,
    max_latency: float = 500.0,
) -> float:
    """Method A: inverse min-max latency normalization."""
    denom = max(max_latency - target_latency, 1e-9)
    return 1.0 - clamp((float(latency_ms) - target_latency) / denom)


def latency_smooth_exponential(latency_ms: float, *, latency_scale: float = 100.0) -> float:
    """Method B: smooth exponential latency normalization."""
    return clamp(math.exp(-float(latency_ms) / max(latency_scale, 1e-9)))


def normalize_latency(
    latency_ms: float,
    method: str,
    config: Mapping[str, Any] | None = None,
) -> float:
    """Normalize latency to [0, 1] with the selected method."""
    norm = config or DIME_V1_CONFIG.normalization_method
    if method == LATENCY_INVERSE_MINMAX:
        return latency_inverse_minmax(
            latency_ms,
            target_latency=float(norm.get("target_latency_ms", 50.0)),
            max_latency=float(norm.get("max_latency_ms", 500.0)),
        )
    if method == LATENCY_SMOOTH_EXPONENTIAL:
        return latency_smooth_exponential(
            latency_ms,
            latency_scale=float(norm.get("latency_scale_ms", 100.0)),
        )
    raise ValueError(f"Unknown latency normalization method: {method}")


def normalize_metrics(
    metrics: Mapping[str, Any],
    *,
    latency_method: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Normalize canonical DIME metrics to [0, 1]."""
    norm = config or DIME_V1_CONFIG.normalization_method
    mttr = float(metrics.get("mttr", metrics.get("MTTR", 0.0)) or 0.0)
    recovery_window = float(norm.get("max_allowed_recovery_time", 10.0))
    resource_cost = float(metrics.get("resource_cost", 0.0) or 0.0)
    max_budget = float(metrics.get("max_budget", metrics.get("initial_cloud_budget", 1.0)) or 1.0)

    return {
        "uptime": clamp(float(metrics.get("uptime", metrics.get("uptime_ratio", 0.0)) or 0.0)),
        "latency_score": normalize_latency(
            float(metrics.get("p99_latency", metrics.get("latency_ms", 0.0)) or 0.0),
            latency_method,
            norm,
        ),
        "throughput": clamp(float(metrics.get("throughput", metrics.get("throughput_ratio", 0.0)) or 0.0)),
        "recovery_speed": 1.0 - clamp(mttr / max(recovery_window, 1e-9)),
        "cost_efficiency": 1.0 - clamp(resource_cost / max(max_budget, 1e-9)),
    }


def compute_dime_index(
    metrics: Mapping[str, Any],
    config_snapshot: Mapping[str, Any] | None = None,
    *,
    latency_method: str | None = None,
) -> dict[str, float]:
    """Compute the official DIME Index and normalized metric breakdown."""
    snapshot = config_snapshot or {}
    method = latency_method or str(snapshot.get("selected_latency_method") or LATENCY_SMOOTH_EXPONENTIAL)
    norm = snapshot.get("normalization_method") if isinstance(snapshot, Mapping) else None
    normalized = normalize_metrics(metrics, latency_method=method, config=norm)
    weights = (
        snapshot.get("metric_weights")
        if isinstance(snapshot, Mapping) and "metric_weights" in snapshot
        else DIME_V1_CONFIG.metric_weights
    )
    score = sum(float(weights[key]) * normalized[key] for key in normalized)
    return {"dime_index": round(clamp(score), 6), **{k: round(v, 6) for k, v in normalized.items()}}


def _rank(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    mean_a = mean(a)
    mean_b = mean(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denom_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    denom_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if denom_a == 0.0 or denom_b == 0.0:
        return 0.0
    return numerator / (denom_a * denom_b)


def _method_quality(records: list[Mapping[str, Any]], method: str) -> dict[str, float]:
    latencies = [float(r.get("p99_latency", r.get("latency_ms", 0.0)) or 0.0) for r in records]
    scores = [normalize_latency(lat, method) for lat in latencies]
    outcomes = [
        float(r.get("task_success", 0.0) or 0.0) + float(r.get("task_score", 0.0) or 0.0)
        for r in records
    ]

    rank_consistency = abs(_pearson([float(v) for v in _rank(scores)], [float(v) for v in _rank(outcomes)]))
    var = variance(scores) if len(scores) > 1 else 0.0
    variance_stability = 1.0 / (1.0 + var)
    sorted_scores = [score for _, score in sorted(zip(latencies, scores), key=lambda item: item[0])]
    jumps = [abs(b - a) for a, b in zip(sorted_scores, sorted_scores[1:])]
    smoothness = 1.0 / (1.0 + (max(jumps) if jumps else 0.0))

    by_task: dict[str, list[float]] = defaultdict(list)
    for record, score in zip(records, scores):
        by_task[str(record.get("task_id", record.get("task", "unknown")))].append(score)
    task_means = [mean(values) for values in by_task.values() if values]
    between = variance(task_means) if len(task_means) > 1 else 0.0
    within_values = []
    for values in by_task.values():
        if len(values) > 1:
            within_values.append(variance(values))
    within = mean(within_values) if within_values else 0.0
    separability = clamp(between / (between + within + 1e-9))

    aggregate = mean([rank_consistency, variance_stability, smoothness, separability])
    return {
        "ranking_consistency": round(rank_consistency, 6),
        "variance_stability": round(variance_stability, 6),
        "score_smoothness": round(smoothness, 6),
        "task_separability": round(separability, 6),
        "aggregate": round(aggregate, 6),
    }


def select_latency_normalization(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate both latency normalization candidates and select the better one."""
    data = list(records)
    if not data:
        return {
            "selected_method": LATENCY_SMOOTH_EXPONENTIAL,
            "method_scores": {
                LATENCY_INVERSE_MINMAX: {"aggregate": 0.0},
                LATENCY_SMOOTH_EXPONENTIAL: {"aggregate": 0.0},
            },
        }

    method_scores = {
        LATENCY_INVERSE_MINMAX: _method_quality(data, LATENCY_INVERSE_MINMAX),
        LATENCY_SMOOTH_EXPONENTIAL: _method_quality(data, LATENCY_SMOOTH_EXPONENTIAL),
    }
    selected = max(
        method_scores,
        key=lambda method: (
            method_scores[method]["aggregate"],
            method_scores[method].get("score_smoothness", 0.0),
        ),
    )
    return {"selected_method": selected, "method_scores": method_scores}
