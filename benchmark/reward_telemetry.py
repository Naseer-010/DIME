"""Reward normalization and verifier telemetry for DIME benchmarks."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, variance
from typing import Any, Iterable, Mapping, Sequence


NORMALIZATION_METHODS = ("tanh", "smooth_sigmoid")
SCALE_CANDIDATES = (0.5, 1.0, 2.0, 5.0, 10.0)
ABLATION_VERIFIERS = ("latency", "cascade_pbrs", "throughput", "efficiency")


@dataclass(frozen=True)
class RewardNormalizationSelection:
    """Selected verifier normalization method and scale."""

    method: str
    scale_factor: float
    method_scores: dict[str, dict[str, float]]


def normalize_reward_value(raw_reward: float, method: str, scale_factor: float) -> float:
    """Normalize a verifier output to [-1, 1] with a smooth monotonic transform."""
    scale = max(float(scale_factor), 1e-9)
    x = float(raw_reward) / scale
    if method == "tanh":
        return math.tanh(x)
    if method == "smooth_sigmoid":
        # Algebraically equivalent to tanh(x / 2), kept explicit for auditability.
        if x >= 0:
            z = math.exp(-x)
            return (2.0 / (1.0 + z)) - 1.0
        z = math.exp(x)
        return (2.0 * z / (1.0 + z)) - 1.0
    raise ValueError(f"Unknown reward normalization method: {method}")


def _values_by_verifier(records: Iterable[Mapping[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        breakdown = record.get("rubric_breakdown", record)
        if not isinstance(breakdown, Mapping):
            continue
        for verifier, value in breakdown.items():
            if verifier == "pbrs_components":
                continue
            try:
                values[str(verifier)].append(float(value))
            except (TypeError, ValueError):
                continue
    return dict(values)


def _variance(values: Sequence[float]) -> float:
    return variance(values) if len(values) > 1 else 0.0


def _temporal_stability(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 1.0
    deltas = [abs(b - a) for a, b in zip(values, values[1:])]
    return 1.0 / (1.0 + mean(deltas))


def _imbalance(normalized: Mapping[str, Sequence[float]]) -> float:
    magnitudes = [mean(abs(v) for v in values) for values in normalized.values() if values]
    if len(magnitudes) < 2:
        return 0.0
    max_mag = max(magnitudes)
    rest = [v for v in magnitudes if v != max_mag]
    other_mean = mean(rest) if rest else min(magnitudes)
    return max_mag / max(other_mean, 1e-9)


def _score_candidate(values: Mapping[str, Sequence[float]], method: str, scale: float) -> dict[str, float]:
    normalized = {
        verifier: [normalize_reward_value(value, method, scale) for value in raw_values]
        for verifier, raw_values in values.items()
    }
    all_values = [value for raw_values in normalized.values() for value in raw_values]
    saturation_rate = (
        sum(1 for value in all_values if abs(value) > 0.95) / len(all_values)
        if all_values
        else 0.0
    )
    variances = [_variance(raw_values) for raw_values in normalized.values() if raw_values]
    variance_consistency = 1.0 / (1.0 + _variance(variances))
    temporal = mean(_temporal_stability(raw_values) for raw_values in normalized.values()) if normalized else 1.0
    imbalance = _imbalance(normalized)
    dominance_penalty = max(0.0, imbalance - 1.0)
    score = (
        variance_consistency
        + temporal
        + (1.0 - min(1.0, saturation_rate))
        + (1.0 / (1.0 + dominance_penalty))
    ) / 4.0
    return {
        "score": round(score, 6),
        "saturation_rate": round(saturation_rate, 6),
        "variance_consistency": round(variance_consistency, 6),
        "temporal_stability": round(temporal, 6),
        "dominance_imbalance": round(imbalance, 6),
    }


def evaluate_reward_normalization(
    step_breakdowns: Iterable[Mapping[str, Any]],
) -> RewardNormalizationSelection:
    """Evaluate both smooth reward normalizers and bounded scale candidates."""
    values = _values_by_verifier(step_breakdowns)
    if not values:
        return RewardNormalizationSelection(
            method="tanh",
            scale_factor=1.0,
            method_scores={"tanh:1.0": {"score": 0.0}, "smooth_sigmoid:1.0": {"score": 0.0}},
        )

    scores: dict[str, dict[str, float]] = {}
    for method in NORMALIZATION_METHODS:
        for scale in SCALE_CANDIDATES:
            scores[f"{method}:{scale}"] = _score_candidate(values, method, scale)

    selected_key = max(
        scores,
        key=lambda key: (
            scores[key]["saturation_rate"] <= 0.20,
            -scores[key]["dominance_imbalance"],
            -scores[key]["saturation_rate"],
            scores[key]["score"],
        ),
    )
    method, scale_text = selected_key.split(":", 1)
    return RewardNormalizationSelection(
        method=method,
        scale_factor=float(scale_text),
        method_scores=scores,
    )


def _summarize_verifier(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "variance": 0.0,
            "saturation_rate": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "temporal_stability": 1.0,
        }
    var = _variance(values)
    std = math.sqrt(var)
    return {
        "n": len(values),
        "mean": round(mean(values), 6),
        "std": round(std, 6),
        "variance": round(var, 6),
        "saturation_rate": round(sum(1 for value in values if abs(value) > 0.95) / len(values), 6),
        "positive_ratio": round(sum(1 for value in values if value > 0.0) / len(values), 6),
        "negative_ratio": round(sum(1 for value in values if value < 0.0) / len(values), 6),
        "temporal_stability": round(_temporal_stability(values), 6),
    }


def _dominance_flags(normalized: Mapping[str, Sequence[float]]) -> dict[str, dict[str, Any]]:
    variances = {name: _variance(values) for name, values in normalized.items()}
    total_variance = sum(variances.values())
    magnitudes = {name: mean(abs(value) for value in values) for name, values in normalized.items() if values}
    flags: dict[str, dict[str, Any]] = {}
    for name, values in normalized.items():
        variance_share = variances.get(name, 0.0) / max(total_variance, 1e-9)
        other_mags = [mag for other, mag in magnitudes.items() if other != name]
        mag_ratio = magnitudes.get(name, 0.0) / max(mean(other_mags) if other_mags else 0.0, 1e-9)
        flags[name] = {
            "dominant": variance_share > 0.50 or mag_ratio > 3.0,
            "variance_share": round(variance_share, 6),
            "magnitude_ratio": round(mag_ratio, 6),
        }
    return flags


def _ablation_report(normalized: Mapping[str, Sequence[float]], ablations: Sequence[str]) -> dict[str, Any]:
    max_len = max((len(v) for v in normalized.values()), default=0)
    baseline = [
        sum(values[idx] for values in normalized.values() if idx < len(values))
        for idx in range(max_len)
    ]
    report: dict[str, Any] = {"enabled": list(ablations), "baseline_mean": round(mean(baseline), 6) if baseline else 0.0}
    for verifier in ablations:
        if verifier not in ABLATION_VERIFIERS:
            continue
        kept = {name: values for name, values in normalized.items() if name != verifier}
        kept_max_len = max((len(v) for v in kept.values()), default=0)
        totals = [
            sum(values[idx] for values in kept.values() if idx < len(values))
            for idx in range(kept_max_len)
        ]
        report[verifier] = {
            "ablated_mean": round(mean(totals), 6) if totals else 0.0,
            "delta_mean": round((mean(totals) - mean(baseline)), 6) if totals and baseline else 0.0,
        }
    return report


def build_reward_telemetry(
    step_breakdowns: Iterable[Mapping[str, Any]],
    selection: RewardNormalizationSelection,
    ablations: Sequence[str] = (),
) -> dict[str, Any]:
    """Build verifier contribution telemetry across benchmark steps."""
    values = _values_by_verifier(step_breakdowns)
    normalized = {
        verifier: [
            normalize_reward_value(value, selection.method, selection.scale_factor)
            for value in raw_values
        ]
        for verifier, raw_values in values.items()
    }
    return {
        "normalization": {
            "method": selection.method,
            "scale_factor": selection.scale_factor,
            "candidate_scores": selection.method_scores,
        },
        "verifiers": {name: _summarize_verifier(vals) for name, vals in sorted(normalized.items())},
        "dominance": _dominance_flags(normalized),
        "ablation_report": _ablation_report(normalized, ablations),
    }
