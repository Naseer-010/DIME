"""Immutable DIME-v1.0 benchmark definition."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from benchmark.benchmark_registry import task_registry_snapshot


@dataclass(frozen=True)
class EvaluationProtocol:
    """Locked evaluation protocol for a benchmark version."""

    episodes_per_task: int
    seeds: tuple[int, ...]
    inference_only: bool
    disable_online_learning: bool


@dataclass(frozen=True)
class DeterministicPolicy:
    """Seed and replay requirements."""

    seed_components: tuple[str, ...]
    torch_deterministic: bool
    trace_wraparound: str
    replay_validation_required: bool


@dataclass(frozen=True)
class TopologyConstraints:
    """Allowed constrained topology templates for DIME-v1.0."""

    node_count: int
    database_node: int
    templates: tuple[str, ...]
    app_nodes: tuple[int, ...]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Frozen benchmark-critical configuration."""

    benchmark_name: str
    benchmark_version: str
    task_registry: Mapping[str, tuple[str, ...]]
    evaluation_protocol: EvaluationProtocol
    metric_weights: Mapping[str, float]
    normalization_method: Mapping[str, object]
    deterministic_policy: DeterministicPolicy
    topology_constraints: TopologyConstraints


_METRIC_WEIGHTS = MappingProxyType(
    {
        "uptime": 0.35,
        "latency_score": 0.25,
        "throughput": 0.20,
        "recovery_speed": 0.10,
        "cost_efficiency": 0.10,
    }
)

_NORMALIZATION_METHOD = MappingProxyType(
    {
        "latency": "auto",
        "latency_candidates": ("inverse_minmax", "smooth_exponential"),
        "target_latency_ms": 50.0,
        "max_latency_ms": 500.0,
        "latency_scale_ms": 100.0,
        "max_allowed_recovery_time": 10.0,
        "max_budget": "episode_initial_cloud_budget",
        "selection_persistence": "run_config_snapshot",
    }
)


DIME_V1_CONFIG = BenchmarkConfig(
    benchmark_name="DIME",
    benchmark_version="DIME-v1.0",
    task_registry=MappingProxyType(dict(task_registry_snapshot(include_hidden=False))),
    evaluation_protocol=EvaluationProtocol(
        episodes_per_task=100,
        seeds=tuple(range(100)),
        inference_only=True,
        disable_online_learning=True,
    ),
    metric_weights=_METRIC_WEIGHTS,
    normalization_method=_NORMALIZATION_METHOD,
    deterministic_policy=DeterministicPolicy(
        seed_components=("seed", "task", "topology_template", "trace_offset"),
        torch_deterministic=True,
        trace_wraparound="(step + trace_offset) % trace_length",
        replay_validation_required=True,
    ),
    topology_constraints=TopologyConstraints(
        node_count=8,
        database_node=0,
        templates=("default", "app_ring", "dense_mesh", "sampled_mesh"),
        app_nodes=(1, 2, 3, 4, 5, 6, 7),
    ),
)
