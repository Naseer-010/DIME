"""Evaluation-only hidden DIME task registry."""

from __future__ import annotations

from benchmark.benchmark_registry import Split, TaskSpec


_HIDDEN_EVAL_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        "hidden.retry_storm.default.011",
        "retry_storm",
        Split.HIDDEN_EVAL,
        5,
        "default",
        11,
        ("trace",),
        topology_seed=101,
        traffic_burst_step=4,
    ),
    TaskSpec(
        "hidden.black_swan.default.029",
        "black_swan_az_failure",
        Split.HIDDEN_EVAL,
        5,
        "default",
        29,
        ("trace",),
        topology_seed=203,
        failure_step=3,
    ),
    TaskSpec(
        "hidden.connection_pool.default.053",
        "connection_pool_deadlock",
        Split.HIDDEN_EVAL,
        5,
        "default",
        53,
        ("trace",),
        topology_seed=307,
        failure_step=6,
    ),
    TaskSpec(
        "hidden.autoscaler.default.089",
        "autoscaler_flapping_trap",
        Split.HIDDEN_EVAL,
        5,
        "default",
        89,
        ("trace",),
        topology_seed=409,
        traffic_burst_step=7,
    ),
    TaskSpec(
        "hidden.retry_storm.ring.137",
        "retry_storm",
        Split.HIDDEN_EVAL,
        5,
        "app_ring",
        137,
        ("topology_variant", "trace"),
        topology_seed=503,
        traffic_burst_step=5,
    ),
    TaskSpec(
        "hidden.black_swan.dense.211",
        "black_swan_az_failure",
        Split.HIDDEN_EVAL,
        5,
        "dense_mesh",
        211,
        ("topology_variant", "trace"),
        topology_seed=607,
        failure_step=5,
    ),
    TaskSpec(
        "hidden.connection_pool.ring.307",
        "connection_pool_deadlock",
        Split.HIDDEN_EVAL,
        5,
        "app_ring",
        307,
        ("topology_variant", "trace"),
        topology_seed=709,
        failure_step=8,
    ),
    TaskSpec(
        "hidden.autoscaler.sampled.401",
        "autoscaler_flapping_trap",
        Split.HIDDEN_EVAL,
        5,
        "sampled_mesh",
        401,
        ("topology_variant", "trace"),
        topology_seed=811,
        traffic_burst_step=9,
    ),
)


def get_hidden_eval_specs(official: bool = False) -> tuple[TaskSpec, ...]:
    """Return hidden specs only for official benchmark evaluation."""
    if not official:
        raise PermissionError("hidden_eval registry is evaluation-only")
    return _HIDDEN_EVAL_TASKS


def hidden_registry_snapshot(official: bool = False) -> tuple[str, ...]:
    """Return hidden registry IDs for official run snapshots."""
    return tuple(spec.registry_id for spec in get_hidden_eval_specs(official=official))
