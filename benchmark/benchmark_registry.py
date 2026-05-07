"""Frozen task split registry for DIME-v1.0."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping


class Split(str, Enum):
    """Canonical benchmark split names."""

    TRAIN = "train"
    VALIDATION = "validation"
    HIDDEN_EVAL = "hidden_eval"


@dataclass(frozen=True)
class TaskSpec:
    """One immutable benchmark task entry."""

    registry_id: str
    task_id: str
    split: Split
    curriculum_level: int
    topology_template: str = "default"
    trace_offset: int = 0
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def reset_kwargs(self) -> dict[str, object]:
        return {
            "task": self.task_id,
            "curriculum_level": self.curriculum_level,
            "topology_template": self.topology_template,
            "trace_offset": self.trace_offset,
        }


TRAIN_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("train.level_1_read_logs", "level_1_read_logs", Split.TRAIN, 1),
    TaskSpec("train.traffic_spike", "traffic_spike", Split.TRAIN, 2),
    TaskSpec("train.node_failure", "node_failure", Split.TRAIN, 2),
    TaskSpec("train.cascading_failure", "cascading_failure", Split.TRAIN, 3),
)

VALIDATION_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("validation.flash_crowd", "flash_crowd", Split.VALIDATION, 4),
    TaskSpec("validation.thundering_herd", "thundering_herd", Split.VALIDATION, 5, trace_offset=17),
    TaskSpec("validation.zombie_node", "zombie_node", Split.VALIDATION, 5, trace_offset=41),
    TaskSpec("validation.hot_shard_skew", "hot_shard_skew", Split.VALIDATION, 5, trace_offset=73),
)

_HIDDEN_EVAL_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("hidden.retry_storm.default.011", "retry_storm", Split.HIDDEN_EVAL, 5, "default", 11, ("trace",)),
    TaskSpec("hidden.black_swan.default.029", "black_swan_az_failure", Split.HIDDEN_EVAL, 5, "default", 29, ("trace",)),
    TaskSpec("hidden.connection_pool.default.053", "connection_pool_deadlock", Split.HIDDEN_EVAL, 5, "default", 53, ("trace",)),
    TaskSpec("hidden.autoscaler.default.089", "autoscaler_flapping_trap", Split.HIDDEN_EVAL, 5, "default", 89, ("trace",)),
    TaskSpec("hidden.retry_storm.ring.137", "retry_storm", Split.HIDDEN_EVAL, 5, "app_ring", 137, ("topology_variant", "trace")),
    TaskSpec("hidden.black_swan.dense.211", "black_swan_az_failure", Split.HIDDEN_EVAL, 5, "dense_mesh", 211, ("topology_variant", "trace")),
    TaskSpec("hidden.connection_pool.ring.307", "connection_pool_deadlock", Split.HIDDEN_EVAL, 5, "app_ring", 307, ("topology_variant", "trace")),
    TaskSpec("hidden.autoscaler.sampled.401", "autoscaler_flapping_trap", Split.HIDDEN_EVAL, 5, "sampled_mesh", 401, ("topology_variant", "trace")),
)


def get_training_task_ids() -> tuple[str, ...]:
    """Return only tasks permitted for RL training."""
    return tuple(task.task_id for task in TRAIN_TASKS)


def get_public_task_specs(split: Split | str) -> tuple[TaskSpec, ...]:
    """Return non-hidden task specs for public training/tuning use."""
    split_value = Split(split)
    if split_value is Split.TRAIN:
        return TRAIN_TASKS
    if split_value is Split.VALIDATION:
        return VALIDATION_TASKS
    raise PermissionError("hidden_eval tasks require the official benchmark harness")


def get_benchmark_task_specs(split: Split | str) -> tuple[TaskSpec, ...]:
    """Return task specs for the official evaluation harness."""
    split_value = Split(split)
    if split_value is Split.HIDDEN_EVAL:
        return _HIDDEN_EVAL_TASKS
    return get_public_task_specs(split_value)


def task_registry_snapshot(include_hidden: bool = True) -> Mapping[str, tuple[str, ...]]:
    """Immutable split-to-registry-id snapshot for benchmark configs."""
    snapshot: dict[str, tuple[str, ...]] = {
        Split.TRAIN.value: tuple(task.registry_id for task in TRAIN_TASKS),
        Split.VALIDATION.value: tuple(task.registry_id for task in VALIDATION_TASKS),
    }
    if include_hidden:
        snapshot[Split.HIDDEN_EVAL.value] = tuple(task.registry_id for task in _HIDDEN_EVAL_TASKS)
    return snapshot


def iter_all_specs(include_hidden: bool = True) -> Iterable[TaskSpec]:
    """Iterate registered specs; hidden specs are opt-in."""
    yield from TRAIN_TASKS
    yield from VALIDATION_TASKS
    if include_hidden:
        yield from _HIDDEN_EVAL_TASKS
