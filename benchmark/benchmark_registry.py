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
    topology_seed: int | None = None
    traffic_burst_step: int | None = None
    failure_step: int | None = None

    @property
    def reset_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "task": self.task_id,
            "curriculum_level": self.curriculum_level,
            "topology_template": self.topology_template,
            "trace_offset": self.trace_offset,
        }
        if self.topology_seed is not None:
            kwargs["topology_seed"] = self.topology_seed
        if self.traffic_burst_step is not None:
            kwargs["traffic_burst_step"] = self.traffic_burst_step
        if self.failure_step is not None:
            kwargs["failure_step"] = self.failure_step
        return kwargs


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
        from benchmark.hidden_eval_registry import get_hidden_eval_specs

        return get_hidden_eval_specs(official=True)
    return get_public_task_specs(split_value)


def task_registry_snapshot(include_hidden: bool = False) -> Mapping[str, tuple[str, ...]]:
    """Immutable split-to-registry-id snapshot for benchmark configs."""
    snapshot: dict[str, tuple[str, ...]] = {
        Split.TRAIN.value: tuple(task.registry_id for task in TRAIN_TASKS),
        Split.VALIDATION.value: tuple(task.registry_id for task in VALIDATION_TASKS),
    }
    if include_hidden:
        from benchmark.hidden_eval_registry import hidden_registry_snapshot

        snapshot[Split.HIDDEN_EVAL.value] = hidden_registry_snapshot(official=True)
    return snapshot


def iter_all_specs(include_hidden: bool = True) -> Iterable[TaskSpec]:
    """Iterate registered specs; hidden specs are opt-in."""
    yield from TRAIN_TASKS
    yield from VALIDATION_TASKS
    if include_hidden:
        from benchmark.hidden_eval_registry import get_hidden_eval_specs

        yield from get_hidden_eval_specs(official=True)
