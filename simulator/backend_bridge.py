from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from server.environment import DistributedInfraEnvironment
from server.models import InfraAction, InfraObservation

TASK_IDS = ("traffic_spike", "node_failure", "cascading_failure")


@dataclass
class BackendStep:
    observation: InfraObservation
    reward: float
    done: bool
    metadata: dict


class EmbeddedBackendBridge:
    """Thin adapter over the in-process backend environment."""

    def __init__(self) -> None:
        self._env = DistributedInfraEnvironment()
        self._last_observation: Optional[InfraObservation] = None

    @property
    def env(self) -> DistributedInfraEnvironment:
        return self._env

    @property
    def last_observation(self) -> Optional[InfraObservation]:
        return self._last_observation

    @property
    def adjacency(self) -> dict[int, list[int]]:
        return {
            node_index: list(neighbors)
            for node_index, neighbors in self._env.sim.adjacency.items()
        }

    def reset(self, *, seed: int | None = None, task_id: str = "traffic_spike") -> BackendStep:
        observation = self._env.reset(seed=seed, task=task_id)
        self._last_observation = observation
        return BackendStep(
            observation=observation,
            reward=float(observation.reward or 0.0),
            done=bool(observation.done),
            metadata=dict(observation.metadata or {}),
        )

    def step(self, action: InfraAction) -> BackendStep:
        observation = self._env.step(action)
        self._last_observation = observation
        return BackendStep(
            observation=observation,
            reward=float(observation.reward or 0.0),
            done=bool(observation.done),
            metadata=dict(observation.metadata or {}),
        )

    def close(self) -> None:
        self._env.close()
