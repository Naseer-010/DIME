"""Seeded random DIME baseline."""

from __future__ import annotations

import random
from typing import Any

from agents.base_agent import BaseAgent
from benchmark.utils import observation_to_dict
from server.models import InfraAction


class RandomAgent(BaseAgent):
    """Uniformly sample valid DIME management actions with deterministic seeding."""

    def __init__(self, seed: int = 0) -> None:
        self._base_seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        self._rng = random.Random(self._base_seed if seed is None else seed)

    def act(self, observation: Any) -> InfraAction:
        obs = observation_to_dict(observation)
        node_count = max(1, len(obs.get("cpu_loads", []) or [0]))
        action_type = self._rng.choice(
            ["restart_node", "reroute_traffic", "throttle", "scale_up", "no_op"]
        )

        if action_type == "restart_node":
            failed = list(obs.get("failed_nodes", []) or [])
            target = int(self._rng.choice(failed)) if failed else self._rng.randrange(node_count)
            return InfraAction(action_type="restart_node", target=target)

        if action_type == "reroute_traffic" and node_count > 1:
            src = self._rng.randrange(node_count)
            dst_choices = [idx for idx in range(node_count) if idx != src]
            return InfraAction(
                action_type="reroute_traffic",
                from_node=src,
                to_node=self._rng.choice(dst_choices),
            )

        if action_type == "throttle":
            return InfraAction(action_type="throttle", rate=self._rng.choice([0.3, 0.5, 0.7, 0.9]))

        if action_type == "scale_up":
            return InfraAction(action_type="scale_up")

        return InfraAction(action_type="no_op")
