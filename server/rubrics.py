"""
Composable Rubric System for DIME.

Each verifier is an independent scoring module that evaluates one aspect
of the agent's performance.  ``compute_composite_reward`` aggregates all
verifiers into a single dense reward signal without coupling them.

Design rationale
----------------
OpenEnv judging criteria explicitly favour **composable rubrics** over a
monolithic scoring equation.  Each verifier can be toggled, re-weighted,
or extended independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Protocol

if TYPE_CHECKING:
    from server.environment import Node, SimulationState


# ---------------------------------------------------------------------------
# Verifier protocol – any object with a ``score`` method qualifies
# ---------------------------------------------------------------------------


class Verifier(Protocol):
    """Minimal interface every rubric verifier must satisfy."""

    name: str

    def score(self, sim: "SimulationState") -> float:
        """Return a reward component for the current step."""
        ...


# ---------------------------------------------------------------------------
# Concrete verifiers
# ---------------------------------------------------------------------------


@dataclass
class CoTFormatVerifier:
    """
    +0.05 for outputting a syntactically correct response/action format.
    -0.5 penalty for unparseable actions.

    Scaled down to prevent loss imbalance (the agent shouldn't get high scores
    just for talking correctly while the cluster burns).
    """

    name: str = "format"
    weight: float = 0.05
    penalty: float = -0.5

    def score(self, sim: "SimulationState") -> float:
        # Assuming the environment sets sim.last_action_valid or sim.last_response_format_valid
        is_valid = getattr(
            sim, "last_response_format_valid", getattr(sim, "last_action_valid", False)
        )
        return self.weight if is_valid else self.penalty


@dataclass
class StabilityVerifier:
    """
    +0.4 for maintaining 100 % uptime during the current step.

    All nodes must be alive (not failed, not restarting).
    """

    name: str = "stability"
    weight: float = 0.4

    def score(self, sim: "SimulationState") -> float:
        if not sim.nodes:
            return 0.0
        alive = sum(1 for n in sim.nodes if not n.is_failed)
        return self.weight if alive == len(sim.nodes) else 0.0


@dataclass
class SafeSliceLatencyVerifier:
    """
    Bounded sigmoid penalty for latency to prevent vanishing gradients.

    Keeps the SLA threshold informative without the zero-gradient cliff from a
    binary pass/fail reward. Multiplied by latency to heavily punish saturation.
    """

    name: str = "latency"
    alpha: float = 0.01
    beta: float = 5.0
    gamma: float = 0.1
    sla_latency: float = 50.0

    def score(self, sim: "SimulationState") -> float:
        lat = sim.latency_ms
        # Clipped to prevent math.exp overflow
        x = max(-10.0, min(10.0, self.gamma * (lat - self.sla_latency)))
        sig = 1.0 / (1.0 + math.exp(-x))
        return -(self.alpha * lat) - (self.beta * lat * sig)


@dataclass
class CascadePBRSVerifier:
    """
    Potential-Based Reward Shaping for cluster stress with Velocity penalty.

    Rewards reduction in overload stress between consecutive states while
    penalizing rapid load oscillation (velocity) to stop "loop-and-farm" exploits.
    """

    name: str = "cascade_pbrs"
    tau_stress: float = 0.85
    beta_stress: float = 10.0
    gamma: float = 0.99
    lambda_velocity: float = 2.0

    def _potential(self, nodes: List["Node"]) -> float:
        stress = sum(
            max(0.0, n.cpu_util - self.tau_stress) ** 2
            for n in nodes
            if not n.is_failed
        )
        return -self.beta_stress * stress

    def score(self, sim: "SimulationState") -> float:
        current_potential = self._potential(sim.nodes)

        # If no previous load data, initialize and return 0
        if not getattr(sim, "prev_node_loads", None):
            sim.prev_potential = current_potential
            return 0.0

        current_loads = [n.cpu_util for n in sim.nodes]
        # Calculate load velocity (squared difference)
        velocity = sum(
            (cur - prev) ** 2
            for cur, prev in zip(
                current_loads, sim.prev_node_loads[: len(current_loads)]
            )
        )

        prev_pot = getattr(sim, "prev_potential", current_potential)
        reward = (
            (self.gamma * current_potential)
            - prev_pot
            - (self.lambda_velocity * velocity)
        )

        # Set current potential for the next step calculation
        sim.prev_potential = current_potential
        return reward


@dataclass
class EconomicEfficiencyVerifier:
    """
    Couples w_cost with latency so lazy policies get punished, plus L1 churn penalty.
    """

    name: str = "efficiency"
    w_cost: float = 1.0
    w_churn: float = 0.2
    kappa_lat: float = 0.5
    sla_latency: float = 50.0

    def score(self, sim: "SimulationState") -> float:
        active_nodes = sum(1 for n in sim.nodes if not n.is_failed)
        total_nodes = max(len(sim.nodes), 1)

        node_ratio = active_nodes / total_nodes

        prev_active = getattr(sim, "prev_active_nodes", active_nodes)
        churn_ratio = abs(active_nodes - prev_active) / total_nodes

        # High latency makes nodes "cost" more, forcing the agent to solve the latency
        # issue rather than just blindly saving money.
        lat_penalty_multiplier = 1.0 + self.kappa_lat * (
            sim.latency_ms / self.sla_latency
        )

        return -(self.w_cost * node_ratio * lat_penalty_multiplier) - (
            self.w_churn * churn_ratio
        )


@dataclass
class ThroughputVerifier:
    """
    -0.5 penalty if the agent drops more than 70 % of traffic via throttle.

    Patches the "zero-service" exploit where an agent sets throttle to 0.0,
    emptying queues and achieving low latency / high uptime while serving
    nothing.
    """

    name: str = "throughput"
    penalty: float = -0.5
    min_throughput_ratio: float = 0.30  # must serve ≥30% of traffic

    def score(self, sim: "SimulationState") -> float:
        if getattr(sim, "throttle_rate", 1.0) < self.min_throughput_ratio:
            return self.penalty
        return 0.0


# Backward-compatible import names for code that referenced the old verifiers.
SLAVerifier = SafeSliceLatencyVerifier
EfficiencyVerifier = EconomicEfficiencyVerifier


# ---------------------------------------------------------------------------
# Default rubric set & composite reward
# ---------------------------------------------------------------------------

DEFAULT_RUBRICS: List[Verifier] = [
    CoTFormatVerifier(),
    StabilityVerifier(),
    SafeSliceLatencyVerifier(),
    CascadePBRSVerifier(),
    EconomicEfficiencyVerifier(),
    ThroughputVerifier(),
]


def compute_composite_reward(
    sim: "SimulationState",
    rubrics: List[Verifier] | None = None,
) -> tuple[float, Dict[str, float]]:
    """
    Aggregate independent verifier scores.

    Returns
    -------
    reward : float
        Sum of all verifier scores, rounded to 4 decimal places.
    breakdown : dict[str, float]
        Per-verifier score for transparency / logging.
    """
    rubrics = rubrics or DEFAULT_RUBRICS
    breakdown: Dict[str, float] = {}
    total = 0.0

    for v in rubrics:
        s = v.score(sim)
        breakdown[v.name] = round(s, 4)
        total += s

    # Clip the final sum to [-10, 10] to stabilize RL gradients
    total_clipped = max(-10.0, min(10.0, total))

    return round(total_clipped, 4), breakdown
