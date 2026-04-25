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
class FormatVerifier:
    """
    +1.0 for outputting a syntactically correct response/action format.

    ``last_response_format_valid`` is set by the environment. For structured
    actions this means valid JSON reached the server; for raw commands it means
    the parser found the required <reasoning> XML block and JSON command body.
    """

    name: str = "format"
    weight: float = 1.0

    def score(self, sim: "SimulationState") -> float:
        return self.weight if sim.last_response_format_valid else 0.0


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
    Smooth bounded sigmoid latency penalty.

    Keeps the SLA threshold informative without the zero-gradient cliff from a
    binary pass/fail reward.
    """

    name: str = "latency"
    alpha: float = 0.001
    beta: float = 0.3
    gamma: float = 0.1
    tau: float = 50.0

    def score(self, sim: "SimulationState") -> float:
        lat = sim.latency_ms
        x = max(-60.0, min(60.0, self.gamma * (lat - self.tau)))
        sigmoid = 1.0 / (1.0 + math.exp(-x))
        return -(self.alpha * lat) - (self.beta * sigmoid)


@dataclass
class CascadePBRSVerifier:
    """
    Potential-Based Reward Shaping for cluster stress.

    Rewards reduction in overload stress between consecutive states while
    preserving the optimal policy under Ng et al.'s PBRS formulation.
    """

    name: str = "cascade_pbrs"
    beta_stress: float = 1.0
    tau_stress: float = 0.85
    gamma: float = 0.99

    def _phi(self, nodes: List["Node"]) -> float:
        stress = sum(
            max(0.0, n.cpu_util - self.tau_stress) ** 2
            for n in nodes
            if not n.is_failed
        )
        return -self.beta_stress * stress

    def score(self, sim: "SimulationState") -> float:
        if not sim.previous_nodes:
            return 0.0
        phi_current = self._phi(sim.nodes)
        phi_prev = self._phi(sim.previous_nodes)
        return (self.gamma * phi_current) - phi_prev


@dataclass
class EconomicEfficiencyVerifier:
    """
    Linear active-node cost plus L1 churn penalty.

    Penalizes cloud footprint every step and discourages flapping capacity up
    and down across consecutive transitions.
    """

    name: str = "efficiency"
    w_cost: float = 0.1
    w_churn: float = 0.2
    n_max: float = 20.0

    def score(self, sim: "SimulationState") -> float:
        n_active = sum(1 for n in sim.nodes if not n.is_failed)
        prev_active = sim.previous_active_nodes
        if prev_active is None:
            prev_active = n_active

        delta_n = abs(n_active - prev_active)
        cost_penalty = self.w_cost * (n_active / self.n_max)
        churn_penalty = self.w_churn * (delta_n / self.n_max)
        return -(cost_penalty + churn_penalty)


# Backward-compatible import names for code that referenced the old verifiers.
SLAVerifier = SafeSliceLatencyVerifier
EfficiencyVerifier = EconomicEfficiencyVerifier


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
        if sim.throttle_rate < self.min_throughput_ratio:
            return self.penalty
        return 0.0


# ---------------------------------------------------------------------------
# Default rubric set & composite reward
# ---------------------------------------------------------------------------

DEFAULT_RUBRICS: List[Verifier] = [
    FormatVerifier(),
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
    return round(total, 4), breakdown
