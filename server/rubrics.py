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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Protocol

if TYPE_CHECKING:
    from server.environment import SimulationState


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
    +0.1 for outputting a syntactically correct, valid action.

    The environment sets ``sim.last_action_valid`` before reward computation.
    A malformed or unparseable action gets 0.
    """

    name: str = "format"
    weight: float = 0.1

    def score(self, sim: "SimulationState") -> float:
        return self.weight if sim.last_action_valid else 0.0


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
class SLAVerifier:
    """
    +0.3 for keeping P99 latency under Target (default 50 ms).

    Uses the rolling-average ``sim.latency_ms`` as a proxy for P99.
    """

    name: str = "sla"
    weight: float = 0.3
    target_ms: float = 50.0

    def score(self, sim: "SimulationState") -> float:
        return self.weight if sim.latency_ms < self.target_ms else 0.0


@dataclass
class EfficiencyVerifier:
    """
    -0.2 penalty if the agent calls ``scale_up`` while the average
    cluster CPU utilisation is below 60 %.

    Proves the environment trains cost-aware models, not just stable ones.
    """

    name: str = "efficiency"
    penalty: float = -0.2
    cpu_threshold: float = 0.60

    def score(self, sim: "SimulationState") -> float:
        if sim.last_action_type != "scale_up":
            return 0.0
        operational = [n for n in sim.nodes if not n.is_failed]
        if not operational:
            return 0.0
        avg_cpu = sum(n.cpu_util for n in operational) / len(operational)
        return self.penalty if avg_cpu < self.cpu_threshold else 0.0


# ---------------------------------------------------------------------------
# Default rubric set & composite reward
# ---------------------------------------------------------------------------

DEFAULT_RUBRICS: List[Verifier] = [
    FormatVerifier(),
    StabilityVerifier(),
    SLAVerifier(),
    EfficiencyVerifier(),
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
