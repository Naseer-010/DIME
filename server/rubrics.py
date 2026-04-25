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

import numpy as np

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


# ---------------------------------------------------------------------------
# Production-grade SRE reward (used for RL training signal)
# ---------------------------------------------------------------------------


class ProductionSREReward:
    def __init__(self):
        # Asymmetric Topology Coefficients
        self.c_db = 100.0
        self.lambda_db = 4.0
        self.c_worker = 1.0

        # Load Shedding Economics
        self.b_max = 100.0
        self.psi_shed = 1.0
        self.delta_shed = 4.0

        # Multi-Dimensional Stress
        self.a_cpu = 10.0
        self.b_mem = 5.0
        self.c_mem = 2.0

        # Temporal Friction Tracking
        self.scale_cooldown_queue = 0.0

    def reset(self) -> None:
        self.scale_cooldown_queue = 0.0

    def calculate_reward(self, state: dict, action: dict) -> float:
        reward = 0.0

        # --- 1. Asymmetric Topology (Heart vs. Pinky) ---
        # If DB dies, it's game over.
        if 0 in state.get("failed_nodes", []):
            return -1000.0

        db_cpu = float(state["cpu_loads"][0]) if state.get("cpu_loads") else 0.0
        # Exponential DB Penalty: Explodes as it nears 1.0
        r_topo = -self.c_db * np.exp(self.lambda_db * db_cpu)

        # Quadratic Worker Penalty: Soft degradation
        worker_cpus = state.get("cpu_loads", [0.0] * 8)[1:]
        r_topo -= self.c_worker * sum(float(cpu) ** 2 for cpu in worker_cpus)
        reward += float(r_topo)

        # --- 2. Economics of Load Shedding (Error Budget) ---
        error_budget = float(state.get("error_budget", self.b_max))
        if action.get("action_type") == "throttle":
            rate = float(action.get("rate", 1.0))
            shed_amount = 1.0 - rate

            # Scarcity Multiplier: Throttling costs more when budget is low
            budget_depletion_factor = (self.b_max / max(1.0, error_budget)) ** 2
            r_shed = -self.psi_shed * shed_amount * (
                1.0 + self.delta_shed * budget_depletion_factor
            )
            reward += float(r_shed)

        # --- 3. Multi-Dimensional Stress (CPU Slope vs Mem Cliff) ---
        mem_usages = state.get("mem_utilizations", [0.4] * 8)
        r_mem = 0.0
        for mem in mem_usages:
            mem = float(mem)
            if mem >= 0.98:
                r_mem -= 500.0  # OOM Terminal Penalty
            else:
                # Hyperbolic approach slope: diverges as mem->1.0
                r_mem -= self.b_mem * np.exp(1.0 / max(0.01, 1.0 - mem))
        reward += float(r_mem)

        # --- 4. Temporal Friction (Cold Start Integral) ---
        if action.get("action_type") == "scale_up":
            self.scale_cooldown_queue += 3 * sum(state.get("queue_lengths", [10]))

        if self.scale_cooldown_queue > 0:
            reward -= 0.1 * self.scale_cooldown_queue
            self.scale_cooldown_queue = max(
                0.0,
                self.scale_cooldown_queue - sum(state.get("queue_lengths", [10])),
            )

        # --- 5. Tail Latency (p99) ---
        p99 = float(state.get("p99_latency", float(state.get("latency_ms", 0.0)) * 1.5))
        if p99 > 50.0:
            reward -= 10.0 * (((p99 - 50.0) / 10.0) ** 2.5)

        return float(np.clip(reward, -1000.0, 10.0))


_SRE_ENGINE: ProductionSREReward | None = None
_SRE_ENGINE_EPISODE: str | None = None


def _get_sre_engine(sim: "SimulationState") -> ProductionSREReward:
    global _SRE_ENGINE, _SRE_ENGINE_EPISODE
    if _SRE_ENGINE is None:
        _SRE_ENGINE = ProductionSREReward()
        _SRE_ENGINE_EPISODE = None

    eid = getattr(sim, "episode_id", None)
    if eid is not None and eid != _SRE_ENGINE_EPISODE:
        _SRE_ENGINE.reset()
        _SRE_ENGINE_EPISODE = eid
    return _SRE_ENGINE


def build_production_state(sim: "SimulationState") -> dict:
    cpu_loads = [float(n.cpu_util) for n in sim.nodes]
    mem_utils = [float(getattr(n, "memory_util", 0.0)) for n in sim.nodes]
    queue_lengths = [int(getattr(n, "queue_length", 0)) for n in sim.nodes]
    failed = [i for i, n in enumerate(sim.nodes) if getattr(n, "is_failed", False)]

    return {
        "cpu_loads": cpu_loads,
        "mem_utilizations": mem_utils,
        "queue_lengths": queue_lengths,
        "failed_nodes": failed,
        "latency_ms": float(getattr(sim, "latency_ms", 0.0)),
        "p99_latency": float(getattr(sim, "last_trace_p99_latency", 0.0)),
        "io_wait": float(getattr(sim, "last_trace_node_0_io", 0.0)),
        "error_budget": float(getattr(sim, "error_budget", 100.0)),
    }


def build_production_action(sim: "SimulationState") -> dict:
    act = str(getattr(sim, "last_action_type", "no_op") or "no_op")
    action: dict = {"action_type": act}
    if act == "throttle":
        action["rate"] = float(getattr(sim, "throttle_rate", 1.0))
    return action


def calculate_step_reward(sim: "SimulationState", is_dead: bool = False) -> float:
    """
    Training reward wrapper for production-grade SRE math.

    Notes:
    - Keeps existing OpenEnv rubric breakdown intact elsewhere.
    - Returns terminal penalty if cluster is dead or DB failed.
    """
    state = build_production_state(sim)
    if is_dead or 0 in state.get("failed_nodes", []):
        return -1000.0

    engine = _get_sre_engine(sim)
    action = build_production_action(sim)
    return float(engine.calculate_reward(state, action))
