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
    Potential-Based Reward Shaping for cluster stress plus a separate velocity penalty.

    Formal shaping term:

        F(s, a, s') = gamma * Phi(s') - Phi(s)

    where Phi(s) is negative weighted cluster stress. Under the standard PBRS
    assumptions that state transitions are Markovian, gamma matches the task
    discount, and Phi depends only on observable simulator state, this shaping
    term preserves optimal policies. The velocity penalty below is deliberately
    reported and subtracted as a separate regularizer; it is not part of the
    policy-invariant PBRS proof and exists only to discourage oscillatory
    "loop-and-farm" behavior.
    """

    name: str = "cascade_pbrs"
    tau_stress: float = 0.85
    beta_stress: float = 10.0
    gamma: float = 0.99
    lambda_velocity: float = 2.0
    w_cpu: float = 1.0
    w_queue: float = 0.35
    w_memory: float = 0.75
    w_failed: float = 1.5

    def _potential(self, nodes: List["Node"]) -> float:
        if not nodes:
            return 0.0
        stress = 0.0
        for node in nodes:
            if node.is_failed:
                stress += self.w_failed
                continue
            cpu_stress = max(0.0, node.cpu_util - self.tau_stress) ** 2
            queue_stress = min(1.0, max(0.0, node.queue_length) / max(1.0, node.capacity * 4.0)) ** 2
            memory_stress = max(0.0, node.memory_util - 0.80) ** 2
            stress += (
                self.w_cpu * cpu_stress
                + self.w_queue * queue_stress
                + self.w_memory * memory_stress
            )
        return -self.beta_stress * stress

    def score(self, sim: "SimulationState") -> float:
        current_potential = self._potential(sim.nodes)

        # If no previous load data, initialize and return 0
        if not getattr(sim, "prev_node_loads", None):
            sim.prev_potential = current_potential
            sim.last_pbrs_components = {
                "potential_previous": current_potential,
                "potential_current": current_potential,
                "pbrs_component": 0.0,
                "velocity_penalty": 0.0,
                "total": 0.0,
            }
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
        pbrs_component = (self.gamma * current_potential) - prev_pot
        velocity_penalty = self.lambda_velocity * velocity
        reward = pbrs_component - velocity_penalty
        sim.last_pbrs_components = {
            "potential_previous": round(prev_pot, 6),
            "potential_current": round(current_potential, 6),
            "pbrs_component": round(pbrs_component, 6),
            "velocity_penalty": round(velocity_penalty, 6),
            "total": round(reward, 6),
        }

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
#
# Design goals (GRPO / PPO compatibility):
#   1.  Output bounded to roughly [-5.0, +5.0] so the advantage estimator
#       has healthy variance.  No hardcoded -1000.0 cliffs.
#   2.  Smooth, continuous penalties so policy gradients don't vanish.
#   3.  Catastrophic states (DB dead, total collapse) receive a bounded
#       maximum penalty (-5.0), not an infinite step-function drop.
# ---------------------------------------------------------------------------


class ProductionSREReward:
    """Bounded, gradient-friendly SRE reward for Policy Gradient RL."""

    def __init__(self):
        # Asymmetric Topology Coefficients (bounded)
        self.c_db = 1.5  # max ≈ 1.5 * e^(2*1.0) ≈ 11  → clipped later
        self.lambda_db = 2.0  # softer exponent than original 4.0
        self.c_worker = 0.05  # mild quadratic worker penalty

        # Load Shedding Economics
        self.b_max = 100.0
        self.psi_shed = 0.3
        self.delta_shed = 2.0

        # Multi-Dimensional Stress
        self.b_mem = 0.5  # scaled down from 5.0

        # Temporal Friction Tracking
        self.scale_cooldown_queue = 0.0

    def reset(self) -> None:
        self.scale_cooldown_queue = 0.0

    def calculate_reward(self, state: dict, action: dict) -> float:
        reward = 0.0

        # --- 1. Asymmetric Topology (Heart vs. Pinky) ---
        # DB failure is catastrophic but bounded
        if 0 in state.get("failed_nodes", []):
            return -5.0  # bounded terminal penalty, NOT -1000

        db_cpu = float(state["cpu_loads"][0]) if state.get("cpu_loads") else 0.0
        # Softer exponential: peaks at ~-11 for cpu=1.0, clipped below
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

            budget_depletion_factor = (self.b_max / max(1.0, error_budget)) ** 2
            r_shed = (
                -self.psi_shed
                * shed_amount
                * (1.0 + self.delta_shed * budget_depletion_factor)
            )
            reward += float(r_shed)

        # --- 3. Multi-Dimensional Stress (CPU Slope vs Mem Cliff) ---
        mem_usages = state.get("mem_utilizations", [0.4] * 8)
        r_mem = 0.0
        for mem in mem_usages:
            mem = float(mem)
            if mem >= 0.98:
                r_mem -= 2.0  # OOM bounded penalty per node, NOT -500
            elif mem > 0.5:
                # Smooth exponential with cap to prevent overflow
                exponent = min(10.0, 1.0 / max(0.02, 1.0 - mem))
                r_mem -= min(1.5, self.b_mem * np.exp(exponent - 2.0))
        reward += float(r_mem)

        # --- 4. Temporal Friction (Cold Start Integral) ---
        if action.get("action_type") == "scale_up":
            self.scale_cooldown_queue += 3 * sum(state.get("queue_lengths", [10]))

        if self.scale_cooldown_queue > 0:
            # Capped contribution
            reward -= min(1.0, 0.01 * self.scale_cooldown_queue)
            self.scale_cooldown_queue = max(
                0.0,
                self.scale_cooldown_queue - sum(state.get("queue_lengths", [10])),
            )

        # --- 5. Tail Latency (p99) ---
        p99 = float(state.get("p99_latency", float(state.get("latency_ms", 0.0)) * 1.5))
        if p99 > 50.0:
            # Quadratic penalty capped at -3.0
            excess = (p99 - 50.0) / 200.0
            reward -= min(3.0, 1.5 * (excess**2))

        # --- 6. Uptime bonus (dense positive signal) ---
        total_nodes = max(len(state.get("cpu_loads", [])), 1)
        failed_count = len(state.get("failed_nodes", []))
        uptime_ratio = (total_nodes - failed_count) / total_nodes
        reward += uptime_ratio * 1.0  # +1.0 for 100% uptime

        # --- 7. Action efficiency (anti-spam) ---
        action_type = str(action.get("action_type", "no_op"))
        if action_type != "no_op":
            reward -= 0.05  # slight tax on taking action
            if action_type == "throttle":
                reward -= 0.10  # extra tax on throttling

        return float(np.clip(reward, -5.0, 5.0))


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

    Returns a bounded scalar in [-5.0, +5.0].  Catastrophic states (DB dead,
    total collapse) return -5.0 — large enough to dominate the advantage but
    small enough to preserve gradient health.
    """
    state = build_production_state(sim)

    # Bounded terminal penalty for catastrophic states
    if is_dead or 0 in state.get("failed_nodes", []):
        return -5.0

    # Bounded terminal penalty for near-total collapse
    total = max(len(state.get("cpu_loads", [])), 1)
    failed_ratio = len(state.get("failed_nodes", [])) / total
    if failed_ratio >= 0.8:
        return -4.0

    engine = _get_sre_engine(sim)
    action = build_production_action(sim)
    return float(engine.calculate_reward(state, action))
