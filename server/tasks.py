"""
Three graded tasks with increasing difficulty for the
Distributed Infrastructure Management Environment.

Each task provides:
    - setup(env, rng): configure initial node states and scenario parameters
    - grade(env): return float in [0.0, 1.0] with partial credit
    - is_done(env): termination condition check
    - hint: natural language task description for the agent
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random
    from server.environment import DistributedInfraEnvironment


# ============================================================================
# Task 1 — Easy: Traffic Spike Recovery
# ============================================================================

def _setup_traffic_spike(env: "DistributedInfraEnvironment", rng: "random.Random"):
    """System receives 3x normal request rate."""
    sim = env.sim
    sim.current_request_rate = sim.base_request_rate * 3.0
    sim.max_steps = 30
    # Start with moderate load
    for node in sim.nodes:
        node.cpu_util = 0.45 + rng.uniform(-0.05, 0.1)
        node.queue_length = rng.randint(5, 15)


def _grade_traffic_spike(env: "DistributedInfraEnvironment") -> float:
    """
    Score = latency below threshold (50%) + uptime (30%) + resource efficiency (20%).
    """
    sim = env.sim
    if not sim.latency_history:
        return 0.0

    # Latency component: fraction of steps where latency was below target
    target = 50.0  # ms
    below_target = sum(1 for lat in sim.latency_history if lat < target)
    latency_score = below_target / len(sim.latency_history)

    # Uptime component: average uptime ratio
    avg_uptime = sum(sim.uptime_history) / len(sim.uptime_history) if sim.uptime_history else 1.0

    # Efficiency: penalty for excessive actions
    max_reasonable = sim.max_steps * 0.5
    efficiency = max(0.0, 1.0 - sim.actions_taken / max(1, max_reasonable))

    score = 0.50 * latency_score + 0.30 * avg_uptime + 0.20 * efficiency
    return round(min(1.0, max(0.0, score)), 4)


def _is_done_traffic_spike(env: "DistributedInfraEnvironment") -> bool:
    return env.sim.step_count >= env.sim.max_steps


# ============================================================================
# Task 2 — Medium: Single Node Failure
# ============================================================================

def _setup_node_failure(env: "DistributedInfraEnvironment", rng: "random.Random"):
    """One node will fail at step 5. Agent must maintain 80%+ uptime."""
    sim = env.sim
    sim.max_steps = 40
    sim.current_request_rate = sim.base_request_rate * 1.5

    # Mark node 3 for pre-programmed failure
    sim.nodes[3].cpu_util = 0.60
    sim.nodes[3].queue_length = 20


def _grade_node_failure(env: "DistributedInfraEnvironment") -> float:
    """
    Score = MTTR (40%) + uptime during failure window (40%) - restart penalty (20%).
    """
    sim = env.sim

    if not sim.uptime_history:
        return 0.0

    # MTTR: how quickly system recovered from the failure
    failure_duration = 0
    in_failure = False
    for uptime in sim.uptime_history:
        if uptime < 1.0:
            in_failure = True
            failure_duration += 1
        elif in_failure:
            break

    max_failure_window = 10
    mttr_score = max(0.0, 1.0 - failure_duration / max_failure_window)

    # Uptime component: fraction of steps with >80% uptime
    above_80 = sum(1 for u in sim.uptime_history if u >= 0.80)
    uptime_score = above_80 / len(sim.uptime_history)

    # Restart penalty: more than 2 restarts is wasteful
    restart_penalty = max(0.0, 1.0 - max(0, sim.restart_count - 1) / 5)

    score = 0.40 * mttr_score + 0.40 * uptime_score + 0.20 * restart_penalty
    return round(min(1.0, max(0.0, score)), 4)


def _is_done_node_failure(env: "DistributedInfraEnvironment") -> bool:
    sim = env.sim
    # Inject failure at step 5
    if sim.step_count == 5 and 3 < len(sim.nodes) and not sim.nodes[3].is_failed:
        sim.nodes[3].is_failed = True
        sim.nodes[3].cpu_util = 0.0
        sim.nodes[3].queue_length = 0
        # Redistribute its load
        env._redistribute_from_node(3)

    return sim.step_count >= sim.max_steps


# ============================================================================
# Task 3 — Hard: Cascading Failure Prevention
# ============================================================================

def _setup_cascading_failure(env: "DistributedInfraEnvironment", rng: "random.Random"):
    """Two nodes near critical CPU. Agent must prevent cascade chain."""
    sim = env.sim
    sim.max_steps = 50
    sim.current_request_rate = sim.base_request_rate * 2.0

    # Put nodes 1 and 4 near critical
    sim.nodes[1].cpu_util = 0.88
    sim.nodes[1].queue_length = 30
    sim.nodes[1].high_cpu_streak = 2

    sim.nodes[4].cpu_util = 0.86
    sim.nodes[4].queue_length = 25
    sim.nodes[4].high_cpu_streak = 1

    # Higher base load across all nodes
    for i, node in enumerate(sim.nodes):
        if i not in (1, 4):
            node.cpu_util = 0.55 + rng.uniform(-0.05, 0.1)
            node.queue_length = rng.randint(8, 20)


def _grade_cascading_failure(env: "DistributedInfraEnvironment") -> float:
    """
    Score = cascade prevented (50%) + nodes below 85% CPU (30%)
            + action efficiency (20%).
    """
    sim = env.sim

    cascade_score = 1.0 if not sim.cascade_occurred else 0.3

    if sim.uptime_history:
        healthy_now = sum(
            1 for n in sim.nodes
            if not n.is_failed and n.cpu_util < 0.85
        )
        total_now = len(sim.nodes)
        cpu_score = healthy_now / total_now if total_now > 0 else 0.0
    else:
        cpu_score = 0.0

    max_reasonable = sim.max_steps * 0.4
    efficiency = max(0.0, 1.0 - sim.actions_taken / max(1, max_reasonable))

    score = 0.50 * cascade_score + 0.30 * cpu_score + 0.20 * efficiency
    return round(min(1.0, max(0.0, score)), 4)


def _is_done_cascading_failure(env: "DistributedInfraEnvironment") -> bool:
    sim = env.sim
    failed_count = sum(1 for n in sim.nodes if n.is_failed)
    if failed_count > len(sim.nodes) // 2:
        return True
    return sim.step_count >= sim.max_steps


# ============================================================================
# Task Registry
# ============================================================================

TASKS = {
    "traffic_spike": {
        "setup": _setup_traffic_spike,
        "grade": _grade_traffic_spike,
        "is_done": _is_done_traffic_spike,
        "hint": (
            "TRAFFIC SPIKE: The system is experiencing 3x normal request volume. "
            "Your goal is to keep latency below 50ms while maintaining full uptime. "
            "Consider rerouting traffic from overloaded nodes, scaling up capacity, "
            "or throttling incoming requests. Minimize unnecessary actions."
        ),
    },
    "node_failure": {
        "setup": _setup_node_failure,
        "grade": _grade_node_failure,
        "is_done": _is_done_node_failure,
        "hint": (
            "NODE FAILURE: A node failure will occur during this episode. "
            "You must detect the failure, restart the affected node, and maintain "
            "system uptime above 80%%. React quickly — Mean Time To Repair matters. "
            "Avoid unnecessary restarts of healthy nodes."
        ),
    },
    "cascading_failure": {
        "setup": _setup_cascading_failure,
        "grade": _grade_cascading_failure,
        "is_done": _is_done_cascading_failure,
        "hint": (
            "CASCADING FAILURE PREVENTION: Two nodes are near critical CPU load "
            "(>85%%). If they reach 90%% for 3 consecutive steps, they will fail "
            "and their load will cascade to neighbors, potentially triggering a "
            "chain reaction. ACT PROACTIVELY: reroute traffic away from hot nodes "
            "BEFORE they fail. Scaling up can help absorb excess load. "
            "Prevention is rewarded more than recovery."
        ),
    },
}
