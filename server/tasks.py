"""
Graded tasks with curriculum-based difficulty levels for the
Distributed Infrastructure Management Environment.

Each task provides:
    - setup(env, rng): configure initial node states and scenario parameters
    - grade(env): return float in (0.0, 1.0) with partial credit
    - is_done(env): termination condition check
    - hint: natural language task description for the agent

Curriculum Levels
-----------------
Level 1  Warm Start     — Identify the failing node from logs (high success rate)
Level 2  Single Fix     — One node fails, agent must restart it
Level 3  Stochastic     — Gaussian traffic spikes, multi-step interventions
Level 4  Expert         — Brutal cascading failures with tight budgets
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random
    from server.environment import DistributedInfraEnvironment


# ============================================================================
# Level 1 — Warm Start: Read Logs & Identify Failing Node
# ============================================================================


def _setup_level_1(env: "DistributedInfraEnvironment", rng: "random.Random"):
    """One node is pre-failed. Agent just needs to identify it via query_logs."""
    sim = env.sim
    sim.max_steps = 15
    sim.current_request_rate = sim.base_request_rate * 1.0  # normal traffic
    # Fail one random node
    fail_idx = rng.randint(0, len(sim.nodes) - 1)
    sim.nodes[fail_idx].is_failed = True
    sim.nodes[fail_idx].cpu_util = 0.0
    sim.nodes[fail_idx].queue_length = 0


def _grade_level_1(env: "DistributedInfraEnvironment") -> float:
    """
    Score = 0.7 * identified failing node (restarted it) + 0.3 * speed.
    """
    sim = env.sim
    # Did the agent restart the failed node?
    all_alive = all(not n.is_failed for n in sim.nodes)
    identification = 1.0 if all_alive else 0.2

    # Speed bonus: faster = better
    speed = max(0.0, 1.0 - sim.step_count / sim.max_steps)

    score = 0.70 * identification + 0.30 * speed
    return round(min(0.99, max(0.01, score)), 4)


def _is_done_level_1(env: "DistributedInfraEnvironment") -> bool:
    sim = env.sim
    # Done if agent fixed the node or time ran out
    all_alive = all(not n.is_failed and n.restart_countdown == 0 for n in sim.nodes)
    return all_alive or sim.step_count >= sim.max_steps


# ============================================================================
# Level 2 / Task 1 — Traffic Spike Recovery
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
        return 0.01

    # Latency component: fraction of steps where latency was below target
    target = 50.0  # ms
    below_target = sum(1 for lat in sim.latency_history if lat < target)
    latency_score = below_target / len(sim.latency_history)

    # Uptime component: average uptime ratio
    avg_uptime = (
        sum(sim.uptime_history) / len(sim.uptime_history) if sim.uptime_history else 1.0
    )

    # Efficiency: penalty for excessive actions
    max_reasonable = sim.max_steps * 0.5
    efficiency = max(0.0, 1.0 - sim.actions_taken / max(1, max_reasonable))

    score = 0.50 * latency_score + 0.30 * avg_uptime + 0.20 * efficiency
    return round(min(0.99, max(0.01, score)), 4)


def _is_done_traffic_spike(env: "DistributedInfraEnvironment") -> bool:
    return env.sim.step_count >= env.sim.max_steps


# ============================================================================
# Level 2 / Task 2 — Single Node Failure
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
        return 0.01

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
    return round(min(0.99, max(0.01, score)), 4)


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
# Level 2 — Alias: Single Fix (same as node_failure)
# ============================================================================

_setup_level_2 = _setup_node_failure
_grade_level_2 = _grade_node_failure
_is_done_level_2 = _is_done_node_failure


# ============================================================================
# Level 3 / Task 3 — Cascading Failure Prevention (Stochastic)
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
        healthy_now = sum(1 for n in sim.nodes if not n.is_failed and n.cpu_util < 0.85)
        total_now = len(sim.nodes)
        cpu_score = healthy_now / total_now if total_now > 0 else 0.0
    else:
        cpu_score = 0.0

    max_reasonable = sim.max_steps * 0.4
    efficiency = max(0.0, 1.0 - sim.actions_taken / max(1, max_reasonable))

    score = 0.50 * cascade_score + 0.30 * cpu_score + 0.20 * efficiency
    return round(min(0.99, max(0.01, score)), 4)


def _is_done_cascading_failure(env: "DistributedInfraEnvironment") -> bool:
    sim = env.sim
    failed_count = sum(1 for n in sim.nodes if n.is_failed)
    if failed_count > len(sim.nodes) // 2:
        return True
    return sim.step_count >= sim.max_steps


# ============================================================================
# Level 3 — Alias: Stochastic (enhanced version of cascading_failure)
# ============================================================================


def _setup_level_3(env: "DistributedInfraEnvironment", rng: "random.Random"):
    """Gaussian stochastic traffic spikes with noisy sensors."""
    _setup_cascading_failure(env, rng)
    sim = env.sim
    # Add Gaussian noise to request rate each step (handled in sim dynamics)
    sim.current_request_rate = sim.base_request_rate * (2.0 + rng.gauss(0, 0.5))
    sim.max_steps = 45


_grade_level_3 = _grade_cascading_failure
_is_done_level_3 = _is_done_cascading_failure


# ============================================================================
# Level 4 / Task 4 — Expert: Flash Crowd
# ============================================================================


def _setup_flash_crowd(env: "DistributedInfraEnvironment", rng: "random.Random"):
    """Massive 5x traffic spike. Agent must scale up AND throttle to survive."""
    sim = env.sim
    sim.current_request_rate = sim.base_request_rate * 5.0
    sim.max_steps = 40
    for node in sim.nodes:
        node.cpu_util = 0.60 + rng.uniform(-0.05, 0.1)
        node.queue_length = rng.randint(15, 30)


def _grade_flash_crowd(env: "DistributedInfraEnvironment") -> float:
    """
    Score = Survival Uptime (50%) + Latency control (50%).
    Cascade penalty applied if the system collapses.
    """
    sim = env.sim

    avg_uptime = (
        sum(sim.uptime_history) / len(sim.uptime_history) if sim.uptime_history else 0.0
    )

    # Latency target is more generous for a massive flash crowd (100ms)
    target = 100.0
    below_target = sum(1 for lat in sim.latency_history if lat < target)
    latency_score = (
        below_target / len(sim.latency_history) if sim.latency_history else 0.0
    )

    cascade_penalty = 0.4 if sim.cascade_occurred else 0.0

    score = 0.50 * avg_uptime + 0.50 * latency_score - cascade_penalty
    return round(min(0.99, max(0.01, score)), 4)


def _is_done_flash_crowd(env: "DistributedInfraEnvironment") -> bool:
    failed_count = sum(1 for n in env.sim.nodes if n.is_failed)
    # Terminate early if more than 60% of the cluster dies
    if failed_count > len(env.sim.nodes) * 0.6:
        return True
    return env.sim.step_count >= env.sim.max_steps


# ============================================================================
# Level 4 — Alias: Expert (flash crowd with tightest constraints)
# ============================================================================

_setup_level_4 = _setup_flash_crowd
_grade_level_4 = _grade_flash_crowd
_is_done_level_4 = _is_done_flash_crowd


# ============================================================================
# Level 5 — Alibaba Trace Replay (Real-World Production Traffic)
# ============================================================================


def _setup_alibaba_trace(env: "DistributedInfraEnvironment", rng: "random.Random"):
    """Load real Alibaba cluster trace and replay it step-by-step."""
    from server.trace_loader import load_default_trace

    sim = env.sim
    sim.max_steps = 60  # ~30 minutes of real time at 30s intervals
    sim.cloud_budget = 8  # tight budget

    trace = load_default_trace()
    if trace is not None:
        sim.trace_replay = trace
        # Start replay from a random offset to vary episodes
        offset = rng.randint(0, max(1, len(trace) - sim.max_steps))
        # We store offset in step_count adjustment — trace_loader wraps around
        sim.current_request_rate = trace.get_step(offset).request_rate
    else:
        # Fallback: synthetic 2x traffic if trace not generated
        sim.current_request_rate = sim.base_request_rate * 2.0

    # Pre-stress the cluster slightly
    for node in sim.nodes:
        if node.role == "app_server":
            node.cpu_util = 0.40 + rng.uniform(-0.05, 0.1)
            node.queue_length = rng.randint(3, 12)
        elif node.role == "database":
            node.cpu_util = 0.35 + rng.uniform(-0.03, 0.05)
            node.queue_length = rng.randint(2, 8)


def _grade_alibaba_trace(env: "DistributedInfraEnvironment") -> float:
    """
    Score = Uptime (35%) + Latency (30%) + Throughput (20%) + Efficiency (15%).
    """
    sim = env.sim

    # Uptime
    avg_uptime = (
        sum(sim.uptime_history) / len(sim.uptime_history) if sim.uptime_history else 0.0
    )

    # Latency: fraction of steps below 80ms (more generous for real traffic)
    target = 80.0
    below_target = sum(1 for lat in sim.latency_history if lat < target)
    latency_score = (
        below_target / len(sim.latency_history) if sim.latency_history else 0.0
    )

    # Throughput: did the agent actually serve requests?
    throughput_ratio = sim.total_requests_served / max(1, sim.total_requests_received)
    throughput_score = min(1.0, throughput_ratio / 0.6)  # 60% = full marks

    # Efficiency: budget conservation
    budget_used = 8 - sim.cloud_budget
    efficiency_score = max(0.0, 1.0 - budget_used / 8)

    score = (
        0.35 * avg_uptime
        + 0.30 * latency_score
        + 0.20 * throughput_score
        + 0.15 * efficiency_score
    )
    return round(min(0.99, max(0.01, score)), 4)


def _is_done_alibaba_trace(env: "DistributedInfraEnvironment") -> bool:
    sim = env.sim
    # Terminate early if >70% of cluster dies
    failed_count = sum(1 for n in sim.nodes if n.is_failed)
    if failed_count > len(sim.nodes) * 0.7:
        return True
    return sim.step_count >= sim.max_steps


# ============================================================================
# Task Registry
# ============================================================================

TASKS = {
    # --- Curriculum levels ---
    "level_1_read_logs": {
        "setup": _setup_level_1,
        "grade": _grade_level_1,
        "is_done": _is_done_level_1,
        "hint": (
            "WARM START (Level 1): One node in your cluster has silently failed. "
            "Use 'query_logs' to investigate nodes with telemetry dropouts and "
            "identify the failing node. Then restart it. "
            "This is a diagnostic exercise — focus on observation before action."
        ),
    },
    "level_2_single_fix": {
        "setup": _setup_level_2,
        "grade": _grade_level_2,
        "is_done": _is_done_level_2,
        "hint": (
            "SINGLE FIX (Level 2): A node failure will occur during this episode. "
            "Detect the failure, restart the affected node, and maintain uptime "
            "above 80%%. Minimise unnecessary restarts."
        ),
    },
    "level_3_stochastic": {
        "setup": _setup_level_3,
        "grade": _grade_level_3,
        "is_done": _is_done_level_3,
        "hint": (
            "STOCHASTIC SPIKES (Level 3): Traffic follows a noisy Gaussian pattern. "
            "Multiple nodes are near critical CPU. Proactively reroute traffic and "
            "scale up before cascading failures occur. Telemetry may be spotty — "
            "use query_logs to investigate timeouts."
        ),
    },
    "level_4_expert": {
        "setup": _setup_level_4,
        "grade": _grade_level_4,
        "is_done": _is_done_level_4,
        "hint": (
            "EXPERT MODE (Level 4): A brutal 5x flash crowd with tight cloud budget. "
            "You MUST aggressively scale up AND throttle to survive. Budget is limited — "
            "every scale_up costs 1 unit. If you exhaust your budget, you cannot add "
            "more capacity. Plan wisely."
        ),
    },
    # --- Original task aliases (backward-compatible) ---
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
    "flash_crowd": {
        "setup": _setup_flash_crowd,
        "grade": _grade_flash_crowd,
        "is_done": _is_done_flash_crowd,
        "hint": (
            "FLASH CROWD: The system is facing an unprecedented 5x traffic surge. "
            "Your objective is pure survival. You MUST aggressively use 'scale_up' "
            "to add capacity AND use 'throttle' to drop excess traffic. "
            "If you do not shed load, the cluster will collapse."
        ),
    },
    # --- Level 5: Alibaba Trace Replay ---
    "level_5_alibaba_trace": {
        "setup": _setup_alibaba_trace,
        "grade": _grade_alibaba_trace,
        "is_done": _is_done_alibaba_trace,
        "hint": (
            "ALIBABA TRACE REPLAY (Level 5): You are operating on REAL production "
            "traffic from Alibaba's microservices cluster (2021 trace data). "
            "Traffic has multimodal spikes, micro-bursts, and silent maintenance windows. "
            "Node 0 is the DATABASE (single point of failure). Nodes 1-7 are app servers. "
            "New nodes have a 3-step cold start. Budget is tight (8 credits). "
            "Read Prometheus metrics carefully — they follow production scrape format."
        ),
    },
}
