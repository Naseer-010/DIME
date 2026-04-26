"""
Distributed Infrastructure Simulation Engine.

Core simulation logic: weighted node graph, load redistribution,
failure probability model, cascading failure triggers, composable
rubric reward system, curriculum API, partial observability,
anti-hacking budgets/cooldowns, and real-world action schemas.
"""

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from server.models import InfraAction, InfraObservation, InfraState
from server.rubrics import compute_composite_reward, calculate_step_reward
from server.command_parser import (
    CommandParseError,
    has_reasoning_json_format,
    parse_command,
)

# Import tasks lazily to avoid circular imports
_TASKS = None


def _get_tasks():
    global _TASKS
    if _TASKS is None:
        from server.tasks import TASKS

        _TASKS = TASKS
    return _TASKS


# ---------------------------------------------------------------------------
# Node & Graph Data Structures
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """One compute node in the distributed system."""

    cpu_util: float = 0.3
    queue_length: int = 0
    capacity: int = 15
    is_failed: bool = False
    memory_util: float = 0.2
    high_cpu_streak: int = 0  # consecutive steps above 90% CPU
    restart_countdown: int = 0  # >0 means the node is restarting
    is_temporary: bool = False  # True for scale-up nodes
    ttl: int = 0  # remaining lifetime for temp nodes
    role: str = "app_server"  # "database" or "app_server"
    booting_steps: int = 0  # >0 means cold start, reduced processing speed


@dataclass
class SimulationState:
    """Full internal simulation state."""

    nodes: List[Node] = field(default_factory=list)
    adjacency: Dict[int, List[int]] = field(default_factory=dict)
    step_count: int = 0
    base_request_rate: float = 100.0
    current_request_rate: float = 100.0
    throttle_rate: float = 1.0  # 1.0 = accept all
    latency_ms: float = 20.0
    actions_taken: int = 0  # non-no_op actions
    cascade_bonus_awarded: bool = False
    task_id: str = ""
    max_steps: int = 30
    episode_id: str = ""
    # history for grading
    uptime_history: List[float] = field(default_factory=list)
    latency_history: List[float] = field(default_factory=list)
    restart_count: int = 0
    cascade_occurred: bool = False

    # --- Composable rubric support ---
    last_action_valid: bool = True
    last_response_format_valid: bool = True
    last_action_type: str = "no_op"
    previous_nodes: List[Node] = field(default_factory=list)
    previous_active_nodes: Optional[int] = None

    # --- Curriculum ---
    curriculum_level: int = 1

    # --- Anti-hacking sandbox ---
    cloud_budget: int = 10  # scale_up credits
    error_budget: float = 100.0  # throttle token bucket
    action_cooldowns: Dict[str, Dict[int, int]] = field(default_factory=dict)
    # e.g. {"restart_node": {3: 4}}  → node 3 has 4 steps of cooldown left
    action_errors: List[str] = field(default_factory=list)

    # --- Partial observability ---
    telemetry_dropout_nodes: List[int] = field(default_factory=list)

    # --- Trace replay ---
    trace_replay: Any = None  # Optional[TraceReplay]
    last_trace_p99_latency: float = 0.0
    last_trace_node_0_io: float = 0.0
    scenario: str = ""  # task-specific chaos scenario overlay
    _black_swan_applied: bool = False

    # --- Throughput tracking (anti-exploit) ---
    total_requests_received: int = 0
    total_requests_served: int = 0

    # --- Advanced RL State Tracking ---
    prev_node_loads: List[float] = field(default_factory=list)
    prev_active_nodes: int = 8
    prev_potential: float = 0.0


# ---------------------------------------------------------------------------
# Default graph topology: 8 nodes in a mesh-like structure
# ---------------------------------------------------------------------------


def _build_default_graph(n: int = 8) -> Tuple[List[Node], Dict[int, List[int]]]:
    """Create a default graph with node roles: node 0 = Database, rest = App Servers."""
    nodes = []
    for i in range(n):
        if i == 0:
            # Database node: higher capacity, single point of failure
            nodes.append(
                Node(
                    cpu_util=0.20 + random.uniform(-0.03, 0.03),
                    capacity=25,
                    role="database",
                )
            )
        else:
            nodes.append(
                Node(
                    cpu_util=0.25 + random.uniform(-0.05, 0.05),
                    capacity=15,
                    role="app_server",
                )
            )

    # Build connected graph: ring + cross-links, DB connected to all app servers
    adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}
    # DB (node 0) connects to every app server
    for i in range(1, n):
        adjacency[0].append(i)
        adjacency[i].append(0)
    # App servers: ring + skip connections among themselves
    for i in range(1, n):
        right = 1 + (i % (n - 1))  # wrap within app server range
        if right not in adjacency[i]:
            adjacency[i].append(right)
            adjacency[right].append(i)
        skip = 1 + ((i + 1) % (n - 1))
        if skip not in adjacency[i] and skip != right:
            adjacency[i].append(skip)
            adjacency[skip].append(i)

    return nodes, adjacency


# ---------------------------------------------------------------------------
# Target latency for reward normalisation
# ---------------------------------------------------------------------------
TARGET_LATENCY_MS = 50.0
HIGH_CPU_THRESHOLD = 0.90
OVERLOAD_THRESHOLD = 0.85
CASCADE_AWARENESS_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------


class DistributedInfraEnvironment(Environment):
    """
    Distributed infrastructure management environment.

    Models a weighted graph of compute nodes with stochastic request arrivals,
    node failures, load redistribution, and cascading failure dynamics.
    """

    def __init__(self):
        super().__init__()
        self._sim = SimulationState()
        self._state = InfraState(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random(42)

    # ----- helpers -----

    @property
    def sim(self) -> SimulationState:
        return self._sim

    @property
    def num_permanent_nodes(self) -> int:
        return sum(1 for n in self._sim.nodes if not n.is_temporary)

    # ----- Environment interface -----

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> InfraObservation:
        """Reset the environment, optionally with a specific task."""
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        task_id = kwargs.get("task", kwargs.get("task_id", "traffic_spike"))
        curriculum_level = int(kwargs.get("curriculum_level", 0))

        # Auto-detect curriculum level from task_id if not explicitly given
        if curriculum_level == 0:
            _level_map = {
                "level_1_read_logs": 1,
                "level_2_single_fix": 2,
                "traffic_spike": 2,
                "node_failure": 2,
                "level_3_stochastic": 3,
                "cascading_failure": 3,
                "level_4_expert": 4,
                "flash_crowd": 4,
            }
            curriculum_level = _level_map.get(task_id, 2)

        nodes, adjacency = _build_default_graph(8)
        self._sim = SimulationState(
            nodes=nodes,
            adjacency=adjacency,
            step_count=0,
            base_request_rate=100.0,
            current_request_rate=100.0,
            throttle_rate=1.0,
            latency_ms=20.0,
            actions_taken=0,
            cascade_bonus_awarded=False,
            task_id=task_id,
            max_steps=30,
            episode_id=episode_id or str(uuid4()),
            curriculum_level=curriculum_level,
            cloud_budget=max(5, 15 - curriculum_level * 2),  # harder = tighter budget
            error_budget=100.0,
        )

        # Apply task-specific setup
        tasks = _get_tasks()
        if task_id in tasks:
            tasks[task_id]["setup"](self, self._rng)

        eid = episode_id or self._sim.episode_id
        self._state = InfraState(
            episode_id=eid,
            step_count=0,
            task_id=task_id,
            task_score=0.01,
        )

        return self._make_observation()

    def step(
        self,
        action: InfraAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> InfraObservation:
        """Execute one time step in the simulation."""
        sim = self._sim

        # 0. Reset per-step error list
        sim.action_errors = []
        sim.previous_nodes = deepcopy(sim.nodes)
        sim.previous_active_nodes = sum(1 for n in sim.nodes if not n.is_failed)

        # 0.5 Error budget regen (token bucket economics)
        sim.error_budget = min(100.0, sim.error_budget + 2.0)

        # 1. Process agent action (handles raw_command, budgets, cooldowns)
        self._apply_action(action)

        # 2. Simulate request arrivals
        self._simulate_requests()

        # 2.5 Apply task-specific chaos overlays (deceptive signals, correlated failures)
        self._apply_scenario_overlays()

        # 3. Update node states (restarts, TTLs)
        self._tick_node_timers()

        # 4. Distribute load across nodes
        self._distribute_load()

        # 5. Update latency model
        self._update_latency()

        # 6. Check failure conditions → cascade
        self._check_failures()

        # 7. Advance step
        sim.step_count += 1
        self._state.step_count = sim.step_count

        # 8. Record history for grading
        alive = sum(1 for n in sim.nodes if not n.is_failed)
        total = len(sim.nodes)
        uptime_ratio = alive / total if total > 0 else 0.0
        sim.uptime_history.append(uptime_ratio)
        sim.latency_history.append(sim.latency_ms)

        # 9. Compute reward
        reward = self._compute_reward()

        # 10. Check termination
        tasks = _get_tasks()
        done = sim.step_count >= sim.max_steps
        task_score = 0.01
        if sim.task_id in tasks:
            task_info = tasks[sim.task_id]
            if task_info["is_done"](self):
                done = True
            task_score = task_info["grade"](self)

        self._state.task_score = task_score

        obs = self._make_observation()
        obs.reward = reward
        obs.done = done
        obs.task_score = task_score

        sim.prev_node_loads = [n.cpu_util for n in sim.nodes]
        sim.prev_active_nodes = sum(1 for n in sim.nodes if not n.is_failed)

        return obs

    @property
    def state(self) -> InfraState:
        return self._state

    # ----- Action handlers -----

    def _apply_action(self, action: InfraAction) -> None:
        sim = self._sim

        # --- Raw command parsing (real-world kubectl/AWS CLI) ---
        if action.raw_command:
            sim.last_response_format_valid = False
            try:
                raw_has_cot_format = has_reasoning_json_format(action.raw_command)
                action = parse_command(action.raw_command)
                sim.last_action_valid = True
                sim.last_response_format_valid = raw_has_cot_format
            except CommandParseError as exc:
                sim.action_errors.append(f"ParseError: {exc}")
                sim.last_action_valid = False
                sim.last_response_format_valid = False
                sim.last_action_type = "parse_error"
                return
        else:
            sim.last_action_valid = True
            sim.last_response_format_valid = True

        sim.last_action_type = action.action_type

        if action.action_type == "no_op":
            return

        # --- query_logs: partial-observability investigation action ---
        if action.action_type == "query_logs":
            idx = action.target
            if idx is not None and idx in sim.telemetry_dropout_nodes:
                sim.telemetry_dropout_nodes.remove(idx)
            # Does NOT count as a management action for efficiency scoring
            return

        sim.actions_taken += 1

        if action.action_type == "restart_node":
            idx = action.target
            if idx is not None and 0 <= idx < len(sim.nodes):
                # --- Cooldown check ---
                restart_cds = sim.action_cooldowns.get("restart_node", {})
                if restart_cds.get(idx, 0) > 0:
                    sim.action_errors.append(
                        f"CooldownActive: restart_node on node {idx} "
                        f"has {restart_cds[idx]} steps remaining."
                    )
                    return

                node = sim.nodes[idx]
                if node.is_failed and node.restart_countdown == 0:
                    node.restart_countdown = 2  # 2-step delay
                    sim.restart_count += 1
                    # Set 5-step cooldown
                    if "restart_node" not in sim.action_cooldowns:
                        sim.action_cooldowns["restart_node"] = {}
                    sim.action_cooldowns["restart_node"][idx] = 5

        elif action.action_type == "reroute_traffic":
            src = action.from_node
            dst = action.to_node
            if (
                src is not None
                and dst is not None
                and 0 <= src < len(sim.nodes)
                and 0 <= dst < len(sim.nodes)
                and not sim.nodes[src].is_failed
                and not sim.nodes[dst].is_failed
            ):
                # Shift 30% of source's load to destination
                transfer = sim.nodes[src].cpu_util * 0.30
                sim.nodes[src].cpu_util = max(0.0, sim.nodes[src].cpu_util - transfer)
                sim.nodes[dst].cpu_util = min(1.0, sim.nodes[dst].cpu_util + transfer)
                # Also move some queue items
                q_transfer = max(1, sim.nodes[src].queue_length // 3)
                sim.nodes[src].queue_length = max(
                    0, sim.nodes[src].queue_length - q_transfer
                )
                sim.nodes[dst].queue_length += q_transfer

                # Check cascade prevention bonus
                if (
                    not sim.cascade_bonus_awarded
                    and sim.nodes[dst].cpu_util < CASCADE_AWARENESS_THRESHOLD
                ):
                    for neighbor_idx in sim.adjacency.get(src, []):
                        if (
                            not sim.nodes[neighbor_idx].is_failed
                            and sim.nodes[neighbor_idx].cpu_util
                            > CASCADE_AWARENESS_THRESHOLD
                        ):
                            sim.cascade_bonus_awarded = True
                            break

        elif action.action_type == "scale_up":
            # --- Budget check ---
            if sim.cloud_budget <= 0:
                sim.action_errors.append(
                    "InsufficientFunds: cloud budget exhausted, cannot scale up."
                )
                return
            sim.cloud_budget -= 1

            # Add temporary capacity node with cold start
            new_idx = len(sim.nodes)
            new_node = Node(
                cpu_util=0.1,
                queue_length=0,
                is_temporary=True,
                ttl=10,
                role="app_server",
                booting_steps=3,  # cold start: 3 steps at 10% speed
            )
            sim.nodes.append(new_node)
            # Connect to a few existing nodes
            sim.adjacency[new_idx] = []
            connect_to = self._rng.sample(range(new_idx), min(3, new_idx))
            for c in connect_to:
                sim.adjacency[new_idx].append(c)
                sim.adjacency[c].append(new_idx)

        elif action.action_type == "throttle":
            # --- Action masking: throttling burns finite error budget ---
            if sim.error_budget <= 0:
                sim.action_errors.append("ACTION FAILED: Error budget exhausted")
                sim.last_action_type = "no_op"
                return
            if action.rate is not None:
                sim.throttle_rate = action.rate
                # Burn budget proportional to traffic dropped (more drop = more budget burn)
                burn = max(0.0, min(1.0, 1.0 - float(action.rate))) * 10.0
                sim.error_budget = max(0.0, sim.error_budget - burn)

    # ----- Simulation dynamics -----

    def _simulate_requests(self) -> None:
        """Simulate incoming request arrivals (trace replay or Gaussian)."""
        sim = self._sim
        scenario = (sim.scenario or sim.task_id or "").strip()

        # --- Trace replay: override request rate from real data ---
        if sim.trace_replay is not None:
            trace_step = sim.trace_replay.get_step(sim.step_count)
            sim.current_request_rate = trace_step.request_rate
            sim.last_trace_p99_latency = float(
                getattr(trace_step, "p99_latency", 0.0) or 0.0
            )
            sim.last_trace_node_0_io = float(
                getattr(trace_step, "node_0_io", 0.0) or 0.0
            )

            # Scenario-specific caps so "named incidents" don't instantly devolve into pure overload.
            if scenario == "memory_leak_slow_burn":
                sim.current_request_rate = min(sim.current_request_rate, 220.0)
            elif scenario == "black_swan_az_failure":
                sim.current_request_rate = min(sim.current_request_rate, 260.0)
            elif scenario in {
                "zombie_node",
                "connection_pool_deadlock",
                "hot_shard_skew",
            }:
                sim.current_request_rate = min(sim.current_request_rate, 280.0)
            # Inject per-node CPU baselines from trace
            for node_idx, cpu_val in trace_step.node_cpu.items():
                if node_idx < len(sim.nodes) and not sim.nodes[node_idx].is_failed:
                    # Blend trace CPU with simulation dynamics (60% trace, 40% sim)
                    sim.nodes[node_idx].cpu_util = (
                        0.6 * cpu_val + 0.4 * sim.nodes[node_idx].cpu_util
                    )
            # Inject per-node memory baselines from trace (if present)
            for node_idx, mem_val in getattr(trace_step, "node_mem", {}).items():
                if node_idx < len(sim.nodes) and not sim.nodes[node_idx].is_failed:
                    if scenario == "memory_leak_slow_burn" and node_idx == 5:
                        continue
                    sim.nodes[node_idx].memory_util = (
                        0.7 * float(mem_val) + 0.3 * sim.nodes[node_idx].memory_util
                    )
            # Add trace latency injection
            sim.latency_ms += trace_step.latency_injection * 0.3

        # Variable-rate with burst potential
        effective_rate = sim.current_request_rate * sim.throttle_rate

        # Arrival count for this step
        arrival_count = self._rng.gauss(effective_rate, effective_rate * 0.15)
        arrival_count = max(0, int(arrival_count))
        sim.total_requests_received += arrival_count

        # Distribute arrivals across operational APP SERVER nodes only
        # (DB receives back-pressure from app servers, not direct traffic)
        operational = [
            i
            for i, n in enumerate(sim.nodes)
            if not n.is_failed and n.restart_countdown == 0 and n.role == "app_server"
        ]
        if not operational:
            return

        per_node = arrival_count / len(operational)
        for idx in operational:
            node = sim.nodes[idx]
            added = int(per_node + self._rng.uniform(-2, 2))
            added = max(0, added)
            node.queue_length += added

    def _tick_node_timers(self) -> None:
        """Update restart countdowns, temporary node TTLs, and action cooldowns."""
        sim = self._sim
        expired_temps = []

        for i, node in enumerate(sim.nodes):
            # Handle restart countdown
            if node.restart_countdown > 0:
                node.restart_countdown -= 1
                if node.restart_countdown == 0 and node.is_failed:
                    node.is_failed = False
                    node.cpu_util = 0.15
                    node.queue_length = 0
                    node.high_cpu_streak = 0
                    node.memory_util = 0.1

            # Handle temp node TTL
            if node.is_temporary:
                node.ttl -= 1
                if node.ttl <= 0:
                    expired_temps.append(i)

        # --- Tick action cooldowns ---
        for action_type in list(sim.action_cooldowns.keys()):
            cds = sim.action_cooldowns[action_type]
            expired_keys = []
            for node_idx in cds:
                cds[node_idx] -= 1
                if cds[node_idx] <= 0:
                    expired_keys.append(node_idx)
            for k in expired_keys:
                del cds[k]
            if not cds:
                del sim.action_cooldowns[action_type]

        # Remove expired temporary nodes (in reverse to preserve indices)
        for idx in reversed(expired_temps):
            # Redistribute load before removing
            self._redistribute_from_node(idx)
            # Remove connections
            for neighbor in sim.adjacency.get(idx, []):
                if idx in sim.adjacency.get(neighbor, []):
                    sim.adjacency[neighbor].remove(idx)
            del sim.adjacency[idx]
            sim.nodes.pop(idx)
            # Fix adjacency indices
            new_adj: Dict[int, List[int]] = {}
            for k, v in sim.adjacency.items():
                new_k = k if k < idx else k - 1
                new_v = [(x if x < idx else x - 1) for x in v if x != idx]
                new_adj[new_k] = new_v
            sim.adjacency = new_adj

    def _distribute_load(self) -> None:
        """Process queued requests → affect CPU utilization (with cold start + DB dependency)."""
        sim = self._sim
        scenario = (sim.scenario or sim.task_id or "").strip()

        # Find the DB node for dependency calculations
        db_node = None
        for i, n in enumerate(sim.nodes):
            if n.role == "database" and not n.is_failed:
                db_node = n
                break

        db_available = db_node is not None and db_node.restart_countdown == 0

        for i, node in enumerate(sim.nodes):
            if node.is_failed or node.restart_countdown > 0:
                continue

            # --- Cold start: booting nodes process at 10% speed ---
            effective_capacity = node.capacity
            if node.booting_steps > 0:
                effective_capacity = max(1, int(node.capacity * 0.10))
                node.booting_steps -= 1

            # --- DB dependency: app servers can't process if DB is down ---
            if node.role == "app_server" and not db_available:
                # App servers can't process requests without the DB
                effective_capacity = 0

            # Process some requests from queue
            processed = min(node.queue_length, effective_capacity)
            node.queue_length = max(0, node.queue_length - processed)
            sim.total_requests_served += processed

            # --- DB back-pressure: each app server request costs DB CPU ---
            if (
                node.role == "app_server"
                and db_node is not None
                and db_available
                and processed > 0
                and scenario != "memory_leak_slow_burn"
            ):
                db_load = processed * 0.006  # each request costs DB ~0.6% CPU
                db_node.cpu_util = min(1.0, db_node.cpu_util + db_load)
                db_node.queue_length += max(1, processed // 5)

            # CPU effect: each processed request adds load, with natural decay
            cpu_from_queue = node.queue_length * 0.008
            cpu_from_processing = processed * 0.012
            if scenario == "memory_leak_slow_burn" and i == 5:
                # Make the leak deceptive: CPU stays "fine" while memory creeps up.
                cpu_from_queue = 0.0
                cpu_from_processing = 0.0
            natural_decay = 0.05

            node.cpu_util = max(
                0.05,
                min(
                    1.0,
                    node.cpu_util
                    + cpu_from_queue
                    + cpu_from_processing
                    - natural_decay
                    + self._rng.uniform(-0.02, 0.02),
                ),
            )

            # Memory correlates with CPU but with lag
            node.memory_util = max(
                0.05,
                min(1.0, node.memory_util * 0.9 + node.cpu_util * 0.15),
            )

    def _update_latency(self) -> None:
        """Compute rolling average latency based on queue & congestion."""
        sim = self._sim
        if not sim.nodes:
            return

        operational = [n for n in sim.nodes if not n.is_failed]
        if not operational:
            sim.latency_ms = 500.0  # Very high latency when all nodes down
            return

        avg_queue = sum(n.queue_length for n in operational) / len(operational)
        avg_cpu = sum(n.cpu_util for n in operational) / len(operational)

        # Latency model: base + queue component + CPU-pressure component
        base_latency = 10.0
        queue_latency = avg_queue * 1.5
        cpu_latency = (avg_cpu**2) * 80.0  # non-linear increase under load

        new_latency = base_latency + queue_latency + cpu_latency
        # Exponential moving average
        sim.latency_ms = sim.latency_ms * 0.3 + new_latency * 0.7

    def _check_failures(self) -> None:
        """Check if nodes fail due to sustained high CPU, trigger cascades.

        DB failure is catastrophic: all app servers lose processing ability.
        """
        sim = self._sim
        newly_failed: List[int] = []

        for i, node in enumerate(sim.nodes):
            if node.is_failed or node.restart_countdown > 0:
                continue

            # --- OOM cliff: instant failure above 0.98 memory ---
            if node.memory_util >= 0.98:
                node.is_failed = True
                node.high_cpu_streak = 0
                newly_failed.append(i)
                sim.action_errors.append(f"OOMKilled: Node-{i} memory exceeded 98%.")
                continue

            # --- DB SPOF fatality: hard crash at 100% CPU ---
            if node.role == "database" and node.cpu_util >= 1.0:
                node.is_failed = True
                node.high_cpu_streak = 0
                newly_failed.append(i)
                sim.action_errors.append(
                    "CRITICAL: Database Node-0 crashed due to 100% CPU lockup."
                )
                continue

            if node.cpu_util > HIGH_CPU_THRESHOLD:
                node.high_cpu_streak += 1
            else:
                node.high_cpu_streak = max(0, node.high_cpu_streak - 1)

            # Fail after 3 consecutive steps above 90%
            if node.high_cpu_streak >= 3:
                node.is_failed = True
                node.high_cpu_streak = 0
                newly_failed.append(i)

            # Probabilistic failure under load
            if node.cpu_util > 0.85 and self._rng.random() < 0.03:
                node.is_failed = True
                newly_failed.append(i)

        # Cascade: redistribute load from failed nodes to neighbors
        if newly_failed:
            sim.cascade_occurred = True
            for idx in newly_failed:
                # Zero out metrics immediately to prevent zombie state in
                # observations.  The load captured here is forwarded to
                # survivors inside _redistribute_from_node.
                node = sim.nodes[idx]
                node.cpu_util = 0.0
                node.queue_length = 0
                node.memory_util = 0.0

                self._redistribute_from_node(idx)
                # If DB fails, log a critical dependency event
                if node.role == "database":
                    sim.action_errors.append(
                        "CRITICAL: Database node failed — all app server "
                        "processing halted until DB is restarted."
                    )

    def _redistribute_from_node(self, idx: int) -> None:
        """Redistribute a failed/removed node's load to its neighbors.

        The caller is responsible for zeroing the source node's metrics
        *before* calling this method (see ``_check_failures``).  For the
        temp-node-expiry path in ``_tick_node_timers``, metrics are
        captured before this call so the load can still be forwarded.
        """
        sim = self._sim
        node = sim.nodes[idx]

        # Capture whatever residual load remains (may already be zero if
        # the caller cleared it, which is fine — we still attempt to
        # distribute for the temp-node-expiry path).
        load_to_move = node.cpu_util
        queue_to_move = node.queue_length

        # Zero out the dead node immediately (prevents zombie state even
        # if called from a path that didn't pre-clear).
        node.cpu_util = 0.0
        node.queue_length = 0
        node.memory_util = 0.0

        # Find healthy neighbors
        neighbors = [
            n
            for n in sim.adjacency.get(idx, [])
            if n < len(sim.nodes) and not sim.nodes[n].is_failed
        ]

        # If no neighbors survive, the load is simply dropped
        # (cascading failure complete).
        if not neighbors:
            return

        load_share = load_to_move / len(neighbors)
        queue_share = queue_to_move // max(1, len(neighbors))

        for neighbor_idx in neighbors:
            neighbor = sim.nodes[neighbor_idx]
            neighbor.cpu_util = min(1.0, neighbor.cpu_util + load_share)
            neighbor.queue_length += queue_share

    # ----- Reward computation (composable rubrics) -----

    def _compute_reward(self) -> float:
        """
        Dense step-level reward using independent, composable verifiers.

        Each verifier scores one aspect:
          - FormatVerifier:              +1.0 for valid response format
          - StabilityVerifier:           +0.4 for 100% uptime
          - SafeSliceLatencyVerifier:    smooth bounded latency penalty
          - CascadePBRSVerifier:         potential-based stress delta
          - EconomicEfficiencyVerifier:  linear cost + L1 churn penalty
          - ThroughputVerifier:          zero-service exploit penalty

        The breakdown is stored in ``_last_rubric_breakdown`` for
        inclusion in the observation metadata.
        """
        sim = self._sim
        if not sim.nodes:
            self._last_rubric_breakdown = {}
            return 0.0

        # Keep OpenEnv composable rubric breakdown for evaluation/analysis
        _, breakdown = compute_composite_reward(sim)
        self._last_rubric_breakdown = breakdown

        # Use ProductionSREReward for the actual RL training signal
        return float(calculate_step_reward(sim, is_dead=False))

    # ----- Observation builder -----

    def _make_observation(self) -> InfraObservation:
        """Build the current observation with partial observability + Prometheus metrics."""
        sim = self._sim
        tasks = _get_tasks()

        task_hint = ""
        if sim.task_id in tasks:
            task_hint = tasks[sim.task_id].get("hint", "")

        # --- Partial observability: 5% telemetry dropout per node ---
        cpu_loads = []
        mem_utils = []
        queue_lengths = []
        telemetry_status: Dict[int, str] = {}

        # Decide new dropouts for this step (keep previously dropped if not queried)
        new_dropouts: List[int] = []
        for i, node in enumerate(sim.nodes):
            if i in sim.telemetry_dropout_nodes:
                new_dropouts.append(i)
            elif self._rng.random() < 0.05:
                new_dropouts.append(i)
        sim.telemetry_dropout_nodes = new_dropouts

        # For targeted incident tasks, keep key nodes observable to prevent
        # learning collapse due to missing critical features.
        scenario = (sim.scenario or sim.task_id or "").strip()
        if scenario == "memory_leak_slow_burn":
            sim.telemetry_dropout_nodes = [
                i for i in sim.telemetry_dropout_nodes if i not in (0, 5)
            ]

        # --- Build Prometheus-style metrics ---
        prometheus_metrics: List[Dict[str, Any]] = []
        ts = sim.step_count * 30  # simulate 30s intervals

        for i, node in enumerate(sim.nodes):
            node_name = f"worker-{i}"
            is_dropped = i in sim.telemetry_dropout_nodes

            if is_dropped:
                cpu_loads.append(-1.0)
                mem_utils.append(-1.0)
                queue_lengths.append(-1)
                telemetry_status[i] = "timeout"
                prometheus_metrics.append(
                    {
                        "metric": "node_scrape_error",
                        "labels": {"node": node_name, "role": node.role},
                        "value": 1,
                        "timestamp": ts,
                    }
                )
            else:
                cpu_loads.append(round(node.cpu_util, 3))
                mem_utils.append(round(node.memory_util, 3))
                queue_lengths.append(node.queue_length)
                telemetry_status[i] = "ok"
                prometheus_metrics.extend(
                    [
                        {
                            "metric": "node_cpu_utilization",
                            "labels": {"node": node_name, "role": node.role},
                            "value": round(node.cpu_util, 4),
                            "timestamp": ts,
                        },
                        {
                            "metric": "node_memory_utilization",
                            "labels": {"node": node_name, "role": node.role},
                            "value": round(node.memory_util, 4),
                            "timestamp": ts,
                        },
                        {
                            "metric": "node_queue_depth",
                            "labels": {"node": node_name, "role": node.role},
                            "value": node.queue_length,
                            "timestamp": ts,
                        },
                    ]
                )
                if node.booting_steps > 0:
                    prometheus_metrics.append(
                        {
                            "metric": "node_boot_remaining_steps",
                            "labels": {"node": node_name},
                            "value": node.booting_steps,
                            "timestamp": ts,
                        }
                    )

        # Global metrics
        prometheus_metrics.extend(
            [
                {
                    "metric": "cluster_latency_ms",
                    "labels": {},
                    "value": round(sim.latency_ms, 2),
                    "timestamp": ts,
                },
                {
                    "metric": "cluster_p99_latency_ms",
                    "labels": {},
                    "value": round(sim.last_trace_p99_latency, 2),
                    "timestamp": ts,
                },
                {
                    "metric": "db_node_0_io_wait",
                    "labels": {},
                    "value": round(sim.last_trace_node_0_io, 4),
                    "timestamp": ts,
                },
                {
                    "metric": "cluster_request_rate",
                    "labels": {},
                    "value": round(sim.current_request_rate * sim.throttle_rate, 2),
                    "timestamp": ts,
                },
                {
                    "metric": "cluster_error_budget",
                    "labels": {},
                    "value": round(sim.error_budget, 2),
                    "timestamp": ts,
                },
                {
                    "metric": "cluster_cloud_budget",
                    "labels": {},
                    "value": sim.cloud_budget,
                    "timestamp": ts,
                },
            ]
        )

        raw_rr = float(sim.current_request_rate * sim.throttle_rate)
        raw_p99 = float(sim.last_trace_p99_latency)
        rr_norm = max(0.0, min(1.0, raw_rr / 5000.0))
        p99_norm = max(0.0, min(1.0, raw_p99 / 1000.0))

        return InfraObservation(
            cpu_loads=cpu_loads,
            mem_utilizations=mem_utils,
            queue_lengths=queue_lengths,
            failed_nodes=[i for i, n in enumerate(sim.nodes) if n.is_failed],
            latency_ms=round(sim.latency_ms, 2),
            request_rate=round(raw_rr, 2),
            io_wait=round(sim.last_trace_node_0_io, 4),
            p99_latency=round(sim.last_trace_p99_latency, 2),
            error_budget=round(sim.error_budget, 2),
            request_rate_norm=round(rr_norm, 6),
            p99_latency_norm=round(p99_norm, 6),
            step=sim.step_count,
            task_hint=task_hint,
            done=False,
            reward=0.0,
            telemetry_status=telemetry_status,
            action_errors=list(sim.action_errors),
            cloud_budget=sim.cloud_budget,
            prometheus_metrics=prometheus_metrics,
        )

    def _apply_scenario_overlays(self) -> None:
        """
        Task-specific chaos overlays to produce deceptive signals and correlated failures.

        These overlays are intentionally simple and deterministic so RL can learn them.
        They are applied on top of trace replay + base simulation dynamics.
        """
        sim = self._sim
        scenario = (sim.scenario or sim.task_id or "").strip()

        # --- Retry storm: if tail latency crosses threshold, traffic doubles next step ---
        if scenario in {"retry_storm", "thundering_herd"}:
            if sim.last_trace_p99_latency > 50.0 or sim.latency_ms > 50.0:
                sim.current_request_rate *= 2.0
                for n in sim.nodes:
                    if not n.is_failed and n.restart_countdown == 0:
                        n.cpu_util = min(1.0, n.cpu_util + 0.25)

        # --- Hot shard skew: one worker melts while others look idle ---
        if scenario == "hot_shard_skew":
            hot = 2
            for i, n in enumerate(sim.nodes):
                if n.role != "app_server" or n.is_failed or n.restart_countdown > 0:
                    continue
                if i == hot:
                    n.cpu_util = min(1.0, n.cpu_util + 0.40)
                else:
                    n.cpu_util = max(0.05, min(n.cpu_util, 0.20))

        # --- Zombie node / deadlock: low CPU but massive tail latency ---
        if scenario in {"zombie_node", "connection_pool_deadlock"}:
            zombie = 3
            if zombie < len(sim.nodes):
                n = sim.nodes[zombie]
                if (
                    not n.is_failed
                    and n.restart_countdown == 0
                    and n.role == "app_server"
                ):
                    n.cpu_util = min(n.cpu_util, 0.08)
                    sim.last_trace_p99_latency = max(sim.last_trace_p99_latency, 300.0)

        # --- Memory leak slow burn: node 5 memory climbs regardless of load ---
        if scenario == "memory_leak_slow_burn":
            leak = 5
            # Keep the rest of the system "apparently healthy" so the leak is the dominant signal.
            for i, node in enumerate(sim.nodes):
                if node.is_failed or node.restart_countdown > 0:
                    continue
                if node.role == "database":
                    node.cpu_util = min(node.cpu_util, 0.65)
                elif i != leak:
                    node.cpu_util = min(node.cpu_util, 0.55)
                    node.queue_length = min(node.queue_length, 25)

            if leak < len(sim.nodes):
                n = sim.nodes[leak]
                if not n.is_failed and n.restart_countdown == 0:
                    n.memory_util = min(1.0, n.memory_util + 0.02)
                    n.cpu_util = min(max(0.05, n.cpu_util), 0.40)

        # --- Split brain / IO bottleneck: DB io_wait spikes; scaling makes it worse ---
        if scenario == "split_brain_io_bottleneck":
            sim.last_trace_node_0_io = max(sim.last_trace_node_0_io, 0.85)
            if sim.last_action_type == "scale_up" and sim.last_trace_node_0_io > 0.80:
                # Connection pool exhaustion style shock
                sim.action_errors.append(
                    "CRITICAL: Scale-up during DB IO saturation caused connection pool exhaustion."
                )
                # Raise DB CPU sharply (lockup spiral)
                if sim.nodes and sim.nodes[0].role == "database":
                    sim.nodes[0].cpu_util = min(1.0, sim.nodes[0].cpu_util + 0.40)

        # --- Black swan: correlated AZ failure at step 3 ---
        if scenario == "black_swan_az_failure" and not sim._black_swan_applied:
            if sim.step_count >= 3:
                dead = [1, 2, 3, 4]
                for idx in dead:
                    if idx < len(sim.nodes):
                        sim.nodes[idx].is_failed = True
                sim.action_errors.append(
                    "CRITICAL: Availability Zone offline. Nodes 1-4 dead."
                )
                for idx in [5, 6, 7]:
                    if idx < len(sim.nodes) and not sim.nodes[idx].is_failed:
                        sim.nodes[idx].cpu_util = max(sim.nodes[idx].cpu_util, 0.95)
                # Give the DB a brief headroom bump to avoid immediate total collapse.
                if sim.nodes and sim.nodes[0].role == "database":
                    sim.nodes[0].cpu_util = min(sim.nodes[0].cpu_util, 0.75)
                sim._black_swan_applied = True
