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
from server.rubrics import compute_composite_reward
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
    action_cooldowns: Dict[str, Dict[int, int]] = field(default_factory=dict)
    # e.g. {"restart_node": {3: 4}}  → node 3 has 4 steps of cooldown left
    action_errors: List[str] = field(default_factory=list)

    # --- Partial observability ---
    telemetry_dropout_nodes: List[int] = field(default_factory=list)

    # --- Trace replay ---
    trace_replay: Any = None  # Optional[TraceReplay]

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

        # 1. Process agent action (handles raw_command, budgets, cooldowns)
        self._apply_action(action)

        # 2. Simulate request arrivals
        self._simulate_requests()

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
            if action.rate is not None:
                sim.throttle_rate = action.rate

    # ----- Simulation dynamics -----

    def _simulate_requests(self) -> None:
        """Simulate incoming request arrivals (trace replay or Gaussian)."""
        sim = self._sim

        # --- Trace replay: override request rate from real data ---
        if sim.trace_replay is not None:
            trace_step = sim.trace_replay.get_step(sim.step_count)
            sim.current_request_rate = trace_step.request_rate
            # Inject per-node CPU baselines from trace
            for node_idx, cpu_val in trace_step.node_cpu.items():
                if node_idx < len(sim.nodes) and not sim.nodes[node_idx].is_failed:
                    # Blend trace CPU with simulation dynamics (60% trace, 40% sim)
                    sim.nodes[node_idx].cpu_util = (
                        0.6 * cpu_val + 0.4 * sim.nodes[node_idx].cpu_util
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

        # Find the DB node for dependency calculations
        db_node = None
        for i, n in enumerate(sim.nodes):
            if n.role == "database" and not n.is_failed:
                db_node = n
                db_idx = i
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
            ):
                db_load = processed * 0.006  # each request costs DB ~0.6% CPU
                db_node.cpu_util = min(1.0, db_node.cpu_util + db_load)
                db_node.queue_length += max(1, processed // 5)

            # CPU effect: each processed request adds load, with natural decay
            cpu_from_queue = node.queue_length * 0.008
            cpu_from_processing = processed * 0.012
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
                self._redistribute_from_node(idx)
                # If DB fails, log a critical dependency event
                if sim.nodes[idx].role == "database":
                    sim.action_errors.append(
                        "CRITICAL: Database node failed — all app server "
                        "processing halted until DB is restarted."
                    )

    def _redistribute_from_node(self, idx: int) -> None:
        """Redistribute a failed/removed node's load to its neighbors."""
        sim = self._sim
        node = sim.nodes[idx]
        neighbors = [
            n
            for n in sim.adjacency.get(idx, [])
            if n < len(sim.nodes) and not sim.nodes[n].is_failed
        ]

        if not neighbors:
            return

        load_share = node.cpu_util / len(neighbors)
        queue_share = node.queue_length // max(1, len(neighbors))

        for neighbor_idx in neighbors:
            neighbor = sim.nodes[neighbor_idx]
            neighbor.cpu_util = min(1.0, neighbor.cpu_util + load_share)
            neighbor.queue_length += queue_share

        node.cpu_util = 0.0
        node.queue_length = 0

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

        reward, breakdown = compute_composite_reward(sim)
        self._last_rubric_breakdown = breakdown
        return reward

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

        # --- Build Prometheus-style metrics ---
        prometheus_metrics: List[Dict[str, Any]] = []
        ts = sim.step_count * 30  # simulate 30s intervals

        for i, node in enumerate(sim.nodes):
            node_name = f"worker-{i}"
            is_dropped = i in sim.telemetry_dropout_nodes

            if is_dropped:
                cpu_loads.append(-1.0)
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
                    "metric": "cluster_request_rate",
                    "labels": {},
                    "value": round(sim.current_request_rate * sim.throttle_rate, 2),
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

        return InfraObservation(
            cpu_loads=cpu_loads,
            queue_lengths=queue_lengths,
            failed_nodes=[i for i, n in enumerate(sim.nodes) if n.is_failed],
            latency_ms=round(sim.latency_ms, 2),
            request_rate=round(sim.current_request_rate * sim.throttle_rate, 2),
            step=sim.step_count,
            task_hint=task_hint,
            done=False,
            reward=0.0,
            telemetry_status=telemetry_status,
            action_errors=list(sim.action_errors),
            cloud_budget=sim.cloud_budget,
            prometheus_metrics=prometheus_metrics,
        )
