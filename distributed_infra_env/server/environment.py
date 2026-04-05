"""
Distributed Infrastructure Simulation Engine.

Core simulation logic: weighted node graph, load redistribution,
failure probability model, cascading failure triggers, and the
dense reward function.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Observation, State

# Import from sibling package — works both standalone and in-repo
try:
    from models import InfraAction, InfraObservation, InfraState
except ImportError:
    from distributed_infra_env.models import InfraAction, InfraObservation, InfraState

# Import tasks lazily to avoid circular imports
_TASKS = None


def _get_tasks():
    global _TASKS
    if _TASKS is None:
        try:
            from server.tasks import TASKS
        except ImportError:
            from distributed_infra_env.server.tasks import TASKS
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
    high_cpu_streak: int = 0       # consecutive steps above 90% CPU
    restart_countdown: int = 0     # >0 means the node is restarting
    is_temporary: bool = False     # True for scale-up nodes
    ttl: int = 0                   # remaining lifetime for temp nodes


@dataclass
class SimulationState:
    """Full internal simulation state."""

    nodes: List[Node] = field(default_factory=list)
    adjacency: Dict[int, List[int]] = field(default_factory=dict)
    step_count: int = 0
    base_request_rate: float = 100.0
    current_request_rate: float = 100.0
    throttle_rate: float = 1.0      # 1.0 = accept all
    latency_ms: float = 20.0
    actions_taken: int = 0          # non-no_op actions
    cascade_bonus_awarded: bool = False
    task_id: str = ""
    max_steps: int = 30
    episode_id: str = ""
    # history for grading
    uptime_history: List[float] = field(default_factory=list)
    latency_history: List[float] = field(default_factory=list)
    restart_count: int = 0
    cascade_occurred: bool = False


# ---------------------------------------------------------------------------
# Default graph topology: 8 nodes in a mesh-like structure
# ---------------------------------------------------------------------------

def _build_default_graph(n: int = 8) -> Tuple[List[Node], Dict[int, List[int]]]:
    """Create a default mesh-like graph of n nodes."""
    nodes = [Node(cpu_util = 0.25 + random.uniform(-0.05, 0.05)) for _ in range(n)]

    # Build connected graph: ring + some cross-links for redundancy
    adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        # Ring connections
        right = (i + 1) % n
        if right not in adjacency[i]:
            adjacency[i].append(right)
            adjacency[right].append(i)
        # Skip-one connection for mesh density
        skip = (i + 2) % n
        if skip not in adjacency[i]:
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
            task_score=0.0,
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

        # 1. Process agent action
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
        task_score = 0.0
        if sim.task_id in tasks:
            task_info = tasks[sim.task_id]
            if task_info["is_done"](self):
                done = True
            task_score = task_info["grade"](self)

        self._state.task_score = task_score

        obs = self._make_observation()
        obs.reward = reward
        obs.done = done
        obs.metadata = {
            "task_score": task_score,
            "uptime_ratio": uptime_ratio,
            "actions_taken": sim.actions_taken,
        }

        return obs

    @property
    def state(self) -> InfraState:
        return self._state

    # ----- Action handlers -----

    def _apply_action(self, action: InfraAction) -> None:
        sim = self._sim

        if action.action_type == "no_op":
            return

        sim.actions_taken += 1

        if action.action_type == "restart_node":
            idx = action.target
            if idx is not None and 0 <= idx < len(sim.nodes):
                node = sim.nodes[idx]
                if node.is_failed and node.restart_countdown == 0:
                    node.restart_countdown = 2  # 2-step delay
                    sim.restart_count += 1

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
                    # Agent is proactively redistributing before cascade
                    for neighbor_idx in sim.adjacency.get(src, []):
                        if (
                            not sim.nodes[neighbor_idx].is_failed
                            and sim.nodes[neighbor_idx].cpu_util > CASCADE_AWARENESS_THRESHOLD
                        ):
                            sim.cascade_bonus_awarded = True
                            break

        elif action.action_type == "scale_up":
            # Add temporary capacity node
            new_idx = len(sim.nodes)
            new_node = Node(
                cpu_util=0.1,
                queue_length=0,
                is_temporary=True,
                ttl=10,
            )
            sim.nodes.append(new_node)
            # Connect to a few existing nodes
            sim.adjacency[new_idx] = []
            connect_to = self._rng.sample(
                range(new_idx), min(3, new_idx)
            )
            for c in connect_to:
                sim.adjacency[new_idx].append(c)
                sim.adjacency[c].append(new_idx)

        elif action.action_type == "throttle":
            if action.rate is not None:
                sim.throttle_rate = action.rate

    # ----- Simulation dynamics -----

    def _simulate_requests(self) -> None:
        """Simulate incoming request arrivals using Poisson process."""
        sim = self._sim
        # Variable-rate Poisson with burst potential
        effective_rate = sim.current_request_rate * sim.throttle_rate

        # Poisson arrival count for this step
        arrival_count = self._rng.gauss(effective_rate, effective_rate * 0.15)
        arrival_count = max(0, int(arrival_count))

        # Distribute arrivals across operational nodes
        operational = [
            i
            for i, n in enumerate(sim.nodes)
            if not n.is_failed and n.restart_countdown == 0
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
        """Update restart countdowns and temporary node TTLs."""
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
                new_v = [
                    (x if x < idx else x - 1) for x in v if x != idx
                ]
                new_adj[new_k] = new_v
            sim.adjacency = new_adj

    def _distribute_load(self) -> None:
        """Process queued requests → affect CPU utilization."""
        sim = self._sim
        for i, node in enumerate(sim.nodes):
            if node.is_failed or node.restart_countdown > 0:
                continue

            # Process some requests from queue
            processed = min(node.queue_length, node.capacity)
            node.queue_length = max(0, node.queue_length - processed)

            # CPU effect: each processed request adds load, with natural decay
            cpu_from_queue = node.queue_length * 0.008
            cpu_from_processing = processed * 0.012
            natural_decay = 0.05

            node.cpu_util = max(
                0.05,
                min(
                    1.0,
                    node.cpu_util + cpu_from_queue + cpu_from_processing - natural_decay
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
        cpu_latency = (avg_cpu ** 2) * 80.0  # non-linear increase under load

        new_latency = base_latency + queue_latency + cpu_latency
        # Exponential moving average
        sim.latency_ms = sim.latency_ms * 0.3 + new_latency * 0.7

    def _check_failures(self) -> None:
        """Check if nodes fail due to sustained high CPU, trigger cascades."""
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

    # ----- Reward computation -----

    def _compute_reward(self) -> float:
        """
        Dense step-level reward signal:

        R(t) = 0.40 * uptime_ratio
             - 0.30 * normalized_latency
             - 0.20 * overload_fraction
             - 0.10 * (actions_taken / max_steps)
             + 0.50 * cascade_prevented_bonus
        """
        sim = self._sim
        total = len(sim.nodes)
        if total == 0:
            return 0.0

        alive = sum(1 for n in sim.nodes if not n.is_failed)
        uptime_ratio = alive / total

        normalized_latency = min(2.0, sim.latency_ms / TARGET_LATENCY_MS)

        overloaded = sum(
            1
            for n in sim.nodes
            if not n.is_failed and n.cpu_util > OVERLOAD_THRESHOLD
        )
        overload_fraction = overloaded / total

        action_penalty = sim.actions_taken / max(1, sim.max_steps)

        cascade_bonus = 0.5 if sim.cascade_bonus_awarded else 0.0

        reward = (
            0.40 * uptime_ratio
            - 0.30 * normalized_latency
            - 0.20 * overload_fraction
            - 0.10 * action_penalty
            + cascade_bonus
        )

        # Reset one-time bonus so it's not double-counted
        if sim.cascade_bonus_awarded:
            sim.cascade_bonus_awarded = False

        return round(reward, 4)

    # ----- Observation builder -----

    def _make_observation(self) -> InfraObservation:
        """Build the current observation for the agent."""
        sim = self._sim
        tasks = _get_tasks()

        task_hint = ""
        if sim.task_id in tasks:
            task_hint = tasks[sim.task_id].get("hint", "")

        return InfraObservation(
            cpu_loads=[round(n.cpu_util, 3) for n in sim.nodes],
            queue_lengths=[n.queue_length for n in sim.nodes],
            failed_nodes=[i for i, n in enumerate(sim.nodes) if n.is_failed],
            latency_ms=round(sim.latency_ms, 2),
            request_rate=round(sim.current_request_rate * sim.throttle_rate, 2),
            step=sim.step_count,
            task_hint=task_hint,
            done=False,
            reward=0.0,
        )
