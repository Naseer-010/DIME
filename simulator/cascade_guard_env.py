from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

from server.models import InfraAction, InfraObservation

from .backend_bridge import EmbeddedBackendBridge, TASK_IDS

RenderMode = Literal["human", "rgb_array"]

# ─── Layout ───────────────────────────────────────────────────────────────────
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
TOP_HUD_H = 72
LEFT_PANEL_W = 208
RIGHT_PANEL_W = 304
BOTTOM_H = 112
_G = 8

GRAPH_LEFT = LEFT_PANEL_W + _G
GRAPH_TOP = TOP_HUD_H + _G
GRAPH_RIGHT = WINDOW_WIDTH - RIGHT_PANEL_W - _G
GRAPH_BOTTOM = WINDOW_HEIGHT - BOTTOM_H - _G
GRAPH_WIDTH = GRAPH_RIGHT - GRAPH_LEFT
GRAPH_HEIGHT = GRAPH_BOTTOM - GRAPH_TOP
DETAIL_LEFT = GRAPH_RIGHT + _G
DETAIL_TOP = GRAPH_TOP
DETAIL_WIDTH = RIGHT_PANEL_W - _G
DETAIL_HEIGHT = GRAPH_HEIGHT

MAX_NODES = 20
MAX_EDGES = 64
HISTORY_LEN = 60

# ─── Labels / Topology ────────────────────────────────────────────────────────
ACTION_LABELS = {
    0: "Observe",
    1: "Reroute",
    2: "Restart",
    3: "Scale Out",
    4: "Throttle",
    5: "Balance",
}
TASK_LABELS = {
    "traffic_spike": "Traffic Spike",
    "node_failure": "Node Failure",
    "cascading_failure": "Cascade Guard",
    "flash_crowd": "Flash Crowd",
    "level_1_read_logs": "Level 1: Read Logs",
    "level_2_single_fix": "Level 2: Single Fix",
    "level_3_stochastic": "Level 3: Stochastic",
    "level_4_expert": "Level 4: Expert",
}

BASE_IDS = ("CTRL", "GW-A", "GW-B", "CMP-1", "CMP-2", "CMP-3", "DB-A", "DB-B")
BASE_LABELS = (
    "Control Nexus",
    "Gateway A",
    "Gateway B",
    "Compute 1",
    "Compute 2",
    "Compute 3",
    "Data Store A",
    "Data Store B",
)
# Positions computed relative to the graph viewport
BASE_POSITIONS = (
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.50), GRAPH_TOP + int(GRAPH_HEIGHT * 0.10)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.22), GRAPH_TOP + int(GRAPH_HEIGHT * 0.26)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.78), GRAPH_TOP + int(GRAPH_HEIGHT * 0.26)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.15), GRAPH_TOP + int(GRAPH_HEIGHT * 0.58)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.40), GRAPH_TOP + int(GRAPH_HEIGHT * 0.42)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.62), GRAPH_TOP + int(GRAPH_HEIGHT * 0.42)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.22), GRAPH_TOP + int(GRAPH_HEIGHT * 0.82)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.78), GRAPH_TOP + int(GRAPH_HEIGHT * 0.82)),
)
TEMP_POSITIONS = (
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.90), GRAPH_TOP + int(GRAPH_HEIGHT * 0.14)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.94), GRAPH_TOP + int(GRAPH_HEIGHT * 0.28)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.95), GRAPH_TOP + int(GRAPH_HEIGHT * 0.44)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.93), GRAPH_TOP + int(GRAPH_HEIGHT * 0.60)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.89), GRAPH_TOP + int(GRAPH_HEIGHT * 0.74)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.80), GRAPH_TOP + int(GRAPH_HEIGHT * 0.88)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.62), GRAPH_TOP + int(GRAPH_HEIGHT * 0.93)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.46), GRAPH_TOP + int(GRAPH_HEIGHT * 0.95)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.28), GRAPH_TOP + int(GRAPH_HEIGHT * 0.93)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.12), GRAPH_TOP + int(GRAPH_HEIGHT * 0.86)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.05), GRAPH_TOP + int(GRAPH_HEIGHT * 0.72)),
    (GRAPH_LEFT + int(GRAPH_WIDTH * 0.04), GRAPH_TOP + int(GRAPH_HEIGHT * 0.55)),
)

# Node kind → sprite shape
NODE_KINDS: dict[str, str] = {
    "CTRL": "control",
    "GW-A": "gateway",
    "GW-B": "gateway",
    "CMP-1": "compute",
    "CMP-2": "compute",
    "CMP-3": "compute",
    "DB-A": "storage",
    "DB-B": "storage",
}

# ─── Retro Arcade Colour Palette (NES / SNES inspired) ───────────────────────
C: dict[str, tuple] = {
    "bg": (8, 12, 32),
    "bg_mid": (14, 20, 48),
    "bg_panel": (20, 28, 60),
    "bg_card": (28, 38, 76),
    "bg_card2": (36, 48, 90),
    "border": (80, 100, 180),
    "border_hi": (140, 160, 255),
    "border_outer": (48, 60, 110),
    "text": (252, 252, 252),
    "text_dim": (188, 200, 228),
    "text_muted": (120, 140, 190),
    "text_dark": (70, 88, 140),
    "healthy": (96, 214, 96),  # Mario green
    "stressed": (248, 200, 40),  # Coin yellow
    "overloaded": (248, 140, 24),  # Fire orange
    "failed": (232, 48, 56),  # Mario red
    "recovering": (176, 96, 248),  # Power-up purple
    "accent": (56, 200, 248),  # P-Switch blue
    "accent2": (248, 200, 40),  # Coin gold
    "success": (96, 214, 96),
    "warning": (248, 200, 40),
    "critical": (232, 48, 56),
    "info": (56, 200, 248),
    "white": (252, 252, 252),
    "coin": (248, 200, 40),
    "node": (78, 149, 255),
    "temp": (199, 142, 255),
    "edge": (54, 86, 122),
    "focus": (255, 230, 102),
}

STATUS_SCALAR = {"healthy": 0.1, "warning": 0.45, "stressed": 0.72, "failed": 1.0}


@dataclass
class NodeSpec:
    node_id: str
    label: str
    position: tuple[int, int]
    index: int
    temporary: bool = False


@dataclass
class EventChip:
    title: str
    detail: str
    severity: str
    created_at: float


class CascadeGuardMissionControlEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        max_steps: int = 240,
        difficulty: float = 1.0,
        reduced_motion: bool = False,
        enable_audio: bool = False,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.reduced_motion = reduced_motion
        self.enable_audio = enable_audio

        self.action_space = spaces.Discrete(len(ACTION_LABELS))
        self.observation_space = spaces.Dict(
            {
                "node_metrics": spaces.Box(
                    0.0, 1.0, shape=(MAX_NODES, 7), dtype=np.float32
                ),
                "edge_metrics": spaces.Box(
                    0.0, 1.0, shape=(MAX_EDGES, 4), dtype=np.float32
                ),
                "global_metrics": spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32),
            }
        )

        self.bridge = EmbeddedBackendBridge()
        self._pygame: Any = None
        self.window: Any = None
        self.canvas: Any = None
        self.clock: Any = None
        self.window_size = (WINDOW_WIDTH, WINDOW_HEIGHT)
        self.quit_requested = False
        self._terminated = False
        self._truncated = False
        self._overlay_renderer: Any | None = None
        self.demo_callout: dict[str, Any] | None = None
        self._ai_mode = False
        self._scenario_name = "Traffic Spike"

        # Retro rendering state
        self._blink_timer: float = 0.0
        self._blink_state: bool = True
        self._last_render_time: float | None = None
        self.hovered_node_id: str | None = None
        self._font_cache: dict[tuple, Any] = {}

        # Game state (set by _sync, populated by reset)
        self.node_specs: list[NodeSpec] = []
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[tuple[int, int, dict[str, Any]]] = []
        self.events: list[EventChip] = []
        self.badges: list[str] = []
        self.metric_history = {
            "latency": deque([20.0] * HISTORY_LEN, maxlen=HISTORY_LEN),
            "throughput": deque([1200.0] * HISTORY_LEN, maxlen=HISTORY_LEN),
            "uptime": deque([100.0] * HISTORY_LEN, maxlen=HISTORY_LEN),
            "resilience": deque([100.0] * HISTORY_LEN, maxlen=HISTORY_LEN),
            "failed": deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN),
        }
        self.selected_index = 0
        self.selected_node_id = BASE_IDS[0]
        self.task_id = "traffic_spike"
        self.task_hint = ""
        self.current_step = 0
        self.request_rate = 100.0
        self.avg_latency_ms = 20.0
        self.throughput_rps = 1200
        self.uptime_pct = 100.0
        self.resilience_score = 100.0
        self.failed_nodes = 0
        self.overloaded_nodes = 0
        self.stable_streak = 0
        self.best_stable_streak = 0
        self.mission_score = 0.0
        self.task_score = 0.0
        self.last_action = ACTION_LABELS[0]
        self.last_backend_action = "no_op"
        self._last_failed: set[int] = set()
        self._last_node_count = 0
        self.reset()

    @property
    def episode_over(self) -> bool:
        return self._terminated or self._truncated

    # ─── Gymnasium API ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        task_id = (
            options.get("task")
            or options.get("task_id")
            or os.environ.get("SIMULATOR_TASK", "traffic_spike")
        )
        if task_id not in TASK_IDS:
            task_id = "traffic_spike"
        self.task_id = task_id
        self._scenario_name = TASK_LABELS.get(
            task_id, task_id.replace("_", " ").title()
        )
        self.quit_requested = False
        self._terminated = False
        self._truncated = False
        self.events = []
        self.badges = []
        self.stable_streak = 0
        self.best_stable_streak = 0
        self.mission_score = 0.0
        self.last_action = ACTION_LABELS[0]
        self.last_backend_action = "no_op"
        step = self.bridge.reset(seed=seed, task_id=task_id)
        self._sync(step.observation, reward=0.0, metadata=step.metadata)
        self._push_event(
            "Mission control online", f"{self._scenario_name} loaded.", "success"
        )
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if self.episode_over:
            return (
                self._get_obs(),
                0.0,
                self._terminated,
                self._truncated,
                self._get_info(),
            )
        backend_action = self._translate_action(action)
        self.last_action = ACTION_LABELS[int(action)]
        self.last_backend_action = backend_action.action_type
        step = self.bridge.step(backend_action)
        self._sync(step.observation, reward=step.reward, metadata=step.metadata)
        self._terminated = bool(step.done)
        self._truncated = False
        if self.render_mode == "human":
            self._render_frame()
        return (
            self._get_obs(),
            float(step.reward),
            self._terminated,
            self._truncated,
            self._get_info(),
        )

    def render(self):
        if self.render_mode is None:
            raise ValueError("render() called without render_mode.")
        return self._render_frame()

    def close(self) -> None:
        self.bridge.close()
        if self._pygame is None:
            return
        if self.window is not None:
            self._pygame.display.quit()
        self._pygame.font.quit()
        self._pygame.quit()
        self._pygame = None

    # ─── Backend bridge helpers ───────────────────────────────────────────────

    def _translate_action(self, action: int) -> InfraAction:
        obs = self.bridge.last_observation
        if obs is None:
            return InfraAction(action_type="no_op")
        active = [i for i in range(len(obs.cpu_loads)) if i not in obs.failed_nodes]
        if action == 0:
            return InfraAction(action_type="no_op")
        if action in {1, 5} and len(active) >= 2:
            if action == 1:
                src = max(
                    active, key=lambda i: (obs.cpu_loads[i], obs.queue_lengths[i])
                )
                dst = min(
                    active, key=lambda i: (obs.cpu_loads[i], obs.queue_lengths[i])
                )
            else:
                src = max(
                    active, key=lambda i: (obs.queue_lengths[i], obs.cpu_loads[i])
                )
                dst = min(
                    active, key=lambda i: (obs.queue_lengths[i], obs.cpu_loads[i])
                )
            return (
                InfraAction(action_type="reroute_traffic", from_node=src, to_node=dst)
                if src != dst
                else InfraAction(action_type="no_op")
            )
        if action == 2 and obs.failed_nodes:
            target = max(
                obs.failed_nodes,
                key=lambda i: obs.queue_lengths[i] if i < len(obs.queue_lengths) else 0,
            )
            return InfraAction(action_type="restart_node", target=target)
        if action == 3:
            return InfraAction(action_type="scale_up")
        if action == 4:
            return InfraAction(
                action_type="throttle", rate=0.65 if obs.latency_ms >= 90 else 0.8
            )
        return InfraAction(action_type="no_op")

    def _sync(
        self, obs: InfraObservation, *, reward: float, metadata: dict[str, Any]
    ) -> None:
        cpu = list(obs.cpu_loads)
        queue = list(obs.queue_lengths)
        failed = set(obs.failed_nodes)
        node_count = len(cpu)
        adjacency = self.bridge.adjacency or self._default_adjacency(node_count)
        self.node_specs = self._build_specs(node_count)
        self.nodes = []
        total_cpu = max(0.001, sum(cpu))
        self.current_step = int(obs.step)
        self.task_hint = obs.task_hint
        self.request_rate = float(obs.request_rate)
        self.avg_latency_ms = float(obs.latency_ms)
        self.failed_nodes = len(failed)
        self.overloaded_nodes = sum(
            1 for i, v in enumerate(cpu) if i not in failed and v >= 0.85
        )
        self.uptime_pct = 100.0 * (node_count - len(failed)) / max(1, node_count)
        stable = (
            self.failed_nodes == 0
            and self.overloaded_nodes <= 1
            and self.avg_latency_ms <= 60
        )
        self.stable_streak = self.stable_streak + 1 if stable else 0
        self.best_stable_streak = max(self.best_stable_streak, self.stable_streak)
        avg_cpu = sum(cpu) / max(1, node_count)
        latency_penalty = min(70.0, max(0.0, self.avg_latency_ms - 20.0) * 0.55)
        self.resilience_score = float(
            np.clip(
                100.0
                - latency_penalty
                - self.failed_nodes * 18.0
                - self.overloaded_nodes * 9.0
                - avg_cpu * 14.0,
                0.0,
                100.0,
            )
        )
        throughput_factor = np.clip(
            1.08 - (self.avg_latency_ms / 220.0) - (self.failed_nodes * 0.08), 0.18, 1.0
        )
        self.throughput_rps = int(self.request_rate * 10.0 * throughput_factor)
        self.mission_score = max(
            0.0,
            self.mission_score + reward * 130.0 + (6.0 if stable else -2.0),
        )
        self.task_score = float(metadata.get("task_score", self.task_score or 0.0))
        for spec in self.node_specs:
            i = spec.index
            is_failed = i in failed
            status = (
                "failed"
                if is_failed
                else "stressed"
                if cpu[i] >= 0.85
                else "warning"
                if cpu[i] >= 0.65 or queue[i] >= 18
                else "healthy"
            )
            risk = float(
                np.clip(
                    cpu[i] * 0.78 + queue[i] / 40.0 + (0.22 if is_failed else 0.0),
                    0.0,
                    1.0,
                )
            )
            latency = (
                12.0 + cpu[i] * 68.0 + queue[i] * 1.45 + (54.0 if is_failed else 0.0)
            )
            note = (
                "Offline. Awaiting restart."
                if is_failed
                else "Critical heat. Intervention needed."
                if cpu[i] >= 0.9
                else "Queue pressure climbing."
                if queue[i] >= 18
                else "Amber band. Watch closely."
                if status == "warning"
                else "Routing within safe bounds."
            )
            self.nodes.append(
                {
                    "cpu": cpu[i],
                    "queue": queue[i],
                    "failed": is_failed,
                    "status": status,
                    "latency": latency,
                    "risk": risk,
                    "traffic": cpu[i] / total_cpu,
                    "bonus": 0.22 if spec.temporary else 0.0,
                    "note": note,
                }
            )
        self.edges = []
        for src, neighbors in adjacency.items():
            if src >= node_count:
                continue
            for dst in neighbors:
                if dst <= src or dst >= node_count:
                    continue
                traffic = float(np.clip((cpu[src] + cpu[dst]) / 2.0, 0.0, 1.0))
                congestion = float(np.clip((queue[src] + queue[dst]) / 40.0, 0.0, 1.0))
                status = (
                    "down"
                    if src in failed or dst in failed
                    else "degraded"
                    if congestion >= 0.6
                    else "nominal"
                )
                self.edges.append(
                    (
                        src,
                        dst,
                        {
                            "traffic": traffic,
                            "congestion": congestion,
                            "latency": 8.0 + traffic * 54.0 + congestion * 42.0,
                            "status": status,
                        },
                    )
                )
        self.edges = self.edges[:MAX_EDGES]
        self.metric_history["latency"].append(self.avg_latency_ms)
        self.metric_history["throughput"].append(float(self.throughput_rps))
        self.metric_history["uptime"].append(self.uptime_pct)
        self.metric_history["resilience"].append(self.resilience_score)
        self.metric_history["failed"].append(float(self.failed_nodes))
        self._sync_events(obs, node_count, metadata)
        self._sync_badges()
        self.selected_index = min(self.selected_index, max(0, len(self.node_specs) - 1))
        if self.node_specs:
            self.selected_node_id = self.node_specs[self.selected_index].node_id
        self._last_failed = failed
        self._last_node_count = node_count

    def _sync_events(
        self, obs: InfraObservation, node_count: int, metadata: dict[str, Any]
    ) -> None:
        failed = set(obs.failed_nodes)
        for index in sorted(failed - self._last_failed):
            self._push_event(
                f"{self.node_specs[index].node_id} down",
                f"{self.node_specs[index].label} dropped out of service.",
                "critical",
            )
        for index in sorted(self._last_failed - failed):
            if index < len(self.node_specs):
                self._push_event(
                    f"{self.node_specs[index].node_id} recovered",
                    f"{self.node_specs[index].label} is back online.",
                    "success",
                )
        if node_count > self._last_node_count:
            self._push_event(
                "Scale-out complete",
                f"Fleet expanded to {node_count} nodes.",
                "success",
            )
        if node_count < self._last_node_count:
            self._push_event(
                "Burst capacity retired",
                f"Fleet contracted to {node_count} nodes.",
                "info",
            )
        if self.avg_latency_ms >= 120 and self.metric_history["latency"][-2] < 120:
            self._push_event(
                "Latency critical",
                f"Average latency is {self.avg_latency_ms:.1f} ms.",
                "warning",
            )
        if self.last_action == "Throttle" and self.request_rate <= 85:
            self._push_event(
                "Ingress trimmed",
                f"Effective load dropped to {self.request_rate:.0f} req/s.",
                "info",
            )
        if metadata.get("task_score", 0.0) >= 0.8 and self.current_step > 6:
            self._push_event(
                "Task score rising",
                f"Current task score is {metadata['task_score']:.2f}.",
                "success",
            )

    def _sync_badges(self) -> None:
        for label, ok in (
            ("Zero-Downtime", self.failed_nodes == 0 and self.current_step >= 8),
            ("Green Zone", self.avg_latency_ms <= 50 and self.failed_nodes == 0),
            ("Stability Lock", self.best_stable_streak >= 6),
        ):
            if ok and label not in self.badges:
                self.badges.append(label)
                self._push_event(label, f"{label} achieved.", "success")

    def _push_event(self, title: str, detail: str, severity: str) -> None:
        if (
            self.events
            and self.events[0].title == title
            and self.events[0].detail == detail
        ):
            return
        self.events.insert(0, EventChip(title, detail, severity, time.perf_counter()))
        self.events = self.events[:8]

    def _build_specs(self, count: int) -> list[NodeSpec]:
        specs = []
        for i in range(count):
            if i < len(BASE_IDS):
                specs.append(
                    NodeSpec(BASE_IDS[i], BASE_LABELS[i], BASE_POSITIONS[i], i, False)
                )
            else:
                j = i - len(BASE_IDS)
                specs.append(
                    NodeSpec(
                        f"TMP-{j + 1}",
                        f"Burst Node {j + 1}",
                        TEMP_POSITIONS[j % len(TEMP_POSITIONS)],
                        i,
                        True,
                    )
                )
        return specs

    def _default_adjacency(self, count: int) -> dict[int, list[int]]:
        adjacency: dict[int, list[int]] = {i: [] for i in range(count)}
        for i in range(count):
            for dst in (
                (i + 1) % count,
                (i + 2) % count if count > 2 else (i + 1) % count,
            ):
                if dst not in adjacency[i]:
                    adjacency[i].append(dst)
                if i not in adjacency[dst]:
                    adjacency[dst].append(i)
        return adjacency

    def _get_obs(self) -> dict[str, np.ndarray]:
        node_metrics = np.zeros((MAX_NODES, 7), dtype=np.float32)
        for spec, node in zip(self.node_specs[:MAX_NODES], self.nodes[:MAX_NODES]):
            node_metrics[spec.index] = np.array(
                [
                    np.clip(node["cpu"], 0.0, 1.0),
                    np.clip(node["queue"] / 40.0, 0.0, 1.0),
                    np.clip(node["traffic"], 0.0, 1.0),
                    np.clip(node["latency"] / 180.0, 0.0, 1.0),
                    np.clip(node["risk"], 0.0, 1.0),
                    STATUS_SCALAR[node["status"]],
                    np.clip(node["bonus"], 0.0, 1.0),
                ],
                dtype=np.float32,
            )
        edge_metrics = np.zeros((MAX_EDGES, 4), dtype=np.float32)
        for i, (_, _, edge) in enumerate(self.edges[:MAX_EDGES]):
            edge_metrics[i] = np.array(
                [
                    np.clip(edge["traffic"], 0.0, 1.0),
                    np.clip(edge["congestion"], 0.0, 1.0),
                    np.clip(edge["latency"] / 180.0, 0.0, 1.0),
                    1.0
                    if edge["status"] == "down"
                    else 0.55
                    if edge["status"] == "degraded"
                    else 0.1,
                ],
                dtype=np.float32,
            )
        globals_ = np.array(
            [
                np.clip(self.request_rate / 300.0, 0.0, 1.0),
                np.clip(self.avg_latency_ms / 180.0, 0.0, 1.0),
                np.clip(self.throughput_rps / 3200.0, 0.0, 1.0),
                np.clip(self.uptime_pct / 100.0, 0.0, 1.0),
                np.clip(len(self.node_specs) / MAX_NODES, 0.0, 1.0),
                np.clip(self.overloaded_nodes / max(1, len(self.node_specs)), 0.0, 1.0),
                np.clip(self.resilience_score / 100.0, 0.0, 1.0),
                np.clip(self.stable_streak / 20.0, 0.0, 1.0),
                np.clip(self.task_score, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return {
            "node_metrics": node_metrics,
            "edge_metrics": edge_metrics,
            "global_metrics": globals_,
        }

    def _get_info(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_label": self._scenario_name,
            "task_hint": self.task_hint,
            "step": self.current_step,
            "latency_ms": round(self.avg_latency_ms, 1),
            "request_rate": round(self.request_rate, 1),
            "throughput_rps": self.throughput_rps,
            "uptime_pct": round(self.uptime_pct, 2),
            "resilience_score": round(self.resilience_score, 1),
            "failed_nodes": self.failed_nodes,
            "overloaded_nodes": self.overloaded_nodes,
            "stable_streak": self.stable_streak,
            "best_stable_streak": self.best_stable_streak,
            "mission_score": int(self.mission_score),
            "task_score": round(self.task_score, 4),
            "selected_node_id": self.selected_node_id,
            "badges": list(self.badges),
            "events": [event.title for event in self.events[:5]],
            "action_label": self.last_action,
            "backend_action": self.last_backend_action,
        }

    # ─── Pygame init ──────────────────────────────────────────────────────────

    def _ensure_pygame(self) -> None:
        if self._pygame is not None:
            return
        try:
            import pygame
        except ImportError as exc:
            raise DependencyNotInstalled(
                "pygame required for CascadeGuard rendering."
            ) from exc

        # Tell Windows we handle DPI ourselves so it doesn't scale the window up
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        self._pygame = pygame
        pygame.init()
        pygame.font.init()
        self.canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

        if self.render_mode == "human":
            # Fit the window inside 90 % of the physical screen (no upscaling)
            info = pygame.display.Info()
            max_w = max(800, int(info.current_w * 0.90))
            max_h = max(600, int(info.current_h * 0.90))
            scale = min(max_w / WINDOW_WIDTH, max_h / WINDOW_HEIGHT, 1.0)
            win_w = int(WINDOW_WIDTH * scale)
            win_h = int(WINDOW_HEIGHT * scale)
            self.window_size = (win_w, win_h)
            self.window = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
            pygame.display.set_caption(
                "CascadeGuard Mission Control  [ ARCADE EDITION ]"
            )
            self.clock = pygame.time.Clock()

    def _handle_events(self) -> None:
        pg = self._pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit_requested = True
            elif event.type == pg.VIDEORESIZE and self.window is not None:
                self.window_size = (max(900, event.w), max(600, event.h))
                self.window = pg.display.set_mode(self.window_size, pg.RESIZABLE)
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_TAB and self.node_specs:
                    self.selected_index = (self.selected_index + 1) % len(
                        self.node_specs
                    )
                    self.selected_node_id = self.node_specs[self.selected_index].node_id
                elif event.key == pg.K_m:
                    self.reduced_motion = not self.reduced_motion
            elif (
                event.type == pg.MOUSEBUTTONDOWN
                and event.button == 1
                and self.window is not None
            ):
                sx = event.pos[0] * WINDOW_WIDTH / max(1, self.window_size[0])
                sy = event.pos[1] * WINDOW_HEIGHT / max(1, self.window_size[1])
                for spec in self.node_specs:
                    dx, dy = sx - spec.position[0], sy - spec.position[1]
                    if dx * dx + dy * dy <= 30 * 30:
                        self.selected_index = spec.index
                        self.selected_node_id = spec.node_id
                        break
            elif event.type == pg.MOUSEMOTION and self.window is not None:
                sx = event.pos[0] * WINDOW_WIDTH / max(1, self.window_size[0])
                sy = event.pos[1] * WINDOW_HEIGHT / max(1, self.window_size[1])
                self.hovered_node_id = None
                for spec in self.node_specs:
                    dx, dy = sx - spec.position[0], sy - spec.position[1]
                    if dx * dx + dy * dy <= 28 * 28:
                        self.hovered_node_id = spec.node_id
                        break

    # ─── Main render frame ────────────────────────────────────────────────────

    def _render_frame(self):
        self._ensure_pygame()
        pg = self._pygame
        now = time.perf_counter()

        # Delta time for animations
        dt = (
            1.0 / self.metadata["render_fps"]
            if self._last_render_time is None
            else min(0.08, now - self._last_render_time)
        )
        self._last_render_time = now

        # 0.5 s blink cycle
        self._blink_timer += dt
        if self._blink_timer > 0.5:
            self._blink_timer = 0.0
            self._blink_state = not self._blink_state

        if self.render_mode == "human":
            self._handle_events()
            if self.quit_requested:
                return None

        # Draw all layers
        self.canvas.fill(C["bg"])
        self._draw_background(self.canvas, now)
        self._draw_left_panel(self.canvas, now)
        self._draw_edges(self.canvas, now)
        self._draw_nodes(self.canvas, now)
        self._draw_hud(self.canvas, now)
        self._draw_event_feed(self.canvas, now)
        self._draw_node_detail(self.canvas, now)
        if self.demo_callout and not self.episode_over:
            self._draw_callout(self.canvas)
        if self._overlay_renderer is not None:
            self._overlay_renderer(self.canvas)
        self._draw_scanlines(self.canvas)

        if self.render_mode == "human":
            if self.window_size == (WINDOW_WIDTH, WINDOW_HEIGHT):
                self.window.blit(self.canvas, (0, 0))
            else:
                scaled = pg.transform.smoothscale(self.canvas, self.window_size)
                self.window.blit(scaled, (0, 0))
            pg.display.flip()
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
            return None

        frame = pg.surfarray.array3d(self.canvas)
        return np.transpose(frame, (1, 0, 2))

    # ─── Layer: background ────────────────────────────────────────────────────

    def _draw_background(self, surface: Any, now: float) -> None:
        pg = self._pygame
        pg.draw.rect(
            surface, C["bg_mid"], (GRAPH_LEFT, GRAPH_TOP, GRAPH_WIDTH, GRAPH_HEIGHT)
        )

        # Animated starfield
        if not self.reduced_motion:
            rng = np.random.default_rng(42 + int(now * 0.3) % 200)
            stars = rng.integers(0, [GRAPH_WIDTH, GRAPH_HEIGHT], size=(60, 2))
            for sx, sy in stars:
                brightness = int(rng.integers(80, 200))
                twinkle = int(abs(math.sin(now * 2.5 + int(sx) * 0.1)) * 80)
                spark = min(255, brightness + twinkle)
                size = 1 if brightness < 140 else 2
                pg.draw.rect(
                    surface,
                    (spark, spark, spark),
                    (GRAPH_LEFT + int(sx), GRAPH_TOP + int(sy), size, size),
                )

        # Semi-transparent pixel grid
        grid_surf = pg.Surface((GRAPH_WIDTH, GRAPH_HEIGHT), pg.SRCALPHA)
        for gx in range(0, GRAPH_WIDTH, 48):
            pg.draw.line(grid_surf, (40, 60, 120, 30), (gx, 0), (gx, GRAPH_HEIGHT))
        for gy in range(0, GRAPH_HEIGHT, 48):
            pg.draw.line(grid_surf, (40, 60, 120, 30), (0, gy), (GRAPH_WIDTH, gy))
        surface.blit(grid_surf, (GRAPH_LEFT, GRAPH_TOP))

        # Horizontal zone bands
        zones = [
            ("COMMAND", 0.00, 0.18, (60, 30, 90)),
            ("INGRESS", 0.18, 0.36, (20, 50, 100)),
            ("COMPUTE", 0.36, 0.68, (20, 80, 60)),
            ("STORAGE", 0.68, 1.00, (80, 40, 20)),
        ]
        for label, ystart, yend, band_col in zones:
            y1 = GRAPH_TOP + int(GRAPH_HEIGHT * ystart)
            h = int(GRAPH_HEIGHT * (yend - ystart))
            band = pg.Surface((GRAPH_WIDTH - 4, h), pg.SRCALPHA)
            band.fill((*band_col, 16))
            surface.blit(band, (GRAPH_LEFT + 2, y1))
            self._draw_pixel_text(
                surface, label, 10, (*band_col, 140), (GRAPH_LEFT + 14, y1 + 5)
            )

        self._draw_pixel_border(
            surface,
            GRAPH_LEFT,
            GRAPH_TOP,
            GRAPH_WIDTH,
            GRAPH_HEIGHT,
            C["border"],
            C["border_outer"],
            thick=3,
        )

        # "INFRASTRUCTURE MAP" label in top-left of graph
        title_bg = pg.Surface((228, 22), pg.SRCALPHA)
        title_bg.fill((0, 0, 0, 140))
        surface.blit(title_bg, (GRAPH_LEFT + 10, GRAPH_TOP + 8))
        self._draw_pixel_text(
            surface,
            "INFRASTRUCTURE MAP",
            11,
            C["accent"],
            (GRAPH_LEFT + 16, GRAPH_TOP + 12),
        )

    # ─── Layer: left stats panel ──────────────────────────────────────────────

    def _draw_left_panel(self, surface: Any, now: float) -> None:
        pg = self._pygame
        px, py = 4, TOP_HUD_H + _G
        pw = LEFT_PANEL_W - 8
        ph = WINDOW_HEIGHT - TOP_HUD_H - BOTTOM_H - _G * 2

        pg.draw.rect(surface, C["bg_panel"], (px, py, pw, ph))
        self._draw_pixel_border(
            surface, px, py, pw, ph, C["border"], C["border_outer"], thick=3
        )
        self._draw_pixel_text(
            surface, "STATS", 14, C["accent2"], (px + pw // 2, py + 12), center=True
        )
        pg.draw.rect(surface, C["border"], (px + 6, py + 28, pw - 12, 2))

        active_nodes = len(self.node_specs) - self.failed_nodes
        dropped = max(0, int(self.overloaded_nodes * 30 + self.failed_nodes * 80))

        kpis = [
            (
                "LATENCY",
                f"{int(self.avg_latency_ms)}ms",
                self.avg_latency_ms,
                0.0,
                200.0,
                "latency",
                C["healthy"]
                if self.avg_latency_ms < 60
                else C["stressed"]
                if self.avg_latency_ms < 120
                else C["critical"],
            ),
            (
                "THRUPUT",
                f"{self.throughput_rps:,}",
                float(self.throughput_rps),
                0.0,
                5000.0,
                "throughput",
                C["healthy"]
                if self.throughput_rps > 2500
                else C["stressed"]
                if self.throughput_rps > 1500
                else C["critical"],
            ),
            (
                "UPTIME",
                f"{self.uptime_pct:.1f}%",
                self.uptime_pct,
                0.0,
                100.0,
                "uptime",
                C["healthy"]
                if self.uptime_pct > 99
                else C["stressed"]
                if self.uptime_pct > 95
                else C["critical"],
            ),
            (
                "RESILNC",
                f"{int(self.resilience_score)}",
                self.resilience_score,
                0.0,
                100.0,
                "resilience",
                C["healthy"]
                if self.resilience_score > 80
                else C["stressed"]
                if self.resilience_score > 50
                else C["critical"],
            ),
            (
                "NODES",
                f"{active_nodes}/{len(self.node_specs)}",
                float(active_nodes),
                0.0,
                float(max(1, len(self.node_specs))),
                "failed",
                C["healthy"]
                if active_nodes == len(self.node_specs)
                else C["stressed"]
                if active_nodes > 6
                else C["critical"],
            ),
            (
                "DROPPED",
                f"{dropped}/s",
                float(dropped),
                0.0,
                300.0,
                None,
                C["healthy"]
                if dropped < 10
                else C["stressed"]
                if dropped < 60
                else C["critical"],
            ),
        ]

        card_h = (ph - 38) // len(kpis)
        for i, (label, val_str, val, vmin, vmax, hist_key, col) in enumerate(kpis):
            cy = py + 36 + i * card_h
            cx = px + 6
            cw = pw - 12
            pg.draw.rect(surface, C["bg_card"], (cx, cy, cw, card_h - 4))
            self._draw_pixel_border(
                surface, cx, cy, cw, card_h - 4, col, C["bg_panel"], thick=2
            )

            show = not (col == C["critical"] and not self._blink_state)
            self._draw_pixel_text(surface, label, 9, C["text_muted"], (cx + 6, cy + 6))
            if show:
                self._draw_pixel_text(
                    surface, val_str, 15, col, (cx + 6, cy + 20), bold=True
                )

            bar_y = cy + card_h - 18
            bar_w = cw - 14
            frac = float(np.clip((val - vmin) / max(vmax - vmin, 1.0), 0.0, 1.0))
            if label in ("NODES", "DROPPED"):
                frac = 1.0 - frac
            self._draw_pixel_bar(surface, cx + 6, bar_y, bar_w, 7, frac, col, chunks=8)

            if hist_key and hist_key in self.metric_history:
                self._draw_pixel_sparkline(
                    surface,
                    self.metric_history[hist_key],
                    cx + 6,
                    bar_y - 13,
                    bar_w,
                    10,
                    col,
                    invert=(hist_key == "failed"),
                )

    # ─── Layer: edges ─────────────────────────────────────────────────────────

    def _draw_edges(self, surface: Any, now: float) -> None:
        pg = self._pygame
        for src, dst, edge in self.edges:
            if src >= len(self.node_specs) or dst >= len(self.node_specs):
                continue
            start = self.node_specs[src].position
            end = self.node_specs[dst].position
            status = edge["status"]

            if status == "down":
                if self._blink_state:
                    self._draw_dashed_line(
                        surface, C["critical"], start, end, width=2, dash_length=8
                    )
            elif status == "degraded":
                t = float(np.clip(edge["traffic"], 0.0, 1.0))
                lw = int(2 + t * 2)
                pg.draw.line(surface, C["overloaded"], start, end, lw)
                self._draw_pixel_glow_line(
                    surface, C["overloaded"], start, end, lw + 4, alpha=50
                )
            else:
                t = float(np.clip(edge["traffic"], 0.0, 1.0))
                col = (
                    int(48 + t * 8),
                    int(160 + t * 40),
                    int(48 + t * 200),
                )
                lw = max(1, int(1 + t * 3))
                pg.draw.line(surface, col, start, end, lw)
                if t > 0.4:
                    self._draw_pixel_glow_line(
                        surface, col, start, end, lw + 3, alpha=45
                    )

    # ─── Layer: nodes ─────────────────────────────────────────────────────────

    def _draw_nodes(self, surface: Any, now: float) -> None:
        pg = self._pygame
        _radii = {"control": 22, "gateway": 20, "compute": 18, "storage": 20}

        for spec in self.node_specs:
            if spec.index >= len(self.nodes):
                continue
            node = self.nodes[spec.index]
            cx, cy = spec.position
            status = node["status"]
            col = self._node_color(status)
            kind = NODE_KINDS.get(spec.node_id, "compute")
            radius = _radii.get(kind, 18)
            sel = spec.index == self.selected_index
            hov = spec.node_id == self.hovered_node_id

            # Subtle bob for healthy nodes
            if status == "healthy" and not self.reduced_motion:
                cy = int(cy + math.sin(now * 1.8 + spec.index * 0.8) * 2.5)

            # Selection / hover pixel ring
            if sel or hov:
                ring_r = radius + 10
                pulse = 0.55 + 0.45 * math.sin(now * 5)
                ring_alpha = int(220 * pulse) if sel else 140
                ring_col = C["focus"] if sel else col
                ring_surf = pg.Surface((ring_r * 2 + 4, ring_r * 2 + 4), pg.SRCALPHA)
                for deg in range(0, 360, 45):
                    a = math.radians(deg)
                    rx = ring_r + 2 + int(ring_r * math.cos(a)) - 4
                    ry = ring_r + 2 + int(ring_r * math.sin(a)) - 4
                    pg.draw.rect(ring_surf, (*ring_col, ring_alpha), (rx, ry, 8, 8))
                surface.blit(ring_surf, (cx - ring_r - 2, cy - ring_r - 2))

            # Failed blink (draw X)
            if status == "failed" and not self._blink_state:
                pg.draw.rect(
                    surface,
                    (*C["failed"], 80),
                    (cx - radius, cy - radius, radius * 2, radius * 2),
                    2,
                )
                pg.draw.line(
                    surface,
                    C["failed"],
                    (cx - radius + 4, cy - radius + 4),
                    (cx + radius - 4, cy + radius - 4),
                    3,
                )
                pg.draw.line(
                    surface,
                    C["failed"],
                    (cx + radius - 4, cy - radius + 4),
                    (cx - radius + 4, cy + radius - 4),
                    3,
                )
            else:
                # Draw kind-specific sprite
                if kind == "control":
                    self._draw_pixel_star(surface, cx, cy, radius, col)
                elif kind == "gateway":
                    self._draw_pixel_shield(surface, cx, cy, radius, col)
                elif kind == "storage":
                    self._draw_pixel_chest(surface, cx, cy, radius, col)
                else:
                    self._draw_pixel_cpu(surface, cx, cy, radius, col, now)

            # Stressed / warning alert !
            if status == "stressed" and self._blink_state:
                self._draw_pixel_text(
                    surface,
                    "!",
                    14,
                    C["overloaded"],
                    (cx, cy - radius - 18),
                    center=True,
                    bold=True,
                )
            elif status == "warning" and self._blink_state:
                self._draw_pixel_text(
                    surface,
                    "!",
                    12,
                    C["stressed"],
                    (cx, cy - radius - 16),
                    center=True,
                    bold=True,
                )

            # Queue pressure bar above node
            q = float(np.clip(node["queue"] / 30.0, 0.0, 1.0))
            if q > 0.05:
                bar_w = radius * 2 + 8
                self._draw_pixel_bar(
                    surface,
                    cx - bar_w // 2,
                    cy - radius - 14,
                    bar_w,
                    5,
                    1.0 - q,
                    col,
                    chunks=6,
                )

            # Node ID label + CPU %
            label_col = col if (sel or status != "healthy") else C["text_dim"]
            self._draw_pixel_text(
                surface,
                spec.node_id,
                11,
                label_col,
                (cx, cy + radius + 14),
                center=True,
                bold=True,
            )
            self._draw_pixel_text(
                surface,
                f"{int(node['cpu'] * 100)}%",
                10,
                C["text_muted"],
                (cx, cy + radius + 28),
                center=True,
            )

    # ─── Layer: HUD top bar ───────────────────────────────────────────────────

    def _draw_hud(self, surface: Any, now: float) -> None:
        pg = self._pygame

        # Gradient strip
        for y in range(TOP_HUD_H):
            shade = int(28 - y * 0.25)
            pg.draw.line(
                surface, (shade, shade + 8, shade + 22), (0, y), (WINDOW_WIDTH, y)
            )
        pg.draw.rect(surface, C["border"], (0, TOP_HUD_H - 4, WINDOW_WIDTH, 4))
        pg.draw.rect(surface, C["accent2"], (0, TOP_HUD_H - 2, WINDOW_WIDTH, 2))

        # CG logo block
        pg.draw.rect(surface, C["accent2"], (8, 8, 56, 56))
        pg.draw.rect(surface, C["bg"], (8, 8, 56, 56), 3)
        self._draw_pixel_text(
            surface, "CG", 22, C["bg"], (36, 12), center=True, bold=True
        )
        self._draw_pixel_text(surface, "SYS", 9, C["bg"], (36, 40), center=True)

        # Title + subtitle
        self._draw_pixel_text(
            surface, "CASCADE GUARD", 24, C["accent"], (76, 8), bold=True
        )
        self._draw_pixel_text(
            surface,
            f"MISSION CONTROL  |  {self._scenario_name.upper()}",
            11,
            C["text_muted"],
            (76, 38),
        )

        pg.draw.rect(surface, C["border"], (340, 8, 2, 56))

        # Wave counter
        self._draw_pixel_text(surface, "WAVE", 10, C["text_muted"], (354, 10))
        self._draw_pixel_text(
            surface,
            f"{self.current_step:03d}/{self.max_steps}",
            18,
            C["accent"],
            (354, 24),
            bold=True,
        )
        wave_pct = self.current_step / max(self.max_steps, 1)
        self._draw_pixel_bar(
            surface, 354, 52, 120, 10, wave_pct, C["accent2"], chunks=12
        )

        pg.draw.rect(surface, C["border"], (498, 8, 2, 56))

        # Stable streak coin counter
        coin_x, coin_y = 524, 32
        if self.stable_streak > 0:
            pg.draw.circle(surface, C["coin"], (coin_x, coin_y), 12)
            pg.draw.circle(surface, (200, 160, 20), (coin_x, coin_y), 12, 2)
            self._draw_pixel_text(
                surface, "$", 14, C["bg"], (coin_x, coin_y - 7), center=True, bold=True
            )
            self._draw_pixel_text(
                surface,
                f"x{self.stable_streak}",
                18,
                C["coin"],
                (coin_x + 20, coin_y - 9),
                bold=True,
            )
        else:
            self._draw_pixel_text(
                surface, "x0", 18, C["text_dark"], (coin_x + 8, coin_y - 9)
            )
        self._draw_pixel_text(surface, "STREAK", 9, C["text_muted"], (coin_x - 6, 56))

        # AI pilot badge
        if self._ai_mode:
            pg.draw.rect(surface, C["bg_card"], (594, 10, 110, 52))
            self._draw_pixel_border(
                surface, 594, 10, 110, 52, C["success"], C["bg"], thick=2
            )
            self._draw_pixel_text(
                surface, "AUTO", 12, C["success"], (649, 18), center=True
            )
            self._draw_pixel_text(
                surface, "PILOT", 12, C["success"], (649, 34), center=True
            )
            if self._blink_state:
                pg.draw.rect(surface, C["success"], (602, 56, 6, 6))

        # Last action chip
        act_col = C["info"] if self.last_action != "Observe" else C["text_dark"]
        act_x = 720
        pg.draw.rect(surface, C["bg_card"], (act_x, 8, 210, 56))
        self._draw_pixel_border(surface, act_x, 8, 210, 56, act_col, C["bg"], thick=2)
        self._draw_pixel_text(surface, "ACTION:", 10, C["text_muted"], (act_x + 10, 14))
        self._draw_pixel_text(
            surface, self.last_action.upper(), 16, act_col, (act_x + 10, 30), bold=True
        )
        hint = self.task_hint[:34] + ("…" if len(self.task_hint) > 34 else "")
        self._draw_pixel_text(surface, hint, 9, C["text_dark"], (act_x + 10, 52))

        # Score block
        score_x = WINDOW_WIDTH - 272
        pg.draw.rect(surface, C["bg_card"], (score_x, 6, 264, 60))
        self._draw_pixel_border(
            surface, score_x, 6, 264, 60, C["accent2"], C["bg"], thick=3
        )
        self._draw_pixel_text(surface, "SCORE", 11, C["text_muted"], (score_x + 14, 12))
        self._draw_pixel_text(
            surface,
            f"{int(self.mission_score):,}",
            30,
            C["accent2"],
            (score_x + 14, 26),
            bold=True,
        )
        self._draw_pixel_text(
            surface, f"TASK {self.task_score:.2f}", 11, C["accent"], (score_x + 162, 12)
        )
        self._draw_pixel_text(
            surface, f"STEP {self.current_step}", 11, C["text_dim"], (score_x + 162, 30)
        )

    # ─── Layer: event feed (bottom bar) ──────────────────────────────────────

    def _draw_event_feed(self, surface: Any, now: float) -> None:
        pg = self._pygame
        bx, by = GRAPH_LEFT, WINDOW_HEIGHT - BOTTOM_H
        bw, bh = GRAPH_WIDTH, BOTTOM_H - 4

        pg.draw.rect(surface, C["bg_panel"], (bx, by, bw, bh))
        self._draw_pixel_border(
            surface, bx, by, bw, bh, C["border"], C["border_outer"], thick=3
        )
        self._draw_pixel_text(
            surface, "MISSION LOG:", 12, C["accent2"], (bx + 14, by + 10)
        )

        if not self.events:
            self._draw_pixel_text(
                surface,
                "_ ALL SYSTEMS NOMINAL _",
                12,
                C["text_dark"],
                (bx + bw // 2, by + bh // 2),
                center=True,
            )
        else:
            n = min(4, len(self.events))
            chip_w = (bw - 30) // n - 4
            sev_icons = {"success": "+", "info": ">", "warning": "!", "critical": "X"}
            sev_colors = {
                "success": C["success"],
                "info": C["accent"],
                "warning": C["warning"],
                "critical": C["critical"],
            }
            for i, ev in enumerate(self.events[:n]):
                cx2 = bx + 14 + i * (chip_w + 6)
                cy2 = by + 28
                sev_col = sev_colors.get(ev.severity, C["text_muted"])
                icon = sev_icons.get(ev.severity, "?")
                if ev.severity == "critical" and not self._blink_state:
                    pg.draw.rect(surface, C["bg_card"], (cx2, cy2, chip_w, 74))
                    self._draw_pixel_border(
                        surface, cx2, cy2, chip_w, 74, sev_col, C["bg"], thick=2
                    )
                    continue
                pg.draw.rect(surface, C["bg_card"], (cx2, cy2, chip_w, 74))
                self._draw_pixel_border(
                    surface, cx2, cy2, chip_w, 74, sev_col, C["bg"], thick=2
                )
                self._draw_pixel_text(
                    surface, f"[{icon}]", 13, sev_col, (cx2 + 8, cy2 + 8), bold=True
                )
                self._draw_pixel_text(
                    surface,
                    ev.title[:22] + ("…" if len(ev.title) > 22 else ""),
                    11,
                    C["text"],
                    (cx2 + 8, cy2 + 26),
                    bold=True,
                )
                self._draw_pixel_text(
                    surface,
                    ev.detail[:34] + ("…" if len(ev.detail) > 34 else ""),
                    9,
                    C["text_muted"],
                    (cx2 + 8, cy2 + 44),
                )

        # Badge strip (right side)
        if self.badges:
            badge_txt = "  *  ".join(self.badges[-4:])
            self._draw_pixel_text(
                surface, badge_txt, 10, C["accent2"], (bx + bw - 14, by + 10), bold=True
            )

        # Key hint bar
        keys = "SPC play  1 reroute  2 restart  3 scale  4 throttle  5 balance  TAB focus  R reset  ESC quit"
        self._draw_pixel_text(
            surface, keys, 9, C["text_dark"], (bx + bw // 2, by + bh - 12), center=True
        )

    # ─── Layer: right-side node detail panel ─────────────────────────────────

    def _draw_node_detail(self, surface: Any, now: float) -> None:
        if not self.node_specs:
            return
        pg = self._pygame
        px, py = DETAIL_LEFT, DETAIL_TOP
        pw, ph = DETAIL_WIDTH, DETAIL_HEIGHT

        pg.draw.rect(surface, C["bg_panel"], (px, py, pw, ph))
        self._draw_pixel_border(
            surface, px, py, pw, ph, C["border"], C["border_outer"], thick=3
        )
        self._draw_pixel_text(
            surface, "INSPECTOR", 13, C["accent2"], (px + pw // 2, py + 12), center=True
        )
        pg.draw.rect(surface, C["border"], (px + 6, py + 28, pw - 12, 2))

        spec = self.node_specs[self.selected_index]
        node = (
            self.nodes[self.selected_index]
            if self.selected_index < len(self.nodes)
            else {}
        )
        status = node.get("status", "healthy")
        col = self._node_color(status)

        # Node header card
        hx, hy, hw, hh = px + 6, py + 36, pw - 12, 84
        pg.draw.rect(surface, C["bg_card"], (hx, hy, hw, hh))
        self._draw_pixel_border(surface, hx, hy, hw, hh, col, C["bg"], thick=3)

        # Status dot (blinks on bad status)
        show_dot = not (status in ("failed", "stressed") and not self._blink_state)
        if show_dot:
            pg.draw.rect(surface, col, (hx + 4, hy + 4, 8, 8))

        kind = NODE_KINDS.get(spec.node_id, "compute")
        self._draw_pixel_text(
            surface, spec.label, 15, C["text"], (hx + 20, hy + 4), bold=True
        )
        self._draw_pixel_text(surface, kind.upper(), 10, col, (hx + 20, hy + 24))
        self._draw_pixel_text(
            surface,
            "TEMPORARY" if spec.temporary else "BASE NODE",
            9,
            C["text_muted"],
            (hx + 20, hy + 38),
        )

        # Status badge
        stext = status.upper()
        sbox_w = len(stext) * 7 + 14
        pg.draw.rect(surface, col, (hx + 16, hy + 56, sbox_w, 20))
        pg.draw.rect(surface, C["bg"], (hx + 16, hy + 56, sbox_w, 20), 2)
        self._draw_pixel_text(
            surface,
            stext,
            10,
            C["bg"],
            (hx + 16 + sbox_w // 2, hy + 60),
            center=True,
            bold=True,
        )

        # Latency in top-right of header
        lat = node.get("latency", 0.0)
        lat_col = (
            C["healthy"] if lat < 60 else C["stressed"] if lat < 120 else C["critical"]
        )
        self._draw_pixel_text(
            surface, f"{int(lat)}ms", 15, lat_col, (hx + hw - 54, hy + 58), bold=True
        )

        # Metric bars
        bar_top = hy + hh + 12
        metrics = [
            ("CPU LOAD", node.get("cpu", 0.0), col),
            (
                "QUEUE",
                min(node.get("queue", 0.0) / 30.0, 1.0),
                C["stressed"] if node.get("queue", 0) > 18 else col,
            ),
            ("TRAFFIC", node.get("traffic", 0.0), col),
            (
                "FAIL RISK",
                node.get("risk", 0.0),
                C["critical"]
                if node.get("risk", 0) > 0.7
                else C["stressed"]
                if node.get("risk", 0) > 0.4
                else col,
            ),
        ]
        for j, (lbl, val, bc) in enumerate(metrics):
            my = bar_top + j * 44
            self._draw_pixel_text(surface, lbl, 10, C["text_muted"], (px + 12, my))
            self._draw_pixel_text(
                surface, f"{int(val * 100)}%", 14, bc, (px + pw - 44, my), bold=True
            )
            self._draw_pixel_bar(
                surface,
                px + 10,
                my + 16,
                pw - 22,
                10,
                float(np.clip(val, 0.0, 1.0)),
                bc,
                chunks=8,
            )

        # Operator note card
        note_y = bar_top + len(metrics) * 44 + 8
        remaining = py + ph - note_y - 10
        if remaining > 44:
            pg.draw.rect(surface, C["bg_card"], (px + 10, note_y, pw - 22, remaining))
            self._draw_pixel_border(
                surface,
                px + 10,
                note_y,
                pw - 22,
                remaining,
                C["accent"],
                C["bg"],
                thick=2,
            )
            self._draw_pixel_text(
                surface, "NOTE:", 10, C["accent"], (px + 18, note_y + 8)
            )
            note = node.get("note", "")
            words = note.split()
            lines: list[str] = []
            line = ""
            for word in words:
                if len(line) + len(word) + 1 > 30:
                    lines.append(line.strip())
                    line = word + " "
                else:
                    line += word + " "
            if line.strip():
                lines.append(line.strip())
            for k, ln in enumerate(lines[:3]):
                self._draw_pixel_text(
                    surface, ln, 9, C["text_muted"], (px + 18, note_y + 24 + k * 16)
                )

        # Mini mission feed below note card
        feed_y = note_y + remaining + 6
        if feed_y + 40 < py + ph:
            pg.draw.rect(surface, C["border"], (px + 6, feed_y, pw - 12, 2))
            feed_y += 6
            self._draw_pixel_text(
                surface, "MISSION FEED", 11, C["accent"], (px + 12, feed_y)
            )
            feed_y += 18
            for ev in self.events[:4]:
                if feed_y + 20 >= py + ph:
                    break
                ev_col = {
                    "critical": C["critical"],
                    "warning": C["stressed"],
                    "success": C["success"],
                }.get(ev.severity, C["text_dim"])
                self._draw_pixel_text(
                    surface, ev.title[:26], 10, ev_col, (px + 12, feed_y), bold=True
                )
                feed_y += 16

    # ─── Layer: demo callout ──────────────────────────────────────────────────

    def _draw_callout(self, surface: Any) -> None:
        pg = self._pygame
        callout = self.demo_callout
        if not callout:
            return
        lines = [str(l) for l in callout.get("lines", [])[:3]]
        title = str(callout.get("title", ""))
        bw_, bh_ = 520, 70 + len(lines) * 22
        bx_ = GRAPH_LEFT + GRAPH_WIDTH // 2 - bw_ // 2
        by_ = GRAPH_BOTTOM - bh_ - 14
        srf = pg.Surface((bw_, bh_), pg.SRCALPHA)
        pg.draw.rect(srf, (*C["bg_card"], 240), (0, 0, bw_, bh_))
        surface.blit(srf, (bx_, by_))
        self._draw_pixel_border(surface, bx_, by_, bw_, bh_, C["border_hi"], C["bg"])
        self._draw_pixel_text(
            surface, title, 16, C["text"], (bx_ + 16, by_ + 12), bold=True
        )
        for k, line in enumerate(lines):
            self._draw_pixel_text(
                surface, line, 11, C["text_muted"], (bx_ + 16, by_ + 36 + k * 22)
            )

    # ─── Layer: CRT scanlines ─────────────────────────────────────────────────

    def _draw_scanlines(self, surface: Any) -> None:
        if self.reduced_motion:
            return
        pg = self._pygame
        scan = pg.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pg.SRCALPHA)
        for y in range(0, WINDOW_HEIGHT, 4):
            pg.draw.line(scan, (0, 0, 0, 22), (0, y), (WINDOW_WIDTH, y), 1)
        surface.blit(scan, (0, 0))

    # ─── Pixel drawing utilities ──────────────────────────────────────────────

    def _node_color(self, status: str) -> tuple:
        return {
            "healthy": C["healthy"],
            "warning": C["stressed"],
            "stressed": C["overloaded"],
            "failed": C["failed"],
        }.get(status, C["text_dim"])

    def _font(self, size: int, bold: bool = False) -> Any:
        key = (size, bold)
        if key not in self._font_cache:
            self._font_cache[key] = self._pygame.font.SysFont(
                "consolas", size, bold=bold
            )
        return self._font_cache[key]

    def _draw_pixel_text(
        self,
        surface: Any,
        text: str,
        size: int,
        color: tuple,
        position: tuple,
        *,
        center: bool = False,
        bold: bool = False,
    ) -> None:
        # Handle RGBA colours (alpha channel via temp surface)
        if len(color) == 4:
            r, g, b, a = color
            if a == 0:
                return
            tmp = self._pygame.Surface(surface.get_size(), self._pygame.SRCALPHA)
            self._draw_pixel_text(
                tmp, text, size, (r, g, b), position, center=center, bold=bold
            )
            tmp.set_alpha(a)
            surface.blit(tmp, (0, 0))
            return
        rendered = self._font(size, bold).render(str(text), True, color[:3])
        rect = rendered.get_rect()
        if center:
            rect.center = position
        else:
            rect.topleft = position
        surface.blit(rendered, rect)

    def _draw_pixel_border(
        self,
        surface: Any,
        x: int,
        y: int,
        w: int,
        h: int,
        col_hi: tuple,
        col_lo: tuple,
        thick: int = 2,
    ) -> None:
        pg = self._pygame
        pg.draw.rect(surface, col_lo, (x, y, w, h), thick)
        pg.draw.line(surface, col_hi, (x, y), (x + w - 1, y), thick)
        pg.draw.line(surface, col_hi, (x, y), (x, y + h - 1), thick)

    def _draw_pixel_bar(
        self,
        surface: Any,
        x: int,
        y: int,
        w: int,
        h: int,
        value: float,
        col: tuple,
        chunks: int = 8,
    ) -> None:
        pg = self._pygame
        frac = float(np.clip(value, 0.0, 1.0))
        pg.draw.rect(surface, C["bg"], (x, y, w, h))
        pg.draw.rect(surface, C["border_outer"], (x, y, w, h), 1)
        if chunks <= 0:
            return
        gap = 2
        chunk_w = max(2, (w - gap * (chunks - 1)) // chunks)
        filled = int(frac * chunks)
        for i in range(chunks):
            cx = x + i * (chunk_w + gap)
            if i < filled:
                pg.draw.rect(surface, col, (cx, y + 1, chunk_w, h - 2))
                # Highlight row
                pg.draw.rect(
                    surface,
                    tuple(min(255, c + 60) for c in col),
                    (cx, y + 1, chunk_w, 2),
                )
            else:
                pg.draw.rect(surface, C["bg_card"], (cx, y + 1, chunk_w, h - 2))

    def _draw_pixel_sparkline(
        self,
        surface: Any,
        data: Any,
        x: int,
        y: int,
        w: int,
        h: int,
        col: tuple,
        invert: bool = False,
    ) -> None:
        pg = self._pygame
        vals = list(data)
        if len(vals) < 2:
            return
        mn, mx = min(vals), max(vals)
        rng = max(mx - mn, 1e-6)
        pts = []
        for i, v in enumerate(vals):
            nx = x + int(i / (len(vals) - 1) * w)
            frac = (v - mn) / rng
            if invert:
                frac = 1.0 - frac
            pts.append((nx, y + h - int(frac * h)))
        if len(pts) >= 2:
            pg.draw.lines(surface, col, False, pts, 1)
            pg.draw.rect(surface, col, (pts[-1][0] - 1, pts[-1][1] - 1, 3, 3))

    def _draw_pixel_glow_line(
        self,
        surface: Any,
        col: tuple,
        start: tuple,
        end: tuple,
        width: int,
        alpha: int = 60,
    ) -> None:
        pg = self._pygame
        tmp = pg.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pg.SRCALPHA)
        pg.draw.line(tmp, (*col, alpha), start, end, width)
        surface.blit(tmp, (0, 0))

    def _draw_dashed_line(
        self,
        surface: Any,
        color: tuple,
        start: tuple,
        end: tuple,
        *,
        width: int,
        dash_length: int,
    ) -> None:
        pg = self._pygame
        dx, dy = end[0] - start[0], end[1] - start[1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            return
        steps = int(dist / dash_length)
        for k in range(0, steps, 2):
            s = k / steps
            e = min((k + 1) / steps, 1.0)
            pg.draw.line(
                surface,
                color,
                (int(start[0] + dx * s), int(start[1] + dy * s)),
                (int(start[0] + dx * e), int(start[1] + dy * e)),
                width,
            )

    # ─── Node sprites ─────────────────────────────────────────────────────────

    def _draw_pixel_star(
        self, surface: Any, cx: int, cy: int, radius: int, col: tuple
    ) -> None:
        """Control node: 5-point pixel star."""
        pg = self._pygame
        pts = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            r = radius if i % 2 == 0 else radius // 2
            pts.append((int(cx + r * math.cos(angle)), int(cy - r * math.sin(angle))))
        pg.draw.polygon(surface, C["bg_card"], pts)
        pg.draw.polygon(surface, col, pts, 3)
        # Semi-transparent inner fill
        off = int(radius * 1.5)
        tmp = pg.Surface((off * 2, off * 2), pg.SRCALPHA)
        inner_pts = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            r = (radius // 2) if i % 2 == 0 else (radius // 4)
            inner_pts.append(
                (int(off + r * math.cos(angle)), int(off - r * math.sin(angle)))
            )
        pg.draw.polygon(tmp, (*col, 80), inner_pts)
        surface.blit(tmp, (cx - off, cy - off))

    def _draw_pixel_shield(
        self, surface: Any, cx: int, cy: int, radius: int, col: tuple
    ) -> None:
        """Gateway node: hexagonal shield."""
        pg = self._pygame
        pts = [
            (cx, cy - radius),
            (cx + radius, cy - radius // 2),
            (cx + radius, cy + radius // 2),
            (cx, cy + radius),
            (cx - radius, cy + radius // 2),
            (cx - radius, cy - radius // 2),
        ]
        pg.draw.polygon(surface, C["bg_card"], pts)
        pg.draw.polygon(surface, col, pts, 3)
        # Inner diamond fill
        off = int(radius * 1.5)
        tmp = pg.Surface((off * 2, off * 2), pg.SRCALPHA)
        half = radius // 2
        chevron = [
            (off, off - half),
            (off + half, off),
            (off, off + half),
            (off - half, off),
        ]
        pg.draw.polygon(tmp, (*col, 70), chevron)
        surface.blit(tmp, (cx - off, cy - off))

    def _draw_pixel_chest(
        self, surface: Any, cx: int, cy: int, radius: int, col: tuple
    ) -> None:
        """Storage node: treasure chest."""
        pg = self._pygame
        w = int(radius * 2.2)
        h = int(radius * 1.8)
        lx, ly = cx - w // 2, cy - h // 2
        pg.draw.rect(surface, C["bg_card"], (lx, ly, w, h))
        self._draw_pixel_border(surface, lx, ly, w, h, col, C["bg"], thick=3)
        # Lid divider
        pg.draw.rect(surface, col, (lx + 2, cy - 3, w - 4, 3))
        # Lock
        lkx, lky = cx - 5, cy - 6
        pg.draw.rect(surface, col, (lkx, lky, 10, 10))
        pg.draw.rect(surface, C["bg_card"], (lkx + 2, lky + 2, 6, 6))
        # Corner rivets
        for rx, ry in (
            (lx + 4, ly + 4),
            (lx + w - 8, ly + 4),
            (lx + 4, ly + h - 8),
            (lx + w - 8, ly + h - 8),
        ):
            pg.draw.rect(surface, col, (rx, ry, 4, 4))

    def _draw_pixel_cpu(
        self, surface: Any, cx: int, cy: int, radius: int, col: tuple, now: float
    ) -> None:
        """Compute / temp node: CPU chip."""
        pg = self._pygame
        r = radius
        # Chip body
        pg.draw.rect(surface, C["bg_card"], (cx - r, cy - r, r * 2, r * 2))
        self._draw_pixel_border(
            surface, cx - r, cy - r, r * 2, r * 2, col, C["bg_card"], thick=2
        )
        # Pins on left and right
        for offset in (-r // 2, 0, r // 2):
            pg.draw.rect(surface, col, (cx - r - 5, cy + offset - 2, 5, 4))
            pg.draw.rect(surface, col, (cx + r, cy + offset - 2, 5, 4))
        # Inner circuit grid
        inner = pg.Surface((r * 2 - 8, r * 2 - 8), pg.SRCALPHA)
        for gi in range(0, r * 2 - 8, 6):
            pg.draw.line(inner, (*col, 50), (gi, 0), (gi, r * 2 - 8))
            pg.draw.line(inner, (*col, 50), (0, gi), (r * 2 - 8, gi))
        surface.blit(inner, (cx - r + 4, cy - r + 4))
        # Pulsing core
        if not self.reduced_motion:
            pulse = int(abs(math.sin(now * 3 + cx)) * 80)
            core_col = tuple(min(255, c + pulse) for c in col)
        else:
            core_col = col
        core_r = max(3, r // 3)
        pg.draw.rect(
            surface, core_col, (cx - core_r, cy - core_r, core_r * 2, core_r * 2)
        )
