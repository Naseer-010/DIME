"""
Pydantic v2 typed Action and Observation models for the
Distributed Infrastructure Management Environment.
"""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, model_validator


class InfraAction(Action):
    """
    Action the LLM agent can take to manage the distributed system.

    Supported action types:
        - restart_node: Bring a failed node back online (2-step delay, 5-step cooldown).
        - reroute_traffic: Shift a fraction of load between two nodes.
        - scale_up: Add a temporary capacity node for 10 steps (costs 1 cloud budget unit).
        - throttle: Reduce incoming request acceptance rate.
        - query_logs: Investigate a node with telemetry dropout (partial observability).
        - no_op: Take no action (passive observation step).

    Optionally, ``raw_command`` can be set to a kubectl/AWS CLI string
    which takes priority and is parsed into structured fields automatically.
    """

    action_type: Literal[
        "restart_node", "reroute_traffic", "scale_up", "throttle", "query_logs", "no_op"
    ] = Field(description="The management action to perform.")

    target: Optional[int] = Field(
        default=None,
        description="Target node index (used by restart_node, query_logs).",
    )
    from_node: Optional[int] = Field(
        default=None,
        description="Source node index (used by reroute_traffic).",
    )
    to_node: Optional[int] = Field(
        default=None,
        description="Destination node index (used by reroute_traffic).",
    )
    rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Throttle rate in [0, 1] (used by throttle). 1.0 = accept all, 0.0 = reject all.",
    )
    raw_command: Optional[str] = Field(
        default=None,
        description=(
            "Raw kubectl/AWS CLI command string. When set, the environment "
            "parses this into a structured action automatically. Takes "
            "priority over other fields."
        ),
    )

    @model_validator(mode="after")
    def validate_action_params(self) -> "InfraAction":
        # Skip validation when raw_command is provided — it gets parsed later
        if self.raw_command:
            return self
        if self.action_type == "restart_node" and self.target is None:
            raise ValueError("restart_node requires 'target' node index.")
        if self.action_type == "reroute_traffic":
            if self.from_node is None or self.to_node is None:
                raise ValueError(
                    "reroute_traffic requires both 'from_node' and 'to_node'."
                )
        if self.action_type == "throttle" and self.rate is None:
            raise ValueError("throttle requires 'rate' parameter.")
        if self.action_type == "query_logs" and self.target is None:
            raise ValueError("query_logs requires 'target' node index.")
        return self


class InfraObservation(Observation):
    """
    Observation returned to the LLM agent at each step.

    Contains the observable state of the distributed system plus
    anti-hacking and partial-observability metadata.
    """

    cpu_loads: List[float] = Field(
        description=(
            "CPU utilization [0.0, 1.0] for each node. "
            "A value of -1.0 indicates telemetry dropout (timeout)."
        )
    )
    queue_lengths: List[int] = Field(
        description="Number of pending requests per node. -1 indicates telemetry dropout."
    )
    failed_nodes: List[int] = Field(
        description="Indices of nodes currently in failed state."
    )
    latency_ms: float = Field(
        description="Rolling average end-to-end latency in milliseconds."
    )
    request_rate: float = Field(
        description="Incoming requests per second into the system."
    )
    mem_utilizations: List[float] = Field(
        default_factory=list,
        description="Memory utilization [0.0, 1.0] per node (same ordering as cpu_loads).",
    )
    io_wait: float = Field(
        default=0.0,
        description="Database disk I/O wait / saturation proxy in [0.0, 1.0].",
    )
    p99_latency: float = Field(
        default=0.0,
        description="P99 tail latency in milliseconds.",
    )
    error_budget: float = Field(
        default=100.0,
        description="Remaining error budget token bucket for throttling actions.",
    )

    # --- ML-friendly normalized features ---
    request_rate_norm: float = Field(
        default=0.0,
        description="request_rate normalized to [0,1] (divide by 5000.0, clipped).",
    )
    p99_latency_norm: float = Field(
        default=0.0,
        description="p99_latency normalized to [0,1] (divide by 1000.0, clipped).",
    )
    step: int = Field(description="Current step within the episode.")
    task_hint: str = Field(
        description="Natural language description of the current task objective."
    )
    task_score: float = Field(default=0.01, description="Current grader score")

    # --- Partial observability ---
    telemetry_status: Dict[int, str] = Field(
        default_factory=dict,
        description="Per-node telemetry status: 'ok' or 'timeout'.",
    )

    # --- Anti-hacking sandbox ---
    action_errors: List[str] = Field(
        default_factory=list,
        description=(
            "Errors from the last action (e.g. InsufficientFunds, "
            "CooldownActive, ParseError)."
        ),
    )
    cloud_budget: int = Field(
        default=10,
        description="Remaining cloud budget units for scale_up.",
    )

    # --- Prometheus-style telemetry ---
    prometheus_metrics: List[Dict] = Field(
        default_factory=list,
        description=(
            "Prometheus-style structured metrics. Each entry is a dict with "
            "'metric', 'labels', 'value', 'timestamp' keys."
        ),
    )


class InfraState(State):
    """
    Internal environment state extending the base OpenEnv State.
    """

    task_id: Optional[str] = Field(default=None, description="Current task identifier.")
    task_score: float = Field(
        default=0.01, description="Current task grader score in (0.0, 1.0) strictly."
    )
