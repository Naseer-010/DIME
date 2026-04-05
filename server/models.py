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
        - restart_node: Bring a failed node back online (2-step delay).
        - reroute_traffic: Shift a fraction of load between two nodes.
        - scale_up: Add a temporary capacity node for 10 steps.
        - throttle: Reduce incoming request acceptance rate.
        - no_op: Take no action (passive observation step).
    """

    action_type: Literal[
        "restart_node", "reroute_traffic", "scale_up", "throttle", "no_op"
    ] = Field(description="The management action to perform.")

    target: Optional[int] = Field(
        default=None,
        description="Target node index (used by restart_node).",
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

    @model_validator(mode="after")
    def validate_action_params(self) -> "InfraAction":
        if self.action_type == "restart_node" and self.target is None:
            raise ValueError("restart_node requires 'target' node index.")
        if self.action_type == "reroute_traffic":
            if self.from_node is None or self.to_node is None:
                raise ValueError(
                    "reroute_traffic requires both 'from_node' and 'to_node'."
                )
        if self.action_type == "throttle" and self.rate is None:
            raise ValueError("throttle requires 'rate' parameter.")
        return self


class InfraObservation(Observation):
    """
    Observation returned to the LLM agent at each step.

    Contains the full observable state of the distributed system.
    """

    cpu_loads: List[float] = Field(
        description="CPU utilization [0.0, 1.0] for each node."
    )
    queue_lengths: List[int] = Field(
        description="Number of pending requests per node."
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
    step: int = Field(
        description="Current step within the episode."
    )
    task_hint: str = Field(
        description="Natural language description of the current task objective."
    )


class InfraState(State):
    """
    Internal environment state extending the base OpenEnv State.
    """

    task_id: Optional[str] = Field(
        default=None, description="Current task identifier."
    )
    task_score: float = Field(
        default=0.0, description="Current task grader score [0.0, 1.0]."
    )
