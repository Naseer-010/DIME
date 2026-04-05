"""
Client for the Distributed Infrastructure Environment.

Provides typed interaction with the environment server via WebSocket.
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import InfraAction, InfraObservation, InfraState


class InfraEnv(EnvClient):
    """
    Client for the Distributed Infrastructure Management Environment.

    Example:
        >>> async with InfraEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task="traffic_spike")
        ...     obs = result.observation
        ...     result = await env.step(InfraAction(action_type="no_op"))
    """

    def _step_payload(self, action: InfraAction) -> Dict[str, Any]:
        """Convert InfraAction to JSON payload for the server."""
        return {"action": action.model_dump(exclude_none=True)}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[InfraObservation]:
        """Parse server response into StepResult with InfraObservation."""
        obs_data = payload.get("data", payload.get("observation", payload))
        # Handle nested observation key
        if "observation" in obs_data and isinstance(obs_data["observation"], dict):
            obs_raw = obs_data["observation"]
        else:
            obs_raw = obs_data

        obs = InfraObservation(**obs_raw)
        return StepResult(
            observation=obs,
            reward=obs_raw.get("reward", obs.reward),
            done=obs_raw.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> InfraState:
        """Parse state response."""
        state_data = payload.get("data", payload.get("state", payload))
        if "state" in state_data and isinstance(state_data["state"], dict):
            state_data = state_data["state"]
        return InfraState(**state_data)
