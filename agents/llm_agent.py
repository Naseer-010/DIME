"""LLM-backed agent adapter for the DIME benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.base_agent import BaseAgent
from benchmark.utils import observation_to_dict
from inference import build_safe_backend_action, llm_decide
from server.models import InfraAction


@dataclass
class LLMResearchAgent(BaseAgent):
    """
    Bridge ``inference.py`` prompting/parsing into the official benchmark agent API.

    The benchmark harness owns resets, seeds, episode loops, metrics, and artifact
    persistence. This adapter only converts observations to LLM prompts and converts
    model output back into a validated ``InfraAction``.
    """

    model_name: str
    mode: str = "endpoint"
    api_base: str = "http://localhost:11434/v1"
    api_key: str | None = "dummy_key"
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"local", "endpoint"}:
            raise ValueError("mode must be either 'local' or 'endpoint'")
        self.last_reasoning = ""
        self.last_raw_output = ""
        self.last_action: dict[str, Any] = {"action_type": "no_op"}
        self.last_backend_action: dict[str, Any] = {"action_type": "no_op"}

    @property
    def name(self) -> str:
        safe_model = self.model_name.replace("/", "_")
        return f"llm:{safe_model}:{self.mode}"

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        self.last_reasoning = ""
        self.last_raw_output = ""
        self.last_action = {"action_type": "no_op"}
        self.last_backend_action = {"action_type": "no_op"}

    def act(self, observation: Any) -> InfraAction:
        obs_dict = observation_to_dict(observation)
        action_dict, reasoning, raw_output = llm_decide(
            observation=obs_dict,
            model_name=self.model_name,
            mode=self.mode,
            api_base=self.api_base,
            api_key=self.api_key,
        )

        safe_action = self._coerce_safe_action(action_dict)
        self.last_reasoning = reasoning
        self.last_raw_output = raw_output
        self.last_action = action_dict if isinstance(action_dict, dict) else {}
        self.last_backend_action = safe_action

        if self.verbose:
            print(f"[LLMResearchAgent] action={safe_action} reasoning={reasoning[:160]}")

        try:
            return InfraAction.model_validate(safe_action)
        except Exception as exc:
            if self.verbose:
                print(f"[LLMResearchAgent] invalid action fallback: {exc}")
            return InfraAction(action_type="no_op")

    @staticmethod
    def _coerce_safe_action(action_dict: Any) -> dict[str, Any]:
        if not isinstance(action_dict, dict):
            return {"action_type": "no_op"}
        try:
            safe_action = build_safe_backend_action(action_dict)
        except Exception:
            return {"action_type": "no_op"}
        if not isinstance(safe_action, dict):
            return {"action_type": "no_op"}
        return safe_action
