"""Canonical symbolic DIME baseline using the shared triage tree."""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from agents.triage import expected_triage_command, triage_action
from server.models import InfraAction


class HeuristicAgent(BaseAgent):
    """Rule-based symbolic SRE baseline."""

    def act(self, observation: Any) -> InfraAction:
        return triage_action(observation)


__all__ = ["HeuristicAgent", "expected_triage_command"]
