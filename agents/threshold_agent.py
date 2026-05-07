"""Classical threshold automation baseline for DIME."""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from benchmark.utils import observation_to_dict
from server.models import InfraAction


class ThresholdAgent(BaseAgent):
    """Reactive autoscaler-style baseline."""

    def act(self, observation: Any) -> InfraAction:
        obs = observation_to_dict(observation)
        cpu_loads = [float(v) for v in obs.get("cpu_loads", []) if float(v) >= 0.0]
        avg_cpu = sum(cpu_loads) / len(cpu_loads) if cpu_loads else 0.0
        latency = float(obs.get("latency_ms", 0.0) or 0.0)
        failed_nodes = list(obs.get("failed_nodes", []) or [])

        if avg_cpu > 0.80:
            return InfraAction(action_type="scale_up")
        if latency > 100.0:
            return InfraAction(action_type="throttle", rate=0.7)
        if failed_nodes:
            return InfraAction(action_type="restart_node", target=int(failed_nodes[0]))
        return InfraAction(action_type="no_op")
