"""Canonical symbolic DIME baseline using the existing triage tree."""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from benchmark.utils import observation_to_dict
from server.command_parser import CommandParseError, parse_command
from server.models import InfraAction


def expected_triage_command(observation: Any) -> str:
    """Return the kubectl command mandated by the DIME triage tree."""
    obs = observation_to_dict(observation)
    cpu = obs.get("cpu_loads", [0.3] * 8)
    mem = obs.get("mem_utilizations", [0.2] * 8)
    failed = set(obs.get("failed_nodes", []) or [])
    io_wait = float(obs.get("io_wait", 0.0) or 0.0)
    p99 = float(obs.get("p99_latency", 0.0) or 0.0)
    request_rate = float(obs.get("request_rate", 100.0) or 100.0)
    error_budget = float(obs.get("error_budget", 100.0) or 100.0)

    for idx, memory in enumerate(mem):
        if float(memory) > 0.92:
            return f"kubectl delete pod node-{idx}"

    if 0 in failed:
        return "kubectl delete pod node-0"

    if io_wait > 0.80:
        return "kubectl throttle ingress --rate=0.5"

    workers = [(idx, float(load)) for idx, load in enumerate(cpu[1:], 1) if float(load) >= 0.0]
    if workers:
        avg_worker_cpu = sum(load for _, load in workers) / len(workers)
        for idx, load in workers:
            if load > 0.90 and avg_worker_cpu < 0.60:
                candidates = [candidate for candidate, _ in workers if candidate != idx and candidate not in failed]
                if candidates:
                    dst = min(candidates, key=lambda node_idx: float(cpu[node_idx]))
                    return f"kubectl exec -it istio-proxy -- traffic shift --from={idx} --to={dst}"

    if p99 > 100.0 and request_rate > 150.0:
        return "kubectl throttle ingress --rate=0.4"

    for idx, load in workers:
        if 0.0 <= load < 0.10 and p99 > 100.0:
            dst = next(
                (candidate for candidate, candidate_load in workers if candidate_load > 0.2 and candidate not in failed and candidate != idx),
                None,
            )
            if dst is not None:
                return f"kubectl exec -it istio-proxy -- traffic shift --from={idx} --to={dst}"

    if len(failed) >= 2:
        return "kubectl throttle ingress --rate=0.3"

    db_cpu = float(cpu[0]) if cpu and float(cpu[0]) >= 0.0 else 0.0
    if db_cpu > 0.80:
        return "kubectl throttle ingress --rate=0.7"

    if workers and sum(load for _, load in workers) / len(workers) > 0.75 and error_budget > 20.0:
        return "kubectl scale deployment frontend --replicas=10"

    return "no_op"


class HeuristicAgent(BaseAgent):
    """Rule-based symbolic SRE baseline."""

    def act(self, observation: Any) -> InfraAction:
        command = expected_triage_command(observation)
        if command == "no_op":
            return InfraAction(action_type="no_op")
        try:
            return parse_command(command)
        except CommandParseError:
            return InfraAction(action_type="no_op")
