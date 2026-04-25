"""
Real-world CLI command parser for DIME.

Converts kubectl / AWS CLI command strings into ``InfraAction`` objects
using regex pattern matching.  This proves the environment can train
models on production-grade syntax rather than abstract JSON schemas.
"""

from __future__ import annotations

import re

from server.models import InfraAction


class CommandParseError(Exception):
    """Raised when a raw CLI command cannot be parsed."""


# ---------------------------------------------------------------------------
# Pattern table: (compiled regex, handler function)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern, callable]] = []


def _register(pattern: str):
    """Decorator to register a regex → handler mapping."""
    compiled = re.compile(pattern, re.IGNORECASE)

    def decorator(fn):
        _PATTERNS.append((compiled, fn))
        return fn

    return decorator


# ---- kubectl scale → scale_up ----
@_register(r"kubectl\s+scale\s+deployment\s+\S+\s+--replicas[=\s]+(\d+)")
def _handle_kubectl_scale(match: re.Match) -> InfraAction:
    return InfraAction(action_type="scale_up")


# ---- aws autoscaling → scale_up ----
@_register(
    r"aws\s+autoscaling\s+set-desired-capacity\s+.*--desired-capacity[=\s]+(\d+)"
)
def _handle_aws_scale(match: re.Match) -> InfraAction:
    return InfraAction(action_type="scale_up")


# ---- kubectl delete pod + apply restart → restart_node ----
@_register(r"kubectl\s+(?:delete\s+pod|rollout\s+restart)\s+.*node[_-]?(\d+)")
def _handle_kubectl_restart(match: re.Match) -> InfraAction:
    target = int(match.group(1))
    return InfraAction(action_type="restart_node", target=target)


# ---- kubectl restart via apply -f restart.yaml ----
@_register(r"kubectl\s+apply\s+-f\s+restart.*node[_-]?(\d+)")
def _handle_kubectl_apply_restart(match: re.Match) -> InfraAction:
    target = int(match.group(1))
    return InfraAction(action_type="restart_node", target=target)


# ---- istio/envoy traffic shift → reroute_traffic ----
@_register(
    r"kubectl\s+exec.*(?:istio|envoy).*traffic\s+shift\s+--from[=\s]+(\d+)\s+--to[=\s]+(\d+)"
)
def _handle_traffic_shift(match: re.Match) -> InfraAction:
    return InfraAction(
        action_type="reroute_traffic",
        from_node=int(match.group(1)),
        to_node=int(match.group(2)),
    )


# ---- kubectl logs → query_logs ----
@_register(r"kubectl\s+logs\s+.*node[_-]?(\d+)")
def _handle_kubectl_logs(match: re.Match) -> InfraAction:
    target = int(match.group(1))
    return InfraAction(action_type="query_logs", target=target)


# ---- kubectl throttle / rate-limit ingress ----
@_register(r"kubectl\s+(?:throttle|annotate)\s+ingress\s+.*--rate[=\s]+([\d.]+)")
def _handle_throttle(match: re.Match) -> InfraAction:
    rate = float(match.group(1))
    rate = max(0.0, min(1.0, rate))
    return InfraAction(action_type="throttle", rate=rate)


# ---- explicit no_op / observe ----
@_register(r"(?:no[_-]?op|observe|noop|kubectl\s+get\s+pods)")
def _handle_noop(match: re.Match) -> InfraAction:
    return InfraAction(action_type="no_op")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_command(raw: str) -> InfraAction:
    """
    Parse a raw CLI command string into an ``InfraAction``.

    Iterates through registered patterns; first match wins.

    Raises
    ------
    CommandParseError
        If no pattern matches the input string.
    """
    raw = raw.strip()
    for pattern, handler in _PATTERNS:
        m = pattern.search(raw)
        if m:
            return handler(m)
    raise CommandParseError(
        f"Unrecognised command: '{raw[:120]}'. "
        "Expected kubectl or AWS CLI syntax. Examples:\n"
        "  kubectl scale deployment frontend --replicas=10\n"
        "  kubectl delete pod node-3\n"
        "  kubectl logs node-2\n"
        "  kubectl throttle ingress --rate=0.8\n"
        "  kubectl exec -it istio-proxy -- traffic shift --from=1 --to=3"
    )
