"""
Real-world CLI command parser for DIME.

Converts kubectl / AWS CLI command strings into ``InfraAction`` objects
using regex pattern matching.  This proves the environment can train
models on production-grade syntax rather than abstract JSON schemas.
"""

from __future__ import annotations

import json
import re

from server.models import InfraAction


class CommandParseError(Exception):
    """Raised when a raw CLI command cannot be parsed."""


# ---------------------------------------------------------------------------
# Pattern table: (compiled regex, handler function)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern, callable]] = []
_REASONING_JSON_RE = re.compile(
    r"^\s*<reasoning>.+?</reasoning>\s*(\{.*\})\s*$",
    re.IGNORECASE | re.DOTALL,
)


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


def _extract_reasoning_json(raw: str) -> dict | None:
    """Return the JSON body after a <reasoning> block, if present and valid."""
    match = _REASONING_JSON_RE.match(raw)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise CommandParseError(f"Invalid JSON after <reasoning>: {exc}") from exc
    if not isinstance(payload, dict):
        raise CommandParseError("JSON command body must be an object.")
    return payload


def has_reasoning_json_format(raw: str) -> bool:
    """Whether raw output uses the required <reasoning> XML + valid JSON shape."""
    try:
        return _extract_reasoning_json(raw.strip()) is not None
    except CommandParseError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_command(raw: str) -> InfraAction:
    """
    Parse a raw CLI command string into an ``InfraAction``.

    Accepts the legacy plain kubectl/AWS command format, and also the CoT
    format:

        <reasoning>...</reasoning>
        {"command": "kubectl ..."}

    The JSON body may use ``command``/``raw_command`` for CLI syntax or a full
    structured ``InfraAction`` object.

    Raises
    ------
    CommandParseError
        If no pattern matches the input string.
    """
    raw = raw.strip()
    payload = _extract_reasoning_json(raw)
    if payload is not None:
        command = payload.get("command", payload.get("raw_command"))
        if isinstance(command, str):
            raw = command.strip()
        else:
            try:
                return InfraAction.model_validate(payload)
            except Exception as exc:
                raise CommandParseError(
                    "JSON body must contain a string 'command'/'raw_command' "
                    "or a valid InfraAction object."
                ) from exc

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
