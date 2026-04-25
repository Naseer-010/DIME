"""
Trace replay loader for DIME.

Loads pre-processed Alibaba-style cluster trace CSV files and provides
step-by-step replay of real-world traffic patterns, CPU baselines,
and latency injections.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TraceStep:
    """One step of trace data."""

    request_rate: float = 100.0
    latency_injection: float = 0.0
    node_cpu: Dict[int, float] = field(default_factory=dict)
    node_mem: Dict[int, float] = field(default_factory=dict)


class TraceReplay:
    """
    Load and replay a trace CSV file step-by-step.

    The CSV must have columns:
        step, node_0_cpu, node_0_mem, ..., request_rate, latency_injection

    Wraps around if the episode exceeds the trace length.
    """

    def __init__(self, csv_path: str) -> None:
        self._steps: List[TraceStep] = []
        self._load(csv_path)

    def _load(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Trace file not found: {csv_path}")

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = TraceStep(
                    request_rate=float(row["request_rate"]),
                    latency_injection=float(row["latency_injection"]),
                )
                # Parse per-node metrics
                for key, val in row.items():
                    if key.startswith("node_") and key.endswith("_cpu"):
                        idx = int(key.split("_")[1])
                        ts.node_cpu[idx] = float(val)
                    elif key.startswith("node_") and key.endswith("_mem"):
                        idx = int(key.split("_")[1])
                        ts.node_mem[idx] = float(val)
                self._steps.append(ts)

    def __len__(self) -> int:
        return len(self._steps)

    def get_step(self, step: int) -> TraceStep:
        """Get trace data for a given step. Wraps around."""
        if not self._steps:
            return TraceStep()
        return self._steps[step % len(self._steps)]


# ---------------------------------------------------------------------------
# Default trace path
# ---------------------------------------------------------------------------

_DEFAULT_TRACE = os.path.join(
    os.path.dirname(__file__), "traces", "alibaba_v2021_8node_500steps.csv"
)


def load_default_trace() -> Optional[TraceReplay]:
    """Load the bundled Alibaba trace, or None if not generated yet."""
    if os.path.exists(_DEFAULT_TRACE):
        return TraceReplay(_DEFAULT_TRACE)
    return None
