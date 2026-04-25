"""
Generate a synthetic Alibaba-style cluster trace CSV.

Produces multimodal traffic patterns mimicking the characteristics
reported in the Alibaba microservices-v2021 SoCC'21 paper:
- Diurnal cycle with morning/evening peaks
- Random micro-bursts (5-10 step duration)
- Silent "maintenance windows" with near-zero traffic
- Per-node CPU variance correlated with request rate

Output: server/traces/alibaba_v2021_8node_500steps.csv
"""

import csv
import math
import os
import random

NUM_NODES = 8
NUM_STEPS = 10000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "traces")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "alibaba_v2021_8node_500steps.csv")


def generate_trace(seed: int = 2021) -> None:
    rng = random.Random(seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pre-compute micro-burst windows
    bursts: list[tuple[int, int, float]] = []
    for _ in range(12):
        start = rng.randint(30, NUM_STEPS - 20)
        duration = rng.randint(5, 12)
        intensity = rng.uniform(2.5, 6.0)
        bursts.append((start, start + duration, intensity))

    # Pre-compute silent windows
    silents: list[tuple[int, int]] = []
    for _ in range(4):
        start = rng.randint(50, NUM_STEPS - 30)
        duration = rng.randint(8, 20)
        silents.append((start, start + duration))

    # Node personality: each node has a base CPU offset and sensitivity
    node_base_cpu = [0.15 + rng.uniform(-0.05, 0.05) for _ in range(NUM_NODES)]
    node_sensitivity = [0.4 + rng.uniform(-0.1, 0.15) for _ in range(NUM_NODES)]

    # Headers
    headers = ["step"]
    for i in range(NUM_NODES):
        headers.extend([f"node_{i}_cpu", f"node_{i}_mem"])
    headers.extend(["request_rate", "latency_injection"])

    rows = []
    for step in range(NUM_STEPS):
        t = step / NUM_STEPS

        # --- Diurnal cycle: two peaks (morning at 0.25, evening at 0.7) ---
        diurnal = (
            0.5
            + 0.3 * math.sin(2 * math.pi * t - math.pi / 2)
            + 0.15 * math.sin(4 * math.pi * t)
        )

        # --- Check for micro-bursts ---
        burst_mult = 1.0
        for b_start, b_end, b_intensity in bursts:
            if b_start <= step < b_end:
                # Bell-curve shape within burst
                mid = (b_start + b_end) / 2.0
                dist = abs(step - mid) / max(1, (b_end - b_start) / 2.0)
                burst_mult = max(burst_mult, b_intensity * math.exp(-dist * dist))

        # --- Check for silent windows ---
        is_silent = any(s_start <= step < s_end for s_start, s_end in silents)

        # --- Request rate ---
        base_rate = 100.0
        if is_silent:
            request_rate = base_rate * rng.uniform(0.05, 0.15)
        else:
            request_rate = base_rate * diurnal * burst_mult
            # Add Poisson-like noise
            request_rate += rng.gauss(0, request_rate * 0.08)
            request_rate = max(10.0, request_rate)

        # --- Per-node CPU and memory ---
        node_data: list[float] = []
        for i in range(NUM_NODES):
            # CPU correlates with request rate but has per-node characteristics
            load_factor = request_rate / (base_rate * 1.5)
            cpu = (
                node_base_cpu[i]
                + node_sensitivity[i] * load_factor
                + rng.gauss(0, 0.03)
            )
            # Occasional per-node anomaly (simulates GC pauses, log rotation)
            if rng.random() < 0.02:
                cpu += rng.uniform(0.15, 0.35)
            cpu = max(0.02, min(0.99, cpu))

            # Memory: slower-moving, correlated with CPU
            mem = 0.3 + cpu * 0.4 + rng.gauss(0, 0.02)
            mem = max(0.05, min(0.95, mem))

            node_data.extend([round(cpu, 4), round(mem, 4)])

        # --- Latency injection: extra latency from trace ---
        if is_silent:
            latency_inj = rng.uniform(0, 3)
        elif burst_mult > 2.0:
            latency_inj = rng.uniform(15, 60) * (burst_mult / 3.0)
        else:
            latency_inj = rng.uniform(0, 12) * diurnal

        row = [step] + node_data + [round(request_rate, 2), round(latency_inj, 2)]
        rows.append(row)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Generated trace: {OUTPUT_FILE} ({NUM_STEPS} steps, {NUM_NODES} nodes)")


if __name__ == "__main__":
    generate_trace()
