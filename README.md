# Distributed Infrastructure Management Environment

A simulated distributed system environment for training and evaluating LLM agents on infrastructure management tasks. Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

## Overview

The environment models a weighted graph of **8 compute nodes** with realistic distributed systems dynamics: stochastic request arrivals, node failures, load redistribution, and **cascading failure chains**. An LLM agent observes the system state and takes management actions to maintain stability.

## Observation Space

| Field          | Type          | Description                                       |
|----------------|---------------|---------------------------------------------------|
| `cpu_loads`    | `list[float]` | CPU utilization [0.0, 1.0] per node               |
| `queue_lengths`| `list[int]`   | Pending request count per node                    |
| `failed_nodes` | `list[int]`   | Indices of currently failed nodes                 |
| `latency_ms`   | `float`       |Rolling average end-to-end latency in ms          |
| `request_rate` | `float`       | Incoming requests per second                      |
| `step`         | `int`         | Current step within the episode                   |
| `task_hint`    | `str`         | Natural language description of current objective |

## Action Space

| Action           | Parameters                          | Effect                                          |
|------------------|-------------------------------------|-------------------------------------------------|
| `restart_node`   | `target: int`                       | Brings failed node back online (2-step delay)   |
| `reroute_traffic`| `from_node: int`, `to_node: int`    | Shifts 30% of load between nodes                |
| `scale_up`       | —                                   | Adds temporary capacity node for 10 steps       |
| `throttle`       | `rate: float [0, 1]`                | Reduces incoming request acceptance rate        |
| `no_op`          | —                                   | Do nothing (passive observation step)           |

## Graded Tasks

### Task 1 — Traffic Spike Recovery (Easy)
System receives 3× normal request rate. Keep latency below 50ms and maintain uptime.

### Task 2 — Single Node Failure (Medium)
A node fails mid-episode. Restart it and maintain >80% system uptime.

### Task 3 — Cascading Failure Prevention (Hard)
Two nodes are near-critical. Prevent the failure chain before it triggers.

## Reward Function

Dense step-level reward at every time step:

```
R(t) = 0.40 × uptime_ratio
     − 0.30 × normalized_latency
     − 0.20 × overload_fraction
     − 0.10 × (actions_taken / max_steps)
     + 0.50 × cascade_prevented_bonus
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run smoke test
python test_smoke.py

# Run the LLM agent
export HF_TOKEN=hf_xxxxx
python inference.py
```

## Docker

```bash
# Build
docker build -t distributed-infra-env .

# Run
docker run -p 8000:8000 distributed-infra-env
```

## API Endpoints

- `POST /reset` — Reset environment (pass `{"task": "traffic_spike"}`)
- `POST /step` — Execute action (pass `{"action": {"action_type": "no_op"}}`)
- `GET /state` — Get current environment state
- `GET /health` — Health check
- `WebSocket /ws` — Persistent session via WebSocket

## Technology Stack

| Component       | Technology                                      |
|----------------|--------------------------------------------------|
| HTTP Server    | FastAPI                                          |
| Data Models    | Pydantic v2                                      |
| Simulation     | Pure Python (node graph, load redistribution)    |
| Containerization | Docker                                         |
| Deployment     | HuggingFace Spaces                               |
| LLM Interface  | OpenAI-compatible client → HF Inference API      |
| LLM Model      | meta-llama/Llama-3.1-8B-Instruct                 |
