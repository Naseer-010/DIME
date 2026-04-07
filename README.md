# DIME — Distributed Infrastructure Management Environment

> A high-fidelity simulated distributed system for training and evaluating LLM agents on complex Site Reliability Engineering (SRE) tasks. Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

---

## Table of Contents

- [Overview](#overview)
- [Simulation Dynamics](#simulation-dynamics)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Graded Tasks](#graded-tasks)
- [Reward Function](#reward-function)
- [Baseline Agent Evaluation](#baseline-agent-evaluation)
- [Setup & Execution](#setup--execution)
- [Docker Deployment](#docker-deployment)
- [Technology Stack](#technology-stack)

---

## Overview

While many LLM environments focus on web browsing or puzzle games, there is a critical gap in evaluating an agent's ability to manage **dynamic, real-world backend infrastructure**. DIME fills this gap.

DIME models a **mesh-topology graph of compute nodes** experiencing realistic distributed systems dynamics. An LLM agent acts as an automated SRE, observing live system telemetry — CPU utilization, queue depths, end-to-end latency — and issuing decisive management actions such as traffic rerouting, node scaling, and load throttling to maintain cluster stability and prevent catastrophic cascading failures.

---

## Simulation Dynamics

DIME is not a static state machine. It is driven by a set of mathematical models designed to produce realistic, unpredictable infrastructure behavior:

| Model                                   | Description                                                                                                                                                                                |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Stochastic Traffic**                  | Request arrivals follow a variable-rate Gaussian distribution (approximating a Poisson process with burst potential) to simulate real-world web traffic surges.                            |
| **Non-Linear Latency (Congestion)**     | End-to-end latency is computed via an Exponential Moving Average (EMA), incorporating base network delay, linear queue delay, and a **quadratic CPU pressure penalty** (`latency ∝ CPU²`). |
| **Stochastic & Deterministic Failures** | Nodes fail **deterministically** if CPU exceeds 90% for 3 consecutive steps, and face a **probabilistic failure risk** above 85% CPU to simulate hardware degradation under stress.        |
| **Cascading Load Redistribution**       | On node failure, its load and pending queue are violently redistributed to adjacent mesh neighbors, actively simulating the physics of a cascading cluster collapse.                       |

---

## Observation Space

At each step, the agent receives a structured observation containing the following fields:

| Field           | Type          | Description                                       |
| --------------- | ------------- | ------------------------------------------------- |
| `cpu_loads`     | `list[float]` | CPU utilization `[0.0, 1.0]` per node             |
| `queue_lengths` | `list[int]`   | Pending request count per node                    |
| `failed_nodes`  | `list[int]`   | Indices of currently failed nodes                 |
| `latency_ms`    | `float`       | Rolling average end-to-end latency in ms          |
| `request_rate`  | `float`       | Incoming requests per second                      |
| `step`          | `int`         | Current step within the episode                   |
| `task_score`    | `float`       | Real-time partial credit grader score             |
| `task_hint`     | `str`         | Natural language description of current objective |

---

## Action Space

The agent may issue one of five actions per step:

| Action            | Parameters                       | Effect                                                                    |
| ----------------- | -------------------------------- | ------------------------------------------------------------------------- |
| `restart_node`    | `target: int`                    | Brings a failed node back online (2-step boot delay)                      |
| `reroute_traffic` | `from_node: int`, `to_node: int` | Shifts 30% of active load & queue between two nodes                       |
| `scale_up`        | —                                | Adds a temporary capacity node (TTL: 10 steps)                            |
| `throttle`        | `rate: float [0, 1]`             | Drops a fraction of incoming traffic (e.g., `0.8` accepts 80%, drops 20%) |
| `no_op`           | —                                | Passive observation step; no intervention                                 |

---

## Graded Tasks

### Task 1 — Traffic Spike Recovery `[Easy]`

The system receives **3× normal request rate**. The agent must keep latency below **50ms** and maintain uptime. Graded on latency control and action efficiency.

### Task 2 — Single Node Failure `[Medium]`

A node fails mid-episode. The agent must restart it and maintain **>80% system uptime**. Graded heavily on **Mean Time To Repair (MTTR)**.

### Task 3 — Cascading Failure Prevention `[Hard]`

Two connected nodes are near-critical. The agent must proactively reroute traffic **before** the failure chain triggers. **Prevention is rewarded over recovery.**

### Task 4 — Flash Crowd Meltdown `[Expert]`

An unprecedented **5× traffic surge**. Pure survival scenario requiring aggressive, simultaneous use of `scale_up` and `throttle` to prevent total cluster collapse.

---

## Reward Function

DIME provides a dense, continuous step-level reward signal:

```
R(t) = + 0.40 × uptime_ratio
       − 0.30 × normalized_latency
       − 0.20 × overload_fraction
       − 0.10 × (actions_taken / max_steps)
       + 0.50 × cascade_prevented_bonus
```

---

## Baseline Agent Evaluation

A baseline evaluation was run using `meta-llama/Llama-3.1-8B-Instruct` (temperature=0.01) via an OpenAI-compatible endpoint. Results confirm the environment's punishing physics and rigorous grading:

| Task                | Score     | Analysis                                                                                                                                                        |
| ------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `traffic_spike`     | **0.090** | Agent spammed `restart_node` during a spike, dropping cluster capacity to zero. Grader correctly tanked the latency score and applied heavy action penalties.   |
| `node_failure`      | **0.050** | Agent failed to proactively route traffic away from the failed node, resulting in missed MTTR targets and massive uptime penalties.                             |
| `cascading_failure` | **0.310** | A fatal routing error sent traffic to a hot node, triggering a cascade. Earned 30% partial credit purely for mathematical action efficiency before termination. |
| `flash_crowd`       | **0.000** | Agent failed to drop load. Queue depths pushed the entire cluster to 100% CPU, triggering total meltdown in exactly **3 steps**.                                |

These metrics confirm that DIME enforces strict distributed systems dynamics and **easily defeats smaller models**, serving as a robust benchmark for frontier-level SRE agents (e.g., Nemotron, Qwen-72B, GPT-4).

---

## Setup & Execution

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run Smoke Tests

```bash
python test_smoke.py
```

### Run Baseline Evaluation

```bash
export HF_TOKEN=your_huggingface_token
python inference.py
```

---

## Docker Deployment

```bash
# Build the image
docker build -t distributed-infra-env .

# Run the container
docker run -p 8000:8000 distributed-infra-env
```

---

## Technology Stack

| Component        | Technology                                  |
| ---------------- | ------------------------------------------- |
| Framework        | Meta PyTorch OpenEnv                        |
| HTTP Server      | FastAPI                                     |
| Data Models      | Pydantic v2                                 |
| Simulation       | Pure Python (Mesh graph, stochastic queues) |
| Containerization | Docker                                      |
| Deployment       | Hugging Face Spaces                         |

---

<div align="center">
  Built for the OpenEnv Hackathon · Co-organized by Meta, PyTorch & Hugging Face
</div>
