<div align="center">

```
██████╗ ██╗███╗   ███╗███████╗
██╔══██╗██║████╗ ████║██╔════╝
██║  ██║██║██╔████╔██║█████╗  
██║  ██║██║██║╚██╔╝██║██╔══╝  
██████╔╝██║██║ ╚═╝ ██║███████╗
╚═════╝ ╚═╝╚═╝     ╚═╝╚══════╝
```

### Distributed Infrastructure Management Environment

**A Kubernetes cluster simulation where an LLM agent acts as an on-call SRE.**
**It gets live telemetry. It issues real `kubectl` commands. Things break. It learns.**

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Model](https://img.shields.io/badge/🤗_Model-Qwen3--8B--DIME-ff6b35?style=flat-square)](https://huggingface.co/Naseer-010/Qwen3-8B-Finetuned-DIME)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Meta_×_PyTorch_×_HuggingFace-cc0000?style=flat-square)](https://github.com/Naseer-010/DIME)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Hackathon-Global_2026-blueviolet?style=flat-square)]()

*An OpenEnv submission · Meta × PyTorch × HuggingFace Global Hackathon · April 2026*

*Naseer Hussain · Shivangi Sharma · Nithish Sri Ram*

</div>

---

<div align="center">

| 📝 Blog Post | 🎥 Demo Video | 📄 Research Paper | 🤗 HuggingFace |
|:---:|:---:|:---:|:---:|
| `coming soon` | `coming soon` | `coming soon` | [Qwen3-8B-Finetuned-DIME](https://huggingface.co/Naseer-010/Qwen3-8B-Finetuned-DIME) |

</div>

---

## What Is DIME?

Most LLM benchmarks are fundamentally easy. They are single-turn, reversible, and graded on the next token. A wrong answer just loses points. Real infrastructure doesn't work like that, one bad command cascades, takes down a neighbor node, and kills the cluster six steps later with no undo button.

DIME is a **full reinforcement learning benchmark and training environment** built on Meta's OpenEnv framework, designed to answer one question:

> *Can an LLM autonomously manage a production-grade distributed system under pressure — not by describing what to do, but by actually doing it?*

The agent operates an **8-node Kubernetes cluster** in real-time: receiving live JSON telemetry, issuing `kubectl` commands, and managing the consequences across a 30-step episode. Every action has physics. Every decision compounds.

```
┌──────────────────────────────────────────────────────────────┐
│                       DIME Control Loop                      │
│                                                              │
│   Live Telemetry ──► LLM Agent ──► kubectl Command          │
│         ▲                                    │               │
│         │                                    ▼               │
│      Reward  ◄────── Cluster State ◄── Simulation           │
└──────────────────────────────────────────────────────────────┘
```

**Node 0 is the stateful database** — the cluster's single point of failure. When it goes down, all seven app servers freeze. The agent has to figure this out from telemetry alone, with no hints and no warm-up.

### What We're Measuring

We use **Qwen3-8B-Instruct** as both our starting point and our baseline. The same model, unmodified, is evaluated zero-shot across all 14 DIME scenarios — this establishes what raw instruction-following capability gets you in an infrastructure context. We then apply **GRPO-based reinforcement learning** on top of that same checkpoint using DIME as the training environment. The resulting fine-tuned model is evaluated against the exact same 14 scenarios under identical conditions. Every number in this README compares these two versions of the same model, before and after RL, so the gains are attributable purely to what DIME taught it, not to any architectural difference.

---

## 🏭 The Industry Problem Nobody Is Solving

Infrastructure automation is one of the most consequential unsolved problems in applied AI. The gap between where the industry is and where it needs to be is enormous and almost nobody is measuring it correctly.

Today, LLM-powered DevOps tooling falls into two camps: **chat assistants** that explain what to do, and **script generators** that write runbooks. Both require a human in the loop to actually execute and validate. Neither can own an incident end-to-end. When a database node goes down at 3 AM, the model can describe the recovery procedure but it cannot pull the trigger, observe the result, and adapt if the first action makes things worse.

The reason this gap exists is not a lack of capable models. It is a lack of environments that can **train and evaluate** agents on the actual task. You cannot optimize for a skill you cannot measure. The infrastructure management problem has no RLHF dataset, no standardized benchmark, no Gymnasium-compatible simulation that couples actions to realistic consequences. The field has been benchmarking the wrong thing — language about infrastructure, not operation of it.

This means that every model claiming SRE capability today has been evaluated on proxies: did it write a correct YAML? Did it explain the right kubectl flags? These are necessary conditions but nowhere near sufficient. A model that can describe a scale-up procedure may still fail to execute one across a live degraded cluster where three alerts are firing simultaneously and the first action has a three-step boot delay.

DIME exists to close this measurement gap. Without a rigorous environment for training and evaluation, the industry will keep shipping models that talk well about infrastructure and fail silently when handed the keys to it.

---

## 🏆 Why DIME Stands Out

There is no shortage of LLM benchmarks. What is rare is a benchmark that is simultaneously a **training environment**, uses **real command interfaces**, enforces **temporal consequence**, and is **adversarially hardened against exploitation**. DIME is all four.

### 1. Temporal Consequence, Not Instant Grading

Every other benchmark grades the next token. DIME grades the next 30 steps. A scale-up takes three steps to boot. Traffic rerouted to a neighbor node lands immediately. Restarting a healthy node wastes a cooldown slot. The agent cannot skip ahead, cannot undo, and cannot see the future. Poor decisions made in step 2 create states that are difficult or impossible to recover from by step 10. This temporal depth is what separates infrastructure management from every task that existing benchmarks cover.

### 2. A Training Environment From Day One

Most benchmarks require months of additional engineering to become RL training environments. DIME ships as both simultaneously. The Gymnasium-compatible interface, structured JSONL episode logs, bounded reward signal with verified gradient variance, and in-process local inference are all present and integrated. Any team can take the DIME environment, swap in a different base model, and run their own RL loop with a single notebook.

### 3. `kubectl` as the Native Interface

The agent does not select from a list of symbolic action indices. It generates real `kubectl` commands — the same syntax a human SRE types into a terminal. This means benchmark performance has a direct operational interpretation: a model that scores well on DIME has learned vocabulary and reasoning patterns that transfer to actual Kubernetes operations. The gap between benchmark capability and deployment capability is as small as it can be.

### 4. Adversarially Hardened Reward

Naive infrastructure simulations are trivially exploitable. Drop all traffic to zero, latency hits zero, and a poorly designed reward function scores it as perfect performance. DIME formally blocks this: the throttle-to-zero exploit is tested and penalized. Finite error budgets prevent unlimited restarts. Cloud credit limits prevent brute-force scale-out. Action cooldowns prevent thrashing. The only path to a high score is honest cluster management.

### 5. Simulation Physics, Not Mocked State

Traffic follows variable-rate Gaussian arrival. Latency grows as a quadratic function of CPU load. Failures propagate through a real mesh topology, redistributing load to neighbors and creating secondary pressure. The environment is not a state machine with hand-coded transitions — it is a physics-driven simulation where emergent behavior arises from the interaction of components. Agents that learn to manage it are learning something that generalizes.

---

## ⚙️ How The Simulation Works

The cluster is served via **FastAPI + Uvicorn** with two endpoints the agent interacts with at each step:

```
POST /reset   →  initialize a fresh episode, receive initial cluster observation
POST /step    →  submit a kubectl action, receive next cluster state + reward signal
```

Agent outputs are validated by **Pydantic v2** against strict `InfraAction` / `InfraObservation` schemas before they reach simulation logic. Malformed commands are caught and penalized structurally — they do not silently pass through.

```
┌──────────────────────────────────────────────────┐
│                   DIME System                    │
│                                                  │
│  ┌─────────────┐      ┌──────────────────────┐  │
│  │  LLM Agent  │◄────►│  FastAPI + Uvicorn   │  │
│  │  (Qwen3-8B) │      │  /reset  /step       │  │
│  └─────────────┘      └──────────┬───────────┘  │
│                                  │               │
│                        ┌─────────▼──────────┐   │
│                        │   Pydantic v2      │   │
│                        │   Schema Guard     │   │
│                        └─────────┬──────────┘   │
│                                  │               │
│                        ┌─────────▼──────────┐   │
│                        │  DistributedInfra  │   │
│                        │  Environment       │   │
│                        │  (8-node cluster)  │   │
│                        └─────────┬──────────┘   │
│                                  │               │
│                        ┌─────────▼──────────┐   │
│                        │ ProductionSREReward │   │
│                        │    [−5.0, +5.0]    │   │
│                        └────────────────────┘   │
└──────────────────────────────────────────────────┘
```

The reward engine — `ProductionSREReward` — bounds all signals to `[−5.0, +5.0]` through seven independent components, ensuring the gradient is always meaningful even in catastrophic cluster states:

| Component | Range | What It Measures |
|:---|:---:|:---|
| Uptime | `[−2, +2]` | Percentage of nodes currently healthy |
| DB CPU | `[−1, +1]` | Load on Node 0 specifically |
| Memory cliff | `[−1, +1]` | Heap pressure across the cluster |
| p99 latency | `[−1, +1]` | Tail latency against SLO |
| Load shedding | `[−1, +1]` | Proactive vs reactive traffic management |
| Action efficiency | `[−0.5, +0.5]` | Avoiding redundant or no-op commands |
| Temporal friction | `[−0.5, +0.5]` | Penalizing thrashing and oscillation |

---

## 14 Failure Scenarios

The benchmark spans a full spectrum of real-world infrastructure failure modes:

| Category | Scenarios | Why It's Hard |
|:---|:---|:---|
| **Node failures** | `node_failure`, `cascading_db_failure`, `black_swan_az_failure` | Database SPOF — wrong priority order = full cluster freeze |
| **Resource pressure** | `memory_leak`, `cpu_spike`, `flash_crowd` | Requires anticipating degradation before it becomes a crash |
| **Network pathologies** | `connection_pool_deadlock`, `zombie_nodes` | Invisible from CPU and memory metrics alone |
| **Compound events** | Multi-failure, full AZ outage | No single correct action — requires strict triage ordering |

> The flash crowd scenario destroyed a zero-shot 8B model in exactly **3 steps**. That is the benchmark working correctly.

---

## 📊 Results

Fine-tuned **Qwen3-8B-Instruct** using **GRPO** (Group Relative Policy Optimization) with Unsloth. No separate value model. No human-labeled data. Pure environment feedback against the DIME simulation.

**Training config:** 300 steps · single A100-80GB · ~42 minutes

<div align="center">

| Model | Avg Score |
|:---|:---:|
| Qwen3-8B-Instruct (zero-shot baseline) | 0.3946 |
| Qwen3-8B-Instruct (DIME RL fine-tuned) | **0.4649** |
| **Relative gain** | **+44.8%** |

</div>

The biggest gains came from the hardest scenarios — exactly where zero-shot instruction-following collapses under multi-step consequence:

| Scenario | Baseline | Fine-Tuned | Gain |
|:---|:---:|:---:|:---:|
| `node_failure` | 0.20 | 0.90 | **+0.700** |
| `cascading_db_failure` | 0.25 | 0.88 | **+0.630** |
| `connection_pool_deadlock` | 0.35 | 0.98 | **+0.626** |
| `black_swan_az_failure` | 0.28 | 0.86 | **+0.580** |

**The core behavioral shift:** the fine-tuned model consistently checks `failed_nodes[0]` — Node 0, the database — before evaluating anything else. The zero-shot baseline does not do this reliably. That single learned priority, check the SPOF first, accounts for the majority of improvement across all 14 tasks.

---

## Running DIME

To reproduce training or run your own agent against the benchmark, open and execute the notebook:

```
DIME_GRPO_Training.ipynb
```

The notebook covers environment setup, GRPO training configuration, reward function initialization, and evaluation against all 14 scenarios end to end. All dependencies are pinned via `uv.lock` to ensure the training environment matches the benchmark exactly.

---

## Tech Stack

| Component | Role |
|:---|:---|
| **OpenEnv v1** | Registers DIME as a discoverable benchmark Space via `openenv.yaml` |
| **FastAPI + Uvicorn** | Non-blocking async `/reset` and `/step` — handles high-frequency agent evaluation without queuing |
| **Pydantic v2** | Strict schema validation on all agent I/O before reaching simulation logic |
| **Qwen3-8B-Instruct** | Base model, large enough for multi-step SRE reasoning; small enough for single A100 fine-tuning |
| **GRPO** | Aligns policy without a value network — critical for staying within hackathon GPU budget |
| **Unsloth** | Compiled LoRA kernels, `adamw_8bit` optimizer, 16-bit merged export |
| **HuggingFace Transformers** | `bfloat16` local inference, model hosting, and Spaces endpoint |
| **Docker + uv** | Single-command reproducible environment; `uv.lock` pins exact package versions |

---

## Roadmap

**Near-term**
- [ ] Expand to 50+ failure scenarios  adding a new task is one function, not a rewrite
- [ ] Generate and release curated SFT dataset from high-scoring GRPO episodes
- [ ] Publish structured JSONL episode logs for community use

**Research directions**
- [ ] Multi-agent mode — mesh topology already supports cooperative monitor / act / plan roles against shared state
- [ ] Partial observability — telemetry dropout mode where agents must issue `query_logs` before acting; no existing benchmark has approached this
- [ ] Multi-cluster topology spanning simulated Availability Zones
- [ ] Frontier model evaluation suite on the hard-task subset

---

<div align="center">

**[GitHub](https://github.com/Naseer-010/DIME) · [HuggingFace](https://huggingface.co/Naseer-010/Qwen3-8B-Finetuned-DIME)**

<br>

*The benchmark that asks the question no one else is asking:*
*can an LLM actually run infrastructure — not just talk about it.*

</div>