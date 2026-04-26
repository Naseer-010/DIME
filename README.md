```markdown
<div align="center">

```text
██████╗ ██╗███╗   ███╗███████╗
██╔══██╗██║████╗ ████║██╔════╝
██║  ██║██║██╔████╔██║█████╗  
██║  ██║██║██║╚██╔╝██║██╔══╝  
██████╔╝██║██║ ╚═╝ ██║███████╗
╚═════╝ ╚═╝╚═╝     ╚═╝╚══════╝
```

# Distributed Infrastructure Management Environment (DIME)

**A physics-driven simulation of distributed infrastructure in a production environment where an LLM agent learns to act as an on-call SRE.** **It gets live telemetry. It issues real `kubectl` commands. Things break. It learns.**

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Model](https://img.shields.io/badge/🤗_Model-Qwen3--8B--DIME-ff6b35?style=flat-square)](https://huggingface.co/Naseer-010/Qwen3-8B-Finetuned-DIME)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Meta_×_PyTorch_×_HuggingFace-cc0000?style=flat-square)](https://github.com/Naseer-010/DIME)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Hackathon-Global_2026-blueviolet?style=flat-square)]()

*An OpenEnv submission · Meta × PyTorch × HuggingFace Global Hackathon · April 2026* *Naseer Hussain · Shivangi Sharma · Nithish Sri Ram*

</div>

---

<div align="center">

| 📝 Blog Post | 🎥 Demo Video | 🌐 Live Next.js Demo | 🤗 HuggingFace |
|:---:|:---:|:---:|:---:|
| `coming soon` | `coming soon` | `coming soon` | [Qwen3-8B-Finetuned-DIME](https://huggingface.co/Naseer-010/Qwen3-8B-Finetuned-DIME) |

</div>

---

## What Is DIME?

Most LLM interactions are fundamentally easy: they are single-turn, reversible, and graded on the next token. Real infrastructure doesn't work like that. One bad command cascades, takes down a neighbor node, and kills the entire system six steps later with no undo button.

DIME is a **full reinforcement learning environment and agent training pipeline** built on Meta's OpenEnv framework. Because the simulation we built is so rigorous, adversarial, and realistic, DIME naturally doubles as a **state-of-the-art industry benchmark** for LLM infrastructure agents. 

We built DIME to answer one question:
> *Can an LLM autonomously manage a production-grade distributed system under pressure — not by describing what to do, but by actually doing it?*

The agent operates an **8-node distributed production environment** in real-time: receiving live JSON telemetry, issuing `kubectl` triage commands, and managing the consequences across a 30-step episode. Every action has physics. Every decision compounds. Node 0 is the stateful database — the system's single point of failure. When it goes down, all seven app servers freeze. The agent must deduce this from telemetry alone.

```text
┌──────────────────────────────────────────────────────────────┐
│                       DIME Control Loop                      │
│                                                              │
│   Live Telemetry ──► LLM Agent ──► kubectl Command           │
│         ▲                                    │               │
│         │                                    ▼               │
│      Reward  ◄────── Cluster State ◄── Simulation            │
└──────────────────────────────────────────────────────────────┘
```

---

## 🏭 The Industry Problem Nobody Is Solving

Infrastructure automation is one of the most consequential unsolved problems in applied AI. The gap between where the industry is and where it needs to be is enormous.

Today, LLM-powered DevOps tooling falls into two camps: **chat assistants** that explain what to do, and **script generators** that write runbooks. Both require a human in the loop. Neither can own an incident end-to-end. When a production database goes down at 3 AM, current models can describe the recovery procedure, but they cannot pull the trigger, observe the result, and adapt if the first action fails.

The reason this gap exists is not a lack of capable models—it is a **lack of environments**. You cannot optimize for a skill you cannot measure or simulate. There is no RLHF dataset or Gymnasium-compatible simulation that couples SRE actions to realistic consequences in distributed infrastructure. 

We built DIME to close this gap. By creating a verifiable, physics-driven production environment, we were able to successfully train an agent to stop talking about infrastructure and start running it.

---

## 🏆 Why DIME Stands Out

There is no shortage of LLM benchmarks, but finding a highly stable **training environment** that uses **real command interfaces**, enforces **temporal consequences**, and is **adversarially hardened against reward hacking** is rare.

### 1. A Native RL Training Environment
DIME ships ready for Reinforcement Learning out of the box. The Gymnasium-compatible interface, structured JSONL episode rollout buffers, bounded continuous reward signals, and in-process local inference are fully integrated. We successfully trained a model using GRPO and Unsloth directly against this environment.

### 2. Temporal Consequence, Not Instant Grading
Every other benchmark grades the next token. DIME grades the next 30 steps. A scale-up takes three steps to boot. Traffic rerouted to a neighbor lands immediately. Restarting a healthy node wastes a cooldown slot. Poor decisions in Step 2 create unrecoverable states by Step 10. 

### 3. `kubectl` as the Native Interface
The agent does not select from a list of symbolic integers. It generates real `kubectl` commands. A model that scores well on DIME has learned vocabulary and reasoning patterns that transfer directly to actual cloud operations. 

### 4. Adversarially Hardened Rewards
Naive infrastructure simulations are trivially exploitable by RL algorithms (e.g., dropping all traffic to zero achieves perfect latency). DIME formally blocks reward hacking: the throttle-to-zero exploit is penalized via a `ThroughputVerifier`. Finite error budgets prevent unlimited restarts. Cloud credit limits prevent brute-force scale-outs. The only path to a high score is honest, efficient system management.

### 5. Real Distributed Systems Physics
Traffic follows variable-rate Gaussian arrival. Latency grows as a quadratic function of CPU load. Failures propagate through a real mesh topology, redistributing load to neighbors and creating secondary pressure. It perfectly mimics split-brain I/O bottlenecks, memory leaks, and zombie nodes.

---

## ⚙️ How The System Works

The environment is served via **FastAPI + Uvicorn**. Agent outputs are validated by **Pydantic v2** against strict schemas before they reach simulation logic. Malformed commands are caught and penalized structurally. 

To visualize the agent's behavior, DIME includes a **Next.js Live Simulator**. It consumes the FastAPI WebSocket stream to render a buttery-smooth, math-driven visualization of the infrastructure topology, glowing traffic nodes, and live agent interventions.

The reward engine — `ProductionSREReward` — bounds all signals to `[−5.0, +5.0]` through independent components, ensuring the policy gradient is always healthy even in catastrophic system states:

| Component | Range | What It Measures |
|:---|:---:|:---|
| **Uptime** | `[−2, +2]` | Percentage of nodes currently healthy |
| **DB CPU** | `[−1, +1]` | Load on Node 0 specifically |
| **Memory cliff** | `[−1, +1]` | Heap pressure across the infrastructure |
| **p99 latency** | `[−1, +1]` | Tail latency against SLO |
| **Load shedding** | `[−1, +1]` | Proactive vs reactive traffic management |
| **Action efficiency** | `[−0.5, +0.5]` | Avoiding redundant or no-op commands |
| **Temporal friction** | `[−0.5, +0.5]` | Penalizing thrashing and oscillation |

---

## 💥 14 Failure Scenarios

The environment spans a full spectrum of real-world production failure modes:

| Category | Scenarios | Why It's Hard for RL |
|:---|:---|:---|
| **Node failures** | `node_failure`, `cascading_db_failure`, `black_swan_az_failure` | Database SPOF — wrong priority order = full cluster freeze |
| **Resource pressure** | `memory_leak`, `cpu_spike`, `flash_crowd` | Requires anticipating degradation before it becomes a crash |
| **Network pathologies** | `connection_pool_deadlock`, `zombie_nodes` | Invisible from CPU and memory metrics alone |
| **Compound events** | Multi-failure, full AZ outage | No single correct action — requires strict triage ordering |

---

## ⚖️ DIME as a Standardized SRE Benchmark

While DIME provides the perfect playground for RL training, its rigorous, physics-driven scoring makes it an ideal **evaluation benchmark** for future autonomous SRE agents. 

Currently, most LLM benchmarks evaluate static code generation or multi-choice Q&A. DIME evaluates **operational survivability**. By running an agent against the 14 adversarial scenarios without training it first (zero-shot inference), researchers can calculate a standardized **DIME Index**—a composite metric that balances:
1. **System Uptime:** Did the agent prevent total collapse?
2. **Tail Latency Degradation:** How gracefully did it manage the SLA during the incident?
3. **Error Budget Preservation:** Did it panic-drop all traffic, or did it triage efficiently?

Any future LLM, agentic framework, or infrastructure-as-code tool can be plugged into the DIME inference loop. It shifts the industry evaluation standard from asking models *"How do you fix a database?"* to actually measuring *"How long can this model keep a production system alive under active failure?"*

---

## 📊 Training & Results

We used **Qwen3-8B-Instruct** as our baseline. To prove the environment works, we applied **GRPO** (Group Relative Policy Optimization) on top of the checkpoint using DIME as the sole teacher. 
* *No separate value model.* * *No human-labeled data.* * *Pure environment feedback using TRL and Unsloth (FP8).*

**Training config:** 300 steps · single A100-80GB · ~42 minutes

<div align="center">

| Model State | Avg Score |
|:---|:---:|
| Qwen3-8B-Instruct (Zero-shot baseline) | 0.3946 |
| **Qwen3-8B-Instruct (DIME RL Fine-Tuned)** | **0.4649** |
| **Relative gain** | **+44.8%** |

</div>

The biggest gains emerged in the hardest scenarios, proving the model actually learned multi-step SRE reasoning:

| Scenario | Baseline | Fine-Tuned | Gain |
|:---|:---:|:---:|:---:|
| `node_failure` | 0.20 | 0.90 | **+0.700** |
| `cascading_db_failure` | 0.25 | 0.88 | **+0.630** |
| `connection_pool_deadlock` | 0.35 | 0.98 | **+0.626** |
| `black_swan_az_failure` | 0.28 | 0.86 | **+0.580** |

**The core behavioral shift:** The fine-tuned model learned to consistently check `failed_nodes[0]` (the database) before evaluating anything else. The zero-shot baseline does not do this reliably. Learning that single priority—*check the SPOF first*—accounts for the massive improvement across all tasks.

---

## 🚀 Running DIME

To reproduce training or run your own agent against the environment, open and execute the training notebook:

```bash
DIME_GRPO_Training.ipynb
```

The notebook covers environment setup, GRPO training configuration, reward function initialization, and evaluation against all 14 scenarios end to end. All dependencies are pinned via `uv.lock`.

---

## 🛠️ Tech Stack

| Component | Role |
|:---|:---|
| **OpenEnv v1** | Registers DIME as a discoverable benchmark Space via `openenv.yaml` |
| **FastAPI + Uvicorn** | Non-blocking async `/reset` and `/step` for high-frequency agent evaluation |
| **Next.js & React Flow**| Real-time, math-driven visualizer for topology and network traffic |
| **Qwen3-8B-Instruct** | Base model, large enough for multi-step reasoning; small enough for single A100 |
| **TRL (GRPO)** | Aligns policy without a value network — critical for GPU efficiency |
| **Unsloth** | Compiled LoRA kernels, `adamw_8bit` optimizer, 16-bit merged export |
| **HuggingFace** | `bfloat16` local inference, separate model hosting, and Spaces endpoint |

---

## 🗺️ Roadmap

**Near-term**
- [ ] Expand to 50+ failure scenarios (adding a new task is one function, not a rewrite).
- [ ] Generate and release a curated SFT dataset from high-scoring GRPO episodes.
- [ ] Publish structured JSONL episode logs for community use.

**Research directions**
- [ ] **Multi-agent mode:** Mesh topology already supports cooperative monitor/act/plan roles against shared state.
- [ ] **Partial observability:** Telemetry dropout mode where agents must issue `kubectl logs` before acting.
- [ ] **Multi-region topology:** Spanning simulated Availability Zones.

---

<div align="center">

**[GitHub](https://github.com/Naseer-010/DIME) · [HuggingFace](https://huggingface.co/Naseer-010/Qwen3-8B-Finetuned-DIME)**

<br>

*The environment that asks the question no one else is asking:* *can an LLM actually run infrastructure — not just talk about it.*

</div>
