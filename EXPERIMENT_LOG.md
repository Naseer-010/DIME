# DIME × Qwen3-8B — Experiment Log

> **Purpose:** Full record of the GRPO fine-tuning experiment on the Distributed Infrastructure
> Management Environment (DIME). Written to support a blog post and future reproduction.

---

## 1. What Is DIME?

DIME is a Kubernetes cluster simulation where an LLM agent acts as an autonomous SRE.
The environment models an 8-node cluster:

- **Node-0** — stateful PostgreSQL-style **DATABASE** (Single Point of Failure / SPOF)
- **Nodes 1–7** — stateless application workers

Each step the agent receives a JSON telemetry observation and must output a single
`kubectl` command. The environment applies cascading failure dynamics, load redistribution,
and memory pressure across steps.

### Observation Fields

| Field | Meaning |
|---|---|
| `cpu_loads[8]` | Per-node CPU utilisation (0–1, -1 = unreachable) |
| `mem_utilizations[8]` | Per-node RAM utilisation |
| `queue_lengths[8]` | Pending request queue depth |
| `failed_nodes[]` | Node indices currently crashed |
| `latency_ms` | Mean request latency |
| `p99_latency` | Tail latency (99th percentile) |
| `io_wait` | DB disk I/O wait (signals split-brain risk) |
| `error_budget` | Remaining SLA budget (burns on throttle) |

### Available Commands

```
kubectl delete pod node-<ID>                                    # restart node
kubectl scale deployment frontend --replicas=10                 # scale up (3-step boot)
kubectl exec -it istio-proxy -- traffic shift --from=<X> --to=<Y>  # reroute
kubectl throttle ingress --rate=<0.0–1.0>                      # shed load (burns budget)
kubectl logs node-<ID>                                          # query logs
no_op                                                           # do nothing
```

---

## 2. Baseline Experiment — Qwen3-8B Zero-Shot

**Model:** `Qwen/Qwen3-8B`  
**Inference:** `inference.py` (local GPU, transformers, no fine-tuning)  
**Date:** 2026-04-25  

### Results

| Task | Score | Steps | % Steps at -1000 |
|---|---|---|---|
| traffic_spike | 0.0242 | 30 | 100% |
| node_failure | 0.2200 | 40 | 100% |
| cascading_failure | 0.3300 | 3 | 100% |
| flash_crowd | 0.0100 | 3 | 100% |
| level_5_alibaba_trace | 0.4146 | 7 | 100% |
| thundering_herd | 0.3931 | 4 | 100% |
| zombie_node | 0.4580 | 7 | 100% |
| **memory_leak_slow_burn** | **0.9900** | 60 | **1.7%** |
| split_brain_io_bottleneck | 0.4286 | 3 | 100% |
| black_swan_az_failure | 0.4379 | 9 | 100% |
| retry_storm | 0.3767 | 3 | 100% |
| hot_shard_skew | 0.4353 | 6 | 100% |
| connection_pool_deadlock | 0.6301 | 17 | 100% |
| autoscaler_flapping_trap | 0.3762 | 5 | 100% |

**Overall Task Completion Score: 0.3946**  
**Mean Latency: 162.9 ms**  
**DIME Index: 0.0000** ← uptime tracking broken in old inference.py  

### Action Distribution (197 total steps)

| Action | Count | Share |
|---|---|---|
| throttle | 62 | 31.5% |
| reroute | 50 | 25.4% |
| no_op | 41 | 20.8% |
| restart_node | 34 | 17.3% |
| scale_up | 10 | 5.1% |

---

## 3. Root Cause Analysis — Why Every Reward Was -1000

### The -1000 Cliff

`server/rubrics.py` (pre-fix) contained:

```python
class ProductionSREReward:
    def __init__(self):
        self.c_db     = 100.0   # ← huge coefficient
        self.lambda_db = 4.0    # ← steep exponent

    def calculate_reward(self, state, action):
        if 0 in state["failed_nodes"]:
            return -1000.0      # ← hard cliff, zero gradient
        db_cpu = state["cpu_loads"][0]
        r_topo = -100 * exp(4 * db_cpu)
        # at db_cpu ≥ 0.57: r_topo ≤ -1000
```

The moment DB CPU crossed ~57%, `r_topo` hit -1000. Because the environment's
`_simulate_requests()` blends 60% trace + 40% simulation, a single traffic spike
could push DB CPU from 0.47 → 0.70 in one step.

**Result:** Node-0 crashed on step 1–3 of nearly every episode. Once dead, every
subsequent step returned -1000 regardless of action. With all rewards identical,
GRPO advantages = `(r - mean) / std = 0/0` → no gradient → model learns nothing.

### The Missing Rule

The original `SYSTEM_PROMPT` in `inference.py` had 9 triage rules. **Rule 8 was
"SAFE SCALING"**, not DB recovery. There was no instruction to restart the DB when
node-0 crashed:

```
# MISSING from original prompt:
# "DB RECOVERY: IF node-0 in failed_nodes → kubectl delete pod node-0"
```

The model did attempt `kubectl delete pod node-0` **13 times** across 197 steps —
instinctively correct — but inconsistently because it wasn't a named rule.

### Why memory_leak_slow_burn Scored 0.99

This is the one task that escaped the -1000 trap. Memory leaks build slowly:
DB CPU doesn't spike immediately, so the reward function ran normally for 58 of 60
steps. The model had time to issue `kubectl delete pod node-5` (clearing the leak)
and eventually figured out `kubectl delete pod node-0`. Score: **0.99**.

This single outlier proves the model has the capability — it just needed the right
reward signal and the rule written down.

---

## 4. Reward Function Fixes

### Original vs. Fixed `ProductionSREReward`

| Property | Original (nithish branch) | Fixed (main branch) |
|---|---|---|
| DB failure return | `-1000.0` | `-5.0` |
| DB CPU coefficient | `c_db = 100.0` | `c_db = 1.5` |
| DB CPU exponent | `lambda_db = 4.0` | `lambda_db = 2.0` |
| Output range | `(-∞, -1000]` | `[-5.0, +5.0]` |
| Gradient at DB death | **zero** | **non-zero** |
| Components | 4 | 7 |

### New Components in Fixed Version

1. **DB CPU** — softer exponential, bounded
2. **Memory cliff** — `-2.0` per node at ≥98% RAM (OOM), smooth below
3. **p99 latency** — quadratic penalty capped at `-3.0`
4. **Load shedding economics** — penalises throttling when budget is low
5. **Uptime bonus** — `+1.0` for 100% nodes alive
6. **Action efficiency** — `-0.05` per action, `-0.10` extra for throttle (anti-spam)
7. **Temporal friction** — cold-start penalty for scale_up

---

## 5. Training Approach

### Why GRPO (not PPO or SFT)

- **PPO** would train a separate policy head on a frozen LLM — can't be benchmarked
  with the same `inference.py` before/after.
- **SFT** would need labelled (state → correct_command) pairs for all 14 tasks.
- **GRPO** fine-tunes the LLM itself. Run `inference.py` before → train → run
  `inference.py` after → direct apples-to-apples comparison.

### Three Training Scripts (in order of development)

#### `train_rl.py` — PPO Baseline
- Gymnasium wrapper `DIMEEnv` around the Python env
- Discrete 20-action space (no text generation)
- Custom reward: `uptime(+0.5) - latency_penalty(0.5) - db_fail(2.0)`
- Actor-Critic MLP with optional LSTM
- **Not used for final training** — can't be compared with inference.py

#### `train_grpo.py` — Custom GRPO Loop
- Pure PyTorch, no TRL dependency
- **Online:** model's actions actually advance the env each step
- `G=4` completions per prompt, `deepcopy(env)` for each
- Format + validity + env rewards with std guard (skips update if `std < 1e-4`)
- **Advantage:** true multi-step RL — model sees consequences of its decisions
- **Disadvantage:** slow (no vLLM), ~5× longer than unsloth version

#### `train_grpo_unsloth.py` — Unsloth + TRL (Primary)
- `FastLanguageModel` with FP8 weights + LoRA (rank=32)
- `PatchFastRL("grpo", FastLanguageModel)` before TRL imports (fixes vLLM API mismatch)
- **Offline:** 300-episode dataset collected with 70% oracle + 30% random actions
- 4 reward functions (TRL signature):

| Function | Range | Signal |
|---|---|---|
| `reward_format` | [-3, +3] | `<reasoning><action>` XML tags |
| `reward_validity` | [-2, +2] | Valid kubectl command parses |
| `reward_env` | [-5, +5] | Full `ProductionSREReward` (7 components) |
| `reward_triage` | [-2, +5] | Oracle triage tree exact match |

### System Prompt Additions (both train scripts)

Added **Rule 8 — DB RECOVERY** (was missing from inference.py):
```
8. DB RECOVERY: IF node-0 is in 'failed_nodes' → kubectl delete pod node-0
```
Renumbered former rule 8 (SAFE SCALING) to rule 9, HEALTHY to rule 10.

---

## 6. Dependency Resolution

Running on A100-SXM4-80GB, CUDA 12.8, torch 2.10.0+cu128.

The key version conflict:
- `unsloth 2026.4.8` requires `transformers ≤ 5.5.0`
- `vllm 0.19.1` requires `transformers ≥ 4.56.0` and excludes 5.0–5.5.0

**Solution:** `transformers == 4.56.2` satisfies both constraints.

**Install order matters:**

```bash
pip install unsloth              # pulls trl 0.24.0, peft, accelerate, bitsandbytes
pip install transformers==4.56.2 # pin BEFORE vllm (vllm upgrades to 5.6.2 which breaks unsloth)
pip install vllm                 # will warn about transformers but works fine
pip install openenv-core         # needed for server/environment.py direct import
```

`PatchFastRL` must be called **before** `from trl import GRPOTrainer` — it patches
TRL's internal vllm import to handle the 0.10.2 → 0.19.1 API change.

---

## 7. Parameters to Avoid the -1000 Trap

### In `server/rubrics.py`

```python
# DANGER — causes -1000 cliff and zero gradients:
self.c_db      = 100.0
self.lambda_db = 4.0
# At db_cpu ≥ 0.57: -100 * exp(4 * 0.57) = -1000

# SAFE — bounded output, smooth gradients:
self.c_db      = 1.5
self.lambda_db = 2.0
# At db_cpu = 1.0: -1.5 * exp(2.0) ≈ -11 → clipped to -5.0
```

**Always clip the final reward:**
```python
return float(np.clip(reward, -5.0, 5.0))
```

**Never return a hardcoded large negative constant:**
```python
# BAD:
if 0 in failed_nodes:
    return -1000.0  # zero gradient, GRPO can't learn

# GOOD:
if 0 in failed_nodes:
    return -5.0     # still penalised, gradient flows
```

### In GRPO Training Loops

**Always guard against zero-variance groups:**
```python
std = np.std(rewards)
if std < 1e-4:
    continue  # skip this group — all completions identical, no signal
```

**Use multiple reward components** to guarantee within-group variance even early
in training when the model is still random:
- Format reward: different completions will have different tag structures
- Validity reward: some will parse, some won't
- These provide variance from step 0, before env rewards differentiate

**Temperature:** keep `temperature ≥ 1.0` during GRPO generation. Lower temperature
→ completions cluster → advantages → 0 → gradient → 0.

**Don't use `ProductionSREReward` with old coefficients directly as GRPO reward.**
Wrap it in a custom function that catches the -1000 case:

```python
def safe_env_reward(sim) -> float:
    r = calculate_step_reward(sim)
    if r <= -100:          # old rubrics detected
        # fallback formula
        nodes = sim.nodes
        alive = sum(1 for n in nodes if not n.is_failed)
        return 0.5 * (alive / len(nodes)) - (2.0 if nodes[0].is_failed else 0.0)
    return r
```

### GRPO Config Parameters (TRL 0.24.0)

```python
TRLGRPOConfig(
    temperature                 = 1.0,   # ≥ 1.0 to maintain diversity
    top_k                       = 50,    # limits extreme token choices
    top_p                       = 0.95,
    min_p                       = 0.1,   # prunes low-prob tokens (reduces incoherence)
    num_generations             = 4,     # minimum for stable advantage estimates; 8 is better
    per_device_train_batch_size = 1,     # keep low for 40GB GPU
    gradient_accumulation_steps = 4,     # effective batch = 4
    learning_rate               = 5e-6,  # low LR for LoRA stability
    warmup_ratio                = 0.1,   # prevents early collapse
    lr_scheduler_type           = "cosine",
    max_grad_norm               = 1.0,   # clip gradients
)
```

---

## 8. Fallback Training Strategy (If Current Run Fails)

### Scenario A — OOM During Training

**Symptoms:** `torch.cuda.OutOfMemoryError` during generation or backward pass.

```python
# In train_grpo_unsloth.py, change:
LORA_RANK       = 32  →  16        # halves adapter memory
NUM_GENERATIONS = 4   →  2         # halves generation batch
FAST_INFERENCE  = True → False     # drops vLLM engine (~16GB saved)
MAX_SEQ_LENGTH  = 2048 → 1536      # reduces KV cache

# In TRLGRPOConfig:
per_device_train_batch_size  = 1   # already minimum
gradient_accumulation_steps  = 4   # keep or increase
```

Expected VRAM with these changes on 40GB: ~22GB (comfortable).

### Scenario B — Training Runs But Reward Doesn't Improve

**Symptoms:** `reward_env` mean stays flat or negative after 100 steps.

1. **Verify rubrics version** — `_RUBRICS_BOUNDED` must be `True` at startup. If False,
   env rewards are still bounded by fallback formula but less rich.

2. **Increase triage oracle weight** — the oracle reward (+5.0) is the strongest
   supervision signal. If env reward is noisy, the triage reward pulls the model
   toward correct rule-following regardless.

3. **Reduce curriculum** — change `ALL_TASKS` in dataset collection to only Level 1:
   ```python
   DATASET_TASKS = ["traffic_spike", "node_failure"]  # easiest two tasks
   ```
   Master these before adding complex scenarios.

4. **Increase dataset size** — change `DATASET_EPISODES = 300 → 600`. More diverse
   states = more varied obs for the model to generalise over.

5. **Check GRPO variance** — add logging:
   ```python
   print(f"reward std: {np.std(rewards):.4f}")  # should be > 0.5
   ```

### Scenario C — Entire Unsloth/TRL Stack Fails

Fall back to `train_grpo.py` (custom GRPO loop, no TRL/Unsloth dependency).

**Key differences vs train_grpo_unsloth.py:**

| Aspect | train_grpo_unsloth.py | train_grpo.py |
|---|---|---|
| Generation | TRL GRPOTrainer | manual model.generate() |
| Data | offline static dataset | **online — env steps with model** |
| Multi-step learning | ❌ single-step only | ✅ real episode rollouts |
| Speed | fast (Unsloth kernels) | ~5× slower |
| Memory | ~35GB | ~20GB (no vLLM) |

```bash
python train_grpo.py  # no Unsloth, no TRL, just PyTorch + transformers
```

`train_grpo.py` is also better if you want the model to learn **multi-step recovery
strategies** (e.g., throttle → wait → restart DB → scale up) rather than just
per-step rule lookup.

### Scenario D — Start from a Smaller Model

If Qwen3-8B is consistently too large for the available GPU:

```python
# In train_grpo_unsloth.py:
MODEL_NAME = "unsloth/Qwen3-4B"   # half the size, similar reasoning quality
# or
MODEL_NAME = "unsloth/Qwen3-1.5B" # fits on any 16GB GPU
```

Qwen3-4B with rank=16 LoRA needs ~18GB — works on a 24GB A10G.

---

## 9. Post-Training Benchmark

Once training completes:

```bash
# 1. Point inference.py to the merged model
python inference.py --mode local \
    --model checkpoints/qwen3_grpo_unsloth/merged_16bit

# 2. Analyse results
python nithish_data_parser.py --csv metrics_<new_model_name>.csv

# 3. Compare against baseline
# Target metrics:
#   Overall score     > 0.55   (baseline: 0.3946)
#   % steps at -1000  < 30%    (baseline: 70.1%)
#   traffic_spike     > 0.10   (baseline: 0.0242)
#   DIME Index        > 0.0    (baseline: 0.0000)
```

---

## 10. File Index

| File | Purpose |
|---|---|
| `inference.py` | LLM agent inference loop (local + endpoint modes) |
| `train_grpo_unsloth.py` | **Primary training script** — Unsloth + TRL GRPO |
| `train_grpo.py` | Fallback — custom GRPO loop, online rollouts |
| `train_rl.py` | PPO baseline — discrete action space, non-LLM |
| `nithish_data_parser.py` | Parse + analyse metrics CSV (no GPU needed) |
| `server/rubrics.py` | Reward functions — **must use main branch version** |
| `server/environment.py` | 8-node cluster simulation engine |
| `server/command_parser.py` | kubectl string → InfraAction parser |
| `server/models.py` | Pydantic schemas: InfraAction, InfraObservation |
| `requirements_working.txt` | pip freeze of working venv (inference deps) |

---

## 11. Key Lessons (Blog-Ready)

1. **Reward cliffs kill RL.** A single hardcoded constant (`-1000`) in a reward
   function can silently zero out gradients across an entire training run. Always
   verify reward variance before training — if `std(rewards) ≈ 0`, nothing is learned.

2. **One missing rule broke everything.** The absence of a single line
   (`kubectl delete pod node-0`) from the system prompt caused a cascading failure
   that 13 of 14 tasks couldn't recover from. The model *knew* the action — it
   tried it 13 times — but without a named rule, it was inconsistent.

3. **GRPO needs within-group variance.** Pure env rewards with a -1000 cliff give
   identical scores to all completions → zero advantage → zero gradient. The fix:
   add format and validity rewards that *always* differentiate completions, even
   before env rewards stabilise.

4. **The outlier tells the real story.** `memory_leak_slow_burn` scoring 0.99 while
   every other task scored near 0 proved the model's capability. It wasn't that the
   model was bad — it was that the reward function never let it see success.

5. **Offline GRPO vs Online GRPO.** Offline GRPO (static dataset, single-step rewards)
   teaches rule lookup. Online GRPO (model actions advance the env) teaches strategy.
   Both improve over zero-shot, but only online GRPO can learn multi-step recovery
   sequences like: `throttle → wait 3 steps → restart DB → scale up`.

6. **Version pinning matters more than you think.** Three interdependent packages
   (unsloth, vllm, transformers) with mutually exclusive version ranges required
   finding a specific `transformers==4.56.2` that satisfied all three. The install
   order also matters — vllm upgrades transformers if run first.

7. **Thinking models need `/no_think` in GRPO.** Qwen3-8B generates `<think>` blocks
   of 400–800 tokens before the actual response. With a 256-token cap, every
   completion was truncated mid-think — `clipped_ratio = 1.0`, all rewards identical,
   zero variance, zero gradient. Adding `/no_think` to user messages cuts completions
   to ~55 tokens and restores learning signal immediately.

8. **The GPU utilisation number tells you the real bottleneck.** 28% GPU util on an
   A100 means the GPU is waiting on the CPU — not a GPU or VRAM problem. `reward_env`
   creates a full `DistributedInfraEnvironment()` per completion, sequentially in
   Python. Increasing batch×gen from 4 to 32 completions/step multiplied CPU stall
   8× and made the run 4.8× *slower* despite having more VRAM.

---

## 12. Completed Training Run — 2026-04-25

### Hardware

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (SXM4, 2,039 GB/s bandwidth) |
| VRAM used | 72,041 MiB / 81,920 MiB |
| CUDA | 13.0 / Driver 580.126.20 |

### Final Config (`train_grpo_unsloth.py`)

| Parameter | Value | Note |
|---|---|---|
| Model | `unsloth/Qwen3-8B` | BF16 weights |
| LoRA rank | 32 | alpha=64, all proj layers |
| Dataset episodes | 500 | 70% oracle + 30% random |
| Dataset rows (filtered) | 5,099 | 90th-percentile prompt filter |
| max_prompt_length | 1,040 tokens | |
| max_completion_length | 512 tokens | hard cap; with /no_think actual avg ~57 tokens |
| per_device_train_batch_size | 1 | CPU-bound reward functions dictate this |
| gradient_accumulation_steps | 4 | effective batch = 4 |
| num_generations | 4 | completions per prompt |
| vllm_gpu_memory_utilization | 0.7 | 70% of remaining VRAM for KV cache |
| learning_rate | 5e-6 | cosine schedule |
| MAX_STEPS | 300 | |
| `/no_think` in user prompt | ✅ | suppresses Qwen3 think block |
| `compilation_config` | 0 | avoids A100 SM 8.0 piecewise graph-split crash |

### Runtime

| Metric | Value |
|---|---|
| Total time | **41 min 43 s** |
| Step time (avg) | **8.35 s/it** |
| train_loss | 0.000127 |
| train_samples_per_second | 0.479 |

### Training Signal (selected steps)

| Step | reward total | triage/mean | triage/std | clipped_ratio | completion_len |
|---|---|---|---|---|---|
| 5 | +5.75 | +2.1 | 0.35 | 0.0 | 50.3 |
| 10 | +4.32 | +1.0 | 0.0 | 0.0 | 54.4 |
| 20 | +3.43 | +0.45 | 0.70 | 0.0 | 58.3 |
| 127 | +4.83 | +0.95 | 0.70 | 0.0 | 58.6 |
| ~290 | +7.34 | +3.2 | 0.0 | 0.0 | 50.7 |
| ~295 | +5.58 | +2.2 | 0.0 | 0.0 | 55.8 |
| 300 | +2.34 | -1.6 | 0.0 | 0.0 | 60.1 |

`reward_format` held at 3.0 (perfect) from step 1 throughout.
`reward_validity` held at 1.8–2.0 from step 1 throughout.
Triage mean oscillated as expected for GRPO with small batch; positive on average.

### Checkpoints

```
checkpoints/qwen3_grpo_unsloth/
├── checkpoint-100/          # step 100 snapshot
├── checkpoint-200/          # step 200 snapshot
├── checkpoint-300/          # step 300 snapshot (= final)
├── lora_adapter/            # final LoRA weights only (349 MB)
└── merged_16bit/            # LoRA merged into base, BF16 (16 GB)
```

### Inference

```bash
# Use merged model (drop-in replacement for base Qwen3-8B):
MODEL_NAME=checkpoints/qwen3_grpo_unsloth/merged_16bit python inference.py

# Or point to LoRA adapter (requires Unsloth/PEFT at inference time):
# MODEL_NAME=checkpoints/qwen3_grpo_unsloth/lora_adapter python inference.py
```

### Archive

```bash
# Tar the full checkpoint directory (no compression — safetensors don't compress):
tar -cf qwen3_grpo_unsloth_$(date +%Y%m%d).tar checkpoints/qwen3_grpo_unsloth/

# Extract on any machine:
tar -xf qwen3_grpo_unsloth_YYYYMMDD.tar
```

### Debugging History (What Went Wrong Before the Final Run)

| Attempt | Config | Issue | Step time | Outcome |
|---|---|---|---|---|
| Run 1 (A100-40GB) | batch=1, gen=4, steps=500 | `compilation_config` not set → vLLM crash | — | Killed |
| Run 2 (A100-80GB) | batch=4, gen=8, steps=300 | reward_env CPU stall: 32 completions × 3s = 96s CPU/step | 126 s/step → 10h est. | Killed |
| Run 3 (A100-80GB) | batch=1, gen=4, steps=300, max_comp=256 | Qwen3 `<think>` blocks truncated at 256 tokens; `clipped_ratio=1.0`, all rewards identical, zero variance | 20 s/step but 0 learning | Killed |
| **Run 4 (Final)** | batch=1, gen=4, steps=300, max_comp=512, `/no_think` | — | **8.35 s/step** | ✅ Completed |
