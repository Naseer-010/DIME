#!/usr/bin/env python3
"""
PPO RL training script for DIME.

Why inference.py is not suitable for RL training:
  1. ProductionSREReward clips to -1000 at any moderate DB CPU (c_db=100,
     lambda_db=4 → penalty >1000 when db_cpu ≥ 0.57).  Every step gives -1000
     regardless of action; gradient signal is zero.
  2. No DB-restart rule in the system prompt → unrecoverable after DB failure.
  3. Reward computed from post-dynamics state, obs built from pre-dynamics state.
  4. HTTP round-trips make the inner loop 100x slower than direct Python calls.
  5. Raw text generation + regex parsing adds noise that masks policy quality.

This file:
  - Wraps DistributedInfraEnvironment directly in a Gymnasium interface.
  - Uses the composable rubric reward (clipped to [-10, 10], not always -1000).
  - Encodes InfraAction as a flat discrete integer space (no text generation).
  - Trains a small MLP actor-critic with PPO.
  - Uses curriculum: easy tasks first, harder tasks after performance threshold.
"""

import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces

from server.environment import DistributedInfraEnvironment
from server.models import InfraAction, InfraObservation
from server.rubrics import compute_composite_reward

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_NODES = 8          # permanent node count
OBS_DIM = N_NODES * 5 + 7  # cpu, mem, queue(norm), failed, dropout + 7 scalars = 47

# Action index → InfraAction mapping (built in decode_action)
# 0         no_op
# 1..8      restart_node(0..7)
# 9         scale_up
# 10        throttle(0.3)
# 11        throttle(0.5)
# 12        throttle(0.7)
# 13..19    reroute worker i → least-loaded healthy worker (dynamic destination)
#           i ∈ {1,2,3,4,5,6,7}
N_ACTIONS = 20

CURRICULUM: List[List[str]] = [
    ["traffic_spike", "node_failure"],
    ["cascading_failure", "flash_crowd"],
    ["thundering_herd", "zombie_node", "hot_shard_skew"],
    ["memory_leak_slow_burn", "split_brain_io_bottleneck",
     "black_swan_az_failure", "retry_storm",
     "connection_pool_deadlock", "autoscaler_flapping_trap"],
]
# Mean composite reward threshold to advance to the next curriculum level
CURRICULUM_ADVANCE_THRESHOLD = -1.5


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------

def encode_obs(obs: InfraObservation) -> np.ndarray:
    """
    Convert InfraObservation to a fixed 47-dim float32 vector.

    Dropout values (-1.0 / -1) are zeroed out; a separate binary mask tracks
    which nodes had dropout so the policy can choose query_logs intelligently.
    The vector is always N_NODES-sized regardless of temporary scale-up nodes.
    """
    cpu = np.zeros(N_NODES, dtype=np.float32)
    mem = np.zeros(N_NODES, dtype=np.float32)
    queue = np.zeros(N_NODES, dtype=np.float32)
    failed = np.zeros(N_NODES, dtype=np.float32)
    mask = np.zeros(N_NODES, dtype=np.float32)  # 1 = telemetry dropped

    for i in range(min(N_NODES, len(obs.cpu_loads))):
        c = obs.cpu_loads[i]
        if c < 0:
            mask[i] = 1.0
        else:
            cpu[i] = float(c)

    for i in range(min(N_NODES, len(obs.mem_utilizations))):
        m = obs.mem_utilizations[i]
        if m < 0:
            mask[i] = 1.0
        else:
            mem[i] = float(m)

    for i in range(min(N_NODES, len(obs.queue_lengths))):
        q = obs.queue_lengths[i]
        if q < 0:
            pass
        else:
            queue[i] = min(float(q) / 500.0, 1.0)

    for idx in obs.failed_nodes:
        if 0 <= idx < N_NODES:
            failed[idx] = 1.0

    scalars = np.array([
        min(obs.latency_ms / 500.0, 1.0),
        min(obs.request_rate / 500.0, 1.0),
        min(obs.p99_latency / 1000.0, 1.0),
        float(obs.io_wait),
        obs.error_budget / 100.0,
        obs.cloud_budget / 10.0,
        obs.step / 30.0,
    ], dtype=np.float32)

    return np.concatenate([cpu, mem, queue, failed, mask, scalars])


# ---------------------------------------------------------------------------
# Action decoding
# ---------------------------------------------------------------------------

def decode_action(action_idx: int, obs: InfraObservation) -> InfraAction:
    """
    Map a flat integer action to a typed InfraAction.

    The reroute actions (13-19) dynamically pick the least-loaded healthy
    destination node at decision time, avoiding stale hardcoded pairs.
    """
    if action_idx == 0:
        return InfraAction(action_type="no_op")

    if 1 <= action_idx <= 8:
        return InfraAction(action_type="restart_node", target=action_idx - 1)

    if action_idx == 9:
        return InfraAction(action_type="scale_up")

    throttle_map = {10: 0.3, 11: 0.5, 12: 0.7}
    if action_idx in throttle_map:
        return InfraAction(action_type="throttle", rate=throttle_map[action_idx])

    # Actions 13-19: reroute from worker (action_idx - 12) to least-loaded healthy worker
    from_node = action_idx - 12  # 1..7
    failed_set = set(obs.failed_nodes)

    # Find least-loaded healthy destination ≠ from_node
    best_dst = None
    best_cpu = float("inf")
    for i, c in enumerate(obs.cpu_loads[:N_NODES]):
        if i == 0 or i == from_node or i in failed_set:
            continue
        cpu_val = c if c >= 0 else 1.0  # treat dropout as high load
        if cpu_val < best_cpu:
            best_cpu = cpu_val
            best_dst = i

    if best_dst is None:
        return InfraAction(action_type="no_op")

    return InfraAction(action_type="reroute_traffic", from_node=from_node, to_node=best_dst)


# ---------------------------------------------------------------------------
# Gymnasium environment wrapper
# ---------------------------------------------------------------------------

class DIMEEnv(gym.Env):
    """
    Gymnasium wrapper around DistributedInfraEnvironment.

    Key differences from inference.py:
    - Uses structured InfraAction directly (no text, no HTTP).
    - Reward comes from compute_composite_reward (composable rubric, bounded
      [-10, 10]) rather than ProductionSREReward (always clips to -1000).
    - Extra -5 terminal penalty when DB node (0) is failed.
    """

    metadata = {"render_modes": []}

    def __init__(self, tasks: List[str], seed: int = 0):
        super().__init__()
        self._tasks = tasks
        self._rng = random.Random(seed)
        self._env = DistributedInfraEnvironment()
        self._last_obs: Optional[InfraObservation] = None

        self.observation_space = spaces.Box(
            low=-0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

    def reset(self, *, seed=None, options=None):
        task = self._rng.choice(self._tasks)
        obs = self._env.reset(task=task)
        self._last_obs = obs
        return encode_obs(obs), {"task": task}

    def step(self, action: int):
        assert self._last_obs is not None, "call reset() before step()"
        infra_action = decode_action(action, self._last_obs)

        obs = self._env.step(infra_action)
        self._last_obs = obs

        reward = self._compute_reward()
        terminated = bool(obs.done)
        truncated = False

        return encode_obs(obs), reward, terminated, truncated, {}

    def _compute_reward(self) -> float:
        """
        Custom reward with clear gradient signal in [-3, +1].

        Why not use the existing rubrics directly:
        - ProductionSREReward: clips to -1000 at any moderate DB CPU (c_db=100,
          lambda_db=4 → penalty >1000 when db_cpu ≥ 0.57).
        - compute_composite_reward: SafeSliceLatencyVerifier uses beta=5 * lat,
          which clips the composite to -10 at any latency > ~25 ms.
        Both give constant floor signal, destroying gradient information.

        This reward decomposes into three well-scaled components:
          +uptime:   fraction of nodes alive (0→1, max +0.5)
          -latency:  quadratic penalty above 50ms SLA (0 at 50ms, -0.5 at 150ms)
          -db_fail:  hard -2.0 when DB (node-0) is failed (clear recovery signal)
        """
        sim = self._env.sim
        nodes = sim.nodes

        # Uptime component
        alive = sum(1 for n in nodes if not n.is_failed)
        total = max(len(nodes), 1)
        uptime_reward = 0.5 * (alive / total)

        # Latency component: 0 below SLA, grows quadratically above
        lat = sim.latency_ms
        excess = max(0.0, lat - 50.0) / 100.0          # 0 at 50ms, 1 at 150ms
        latency_penalty = 0.5 * min(excess ** 2, 1.0)  # cap at -0.5

        # DB survival: hard signal so the policy learns to protect/restart node-0
        db_failed = bool(nodes and nodes[0].is_failed)
        db_penalty = 2.0 if db_failed else 0.0

        return float(uptime_reward - latency_penalty - db_penalty)


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared-trunk MLP with separate policy head and value head.

    Architecture note: a feed-forward policy works for full observability.
    With heavy telemetry dropout (5% per node per step) an LSTM trunk would
    help; set use_lstm=True to enable it (increases training time ~3×).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128,
                 use_lstm: bool = False):
        super().__init__()
        self.use_lstm = use_lstm

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        if use_lstm:
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
            self._hidden = None

        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

        # Orthogonal init recommended for PPO
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def reset_hidden(self):
        self._hidden = None

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)

        if self.use_lstm:
            h = h.unsqueeze(1)
            h, self._hidden = self.lstm(h, self._hidden)
            h = h.squeeze(1)

        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None):
        logits, value = self(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# PPO rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    # Rollout
    n_steps: int = 512         # steps per environment per update
    n_envs: int = 4            # parallel environments

    # Optimisation
    lr: float = 3e-4
    n_epochs: int = 4          # gradient epochs per PPO update
    batch_size: int = 64       # mini-batch size
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Training budget
    total_timesteps: int = 500_000
    log_interval: int = 10     # log every N updates

    # Curriculum
    curriculum_window: int = 50  # episodes averaged for curriculum advancement

    # Model
    hidden_dim: int = 128
    use_lstm: bool = False

    # Save
    save_path: str = "checkpoints/dime_ppo.pt"


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    """Generalised Advantage Estimation (Schulman et al. 2016)."""
    advantages = []
    returns = []
    gae = 0.0
    next_value = last_value

    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
        next_value = values[t]

    return advantages, returns


class PPOTrainer:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PPO] Device: {self.device}")

        self.policy = ActorCritic(
            OBS_DIM, N_ACTIONS, hidden=cfg.hidden_dim, use_lstm=cfg.use_lstm
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=1e-5)

        self._curriculum_level = 0
        self._episode_rewards: deque = deque(maxlen=cfg.curriculum_window)
        self._global_step = 0

    def _make_envs(self) -> List[DIMEEnv]:
        tasks = CURRICULUM[self._curriculum_level]
        return [DIMEEnv(tasks, seed=i) for i in range(self.cfg.n_envs)]

    def _try_advance_curriculum(self):
        if (
            self._curriculum_level < len(CURRICULUM) - 1
            and len(self._episode_rewards) == self.cfg.curriculum_window
        ):
            mean_r = float(np.mean(self._episode_rewards))
            if mean_r >= CURRICULUM_ADVANCE_THRESHOLD:
                self._curriculum_level += 1
                self._episode_rewards.clear()
                print(
                    f"\n[CURRICULUM] Advanced to level {self._curriculum_level + 1}"
                    f" | Tasks: {CURRICULUM[self._curriculum_level]}\n"
                )

    def train(self):
        cfg = self.cfg
        os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)

        envs = self._make_envs()
        obs_list = [env.reset()[0] for env in envs]
        buffers = [RolloutBuffer() for _ in envs]
        ep_rewards = [0.0] * cfg.n_envs
        update_count = 0

        print(f"[PPO] Starting training | {cfg.total_timesteps} timesteps "
              f"| {cfg.n_envs} envs | curriculum level 1")
        print(f"[PPO] Tasks: {CURRICULUM[self._curriculum_level]}\n")

        while self._global_step < cfg.total_timesteps:
            # ---- Collect rollout ----
            self.policy.eval()
            for _ in range(cfg.n_steps):
                obs_tensor = torch.tensor(
                    np.stack(obs_list), dtype=torch.float32, device=self.device
                )

                with torch.no_grad():
                    actions, log_probs, _, values = self.policy.get_action_and_value(obs_tensor)

                for env_idx, env in enumerate(envs):
                    a = int(actions[env_idx].item())
                    next_obs, reward, terminated, truncated, _ = env.step(a)
                    done = terminated or truncated

                    buffers[env_idx].obs.append(obs_list[env_idx].copy())
                    buffers[env_idx].actions.append(a)
                    buffers[env_idx].log_probs.append(float(log_probs[env_idx].item()))
                    buffers[env_idx].rewards.append(float(reward))
                    buffers[env_idx].dones.append(done)
                    buffers[env_idx].values.append(float(values[env_idx].item()))

                    ep_rewards[env_idx] += reward

                    if done:
                        self._episode_rewards.append(ep_rewards[env_idx])
                        ep_rewards[env_idx] = 0.0
                        next_obs, _ = env.reset()

                    obs_list[env_idx] = next_obs
                    self._global_step += 1

            # ---- Compute advantages ----
            all_obs, all_actions, all_log_probs = [], [], []
            all_advantages, all_returns = [], []

            last_obs_tensor = torch.tensor(
                np.stack(obs_list), dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                _, last_values = self.policy(last_obs_tensor)

            for env_idx in range(cfg.n_envs):
                buf = buffers[env_idx]
                last_val = float(last_values[env_idx].item())

                advs, rets = compute_gae(
                    buf.rewards, buf.values, buf.dones,
                    last_val, cfg.gamma, cfg.gae_lambda
                )
                all_obs.extend(buf.obs)
                all_actions.extend(buf.actions)
                all_log_probs.extend(buf.log_probs)
                all_advantages.extend(advs)
                all_returns.extend(rets)
                buf.clear()

            # ---- PPO update ----
            self.policy.train()
            obs_t = torch.tensor(np.array(all_obs), dtype=torch.float32, device=self.device)
            act_t = torch.tensor(all_actions, dtype=torch.long, device=self.device)
            old_lp_t = torch.tensor(all_log_probs, dtype=torch.float32, device=self.device)
            adv_t = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)
            ret_t = torch.tensor(all_returns, dtype=torch.float32, device=self.device)

            # Normalise advantages per mini-batch update
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            n_samples = len(all_obs)
            indices = np.arange(n_samples)

            pg_losses, vf_losses, ent_losses = [], [], []

            for _ in range(cfg.n_epochs):
                np.random.shuffle(indices)
                for start in range(0, n_samples, cfg.batch_size):
                    mb = indices[start: start + cfg.batch_size]
                    mb_t = torch.tensor(mb, dtype=torch.long, device=self.device)

                    _, new_lp, entropy, new_val = self.policy.get_action_and_value(
                        obs_t[mb_t], act_t[mb_t]
                    )

                    ratio = torch.exp(new_lp - old_lp_t[mb_t])
                    clip_ratio = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)

                    pg_loss = -torch.min(
                        ratio * adv_t[mb_t], clip_ratio * adv_t[mb_t]
                    ).mean()
                    vf_loss = 0.5 * (new_val - ret_t[mb_t]).pow(2).mean()
                    ent_loss = -entropy.mean()

                    loss = pg_loss + cfg.vf_coef * vf_loss + cfg.ent_coef * ent_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()

                    pg_losses.append(pg_loss.item())
                    vf_losses.append(vf_loss.item())
                    ent_losses.append(ent_loss.item())

            update_count += 1
            self._try_advance_curriculum()

            if update_count % cfg.log_interval == 0:
                mean_ep = (
                    float(np.mean(self._episode_rewards))
                    if self._episode_rewards else float("nan")
                )
                print(
                    f"[{self._global_step:>7d}] update={update_count:>4d} | "
                    f"ep_rew={mean_ep:+.3f} | "
                    f"pg={np.mean(pg_losses):.4f} | "
                    f"vf={np.mean(vf_losses):.4f} | "
                    f"ent={np.mean(ent_losses):.4f} | "
                    f"curriculum={self._curriculum_level + 1}"
                )

        # Final save
        torch.save(
            {"policy_state": self.policy.state_dict(), "step": self._global_step},
            cfg.save_path,
        )
        print(f"\n[PPO] Training done. Model saved to {cfg.save_path}")

        # Close envs
        for env in envs:
            env.close()


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(checkpoint_path: str, tasks: List[str], n_episodes: int = 20) -> Dict:
    """Run greedy rollouts and report per-task mean composite reward."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ActorCritic(OBS_DIM, N_ACTIONS).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()

    results: Dict[str, List[float]] = {t: [] for t in tasks}

    for task in tasks:
        env = DIMEEnv([task], seed=42)
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = policy(obs_t)
                action = int(logits.argmax(dim=-1).item())
                obs, r, terminated, truncated, _ = env.step(action)
                ep_reward += r
                done = terminated or truncated
            results[task].append(ep_reward)
        env.close()

    summary = {t: float(np.mean(v)) for t, v in results.items()}
    overall = float(np.mean(list(summary.values())))
    summary["overall"] = overall
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = PPOConfig(
        n_steps=512,
        n_envs=4,
        lr=3e-4,
        n_epochs=4,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        total_timesteps=500_000,
        log_interval=10,
        curriculum_window=50,
        hidden_dim=128,
        use_lstm=False,       # flip to True if partial-obs hurts training
        save_path="checkpoints/dime_ppo.pt",
    )

    trainer = PPOTrainer(cfg)
    trainer.train()

    print("\n[EVAL] Evaluating on all tasks...")
    all_tasks = [t for level in CURRICULUM for t in level]
    results = evaluate(cfg.save_path, all_tasks, n_episodes=10)
    print("\n[EVAL] Results:")
    for task, score in results.items():
        print(f"  {task:<35} {score:+.3f}")


if __name__ == "__main__":
    main()
