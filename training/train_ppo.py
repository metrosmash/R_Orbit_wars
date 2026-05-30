"""
train_ppo.py
============
Phase 2: PPO (Proximal Policy Optimisation) RL fine-tuning.

WHAT IS PPO? (Plain English)
-----------------------------
PPO is like giving the bot feedback after each game:
  "That move increased your ship count — do more of that."
  "That move got your planet captured — do less of that."

It does this carefully so it doesn't change the strategy too drastically in
one update (the "proximal" = "nearby" part). This stability is why PPO is
the most popular RL algorithm for game AI.

KEY CONCEPTS
------------
  Trajectory   : a sequence of (state, action, reward) from one episode.
  Advantage    : "was this action better or worse than expected?"
                 A = actual_return - value_estimate
                 Positive advantage → do this more often.
                 Negative advantage → do this less often.
  Clip ratio   : PPO clips the policy update to stay within [1-ε, 1+ε]
                 of the old policy. This is the safety mechanism.
  Value loss   : keeps the Critic's estimates accurate.
  Entropy bonus: encourages exploration — prevents the bot from getting
                 stuck in one strategy too early.

WORKFLOW
--------
1. Collect N steps of experience by running the bot against the opponent.
2. Compute advantages using GAE (Generalised Advantage Estimation).
3. Run K epochs of mini-batch updates on the collected experience.
4. Repeat from step 1.

USAGE
-----
  # Start from scratch (no IL checkpoint):
  python train_ppo.py

  # Start from IL checkpoint (RECOMMENDED):
  python train_ppo.py --init_checkpoint ./checkpoints/il_best.pt

  # Resume a PPO run:
  python train_ppo.py --init_checkpoint ./checkpoints/ppo_latest.pt
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from collections import deque

# Allow imports from agent/ whether run from repo root or training/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AGENT_DIR = os.path.join(_REPO_ROOT, "agent")
for _p in [_AGENT_DIR, _REPO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model        import OrbitWarsNet, save_model, load_model
from env_wrapper  import OrbitWarsEnv
from feature_utils import FEATURE_DIM, MAX_PLANETS
from action_utils  import MAX_MY_PLANETS, NUM_ACTIONS_PER_PLANET, decode_action

# ── PPO Hyperparameters ───────────────────────────────────────────────────────
# These are sensible defaults for this game — feel free to tune.

DEFAULTS = dict(
    # Environment
    n_envs          = 8,          # parallel environments (uses multiprocessing)
    opponent        = "hardcoded",# "hardcoded" | "random" | "self"

    # Rollout
    n_steps         = 256,        # steps per env per rollout
    total_timesteps = 5_000_000,  # total environment steps to train

    # PPO
    gamma           = 0.995,      # discount factor (higher = care more about future)
    gae_lambda      = 0.95,       # GAE smoothing (0 = TD, 1 = Monte Carlo)
    clip_eps        = 0.2,        # PPO clip range
    value_coef      = 0.5,        # value loss weight
    entropy_coef    = 0.01,       # entropy bonus weight
    max_grad_norm   = 0.5,

    # Optimiser
    lr              = 2.5e-4,
    n_epochs        = 4,          # PPO update epochs per rollout
    batch_size      = 256,        # mini-batch size for PPO updates

    # Model
    hidden_dim      = 512,
    num_layers      = 3,

    # Saving
    save_every      = 50_000,     # save checkpoint every N timesteps
    eval_every      = 25_000,     # evaluate vs hardcoded_bot every N timesteps
    out_dir         = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"),
    init_checkpoint = None,
)


# ─────────────────────────────────────────────────────────────────────────────
# RolloutBuffer
# Stores experience from multiple parallel environments.
# ─────────────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, n_steps, n_envs, device):
        self.n_steps = n_steps
        self.n_envs  = n_envs
        self.device  = device
        self.reset()

    def reset(self):
        T, E = self.n_steps, self.n_envs
        self.obs      = torch.zeros(T, E, FEATURE_DIM,     device=self.device)
        self.actions  = torch.zeros(T, E, MAX_MY_PLANETS,  device=self.device, dtype=torch.long)
        self.rewards  = torch.zeros(T, E,                  device=self.device)
        self.dones    = torch.zeros(T, E,                  device=self.device)
        self.values   = torch.zeros(T, E,                  device=self.device)
        self.log_probs= torch.zeros(T, E, MAX_MY_PLANETS,  device=self.device)
        self.pos      = 0

    def add(self, obs, actions, rewards, dones, values, log_probs):
        self.obs[self.pos]       = obs
        self.actions[self.pos]   = actions
        self.rewards[self.pos]   = rewards
        self.dones[self.pos]     = dones
        self.values[self.pos]    = values
        self.log_probs[self.pos] = log_probs
        self.pos += 1

    def compute_returns_and_advantages(self, last_values, last_dones, gamma, gae_lambda):
        """GAE advantage estimation."""
        T, E = self.n_steps, self.n_envs
        advantages = torch.zeros_like(self.rewards)
        last_gae   = torch.zeros(E, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - last_dones.float()
                next_values       = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values       = self.values[t + 1]

            delta     = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae  = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.returns    = advantages + self.values
        self.advantages = advantages

    def get_batches(self, batch_size):
        """Yield mini-batches for PPO update."""
        T, E   = self.n_steps, self.n_envs
        n_data = T * E

        # Flatten time and env dimensions
        obs       = self.obs.view(n_data, FEATURE_DIM)
        actions   = self.actions.view(n_data, MAX_MY_PLANETS)
        log_probs = self.log_probs.view(n_data, MAX_MY_PLANETS)
        returns   = self.returns.view(n_data)
        advantages= self.advantages.view(n_data)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = torch.randperm(n_data, device=self.device)
        for start in range(0, n_data, batch_size):
            idx = indices[start: start + batch_size]
            yield (obs[idx], actions[idx], log_probs[idx],
                   returns[idx], advantages[idx])


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_vs_hardcoded
# Run N games against hardcoded_bot, return win rate.
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_vs_hardcoded(model, n_games=10, device="cpu"):
    model.eval()
    wins = 0

    for game_i in range(n_games):
        env = OrbitWarsEnv(opponent="hardcoded", seed=game_i * 7)
        obs, _ = env.reset()
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                actions, _ = model.greedy_actions(obs_t)
            action_np = actions[0].cpu().numpy()
            obs, reward, done, truncated, info = env.step(action_np)
            if truncated:
                done = True

        # Check reward from last step — positive = win
        if reward > 0:
            wins += 1
        env.close()

    win_rate = wins / n_games
    model.train()
    return win_rate


# ─────────────────────────────────────────────────────────────────────────────
# VecEnv — simple sequential multi-env wrapper
# (For true parallelism use SubprocVecEnv from stable-baselines3)
# ─────────────────────────────────────────────────────────────────────────────
class SimpleVecEnv:
    """Run multiple envs sequentially. Simple but works for getting started."""
    def __init__(self, n_envs, opponent="hardcoded"):
        self.envs = [OrbitWarsEnv(opponent=opponent, seed=i) for i in range(n_envs)]
        self.n_envs = n_envs

    def reset(self):
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return np.stack(obs_list)   # (n_envs, FEATURE_DIM)

    def step(self, actions):
        """actions: (n_envs, MAX_MY_PLANETS)"""
        obs_list, rew_list, done_list = [], [], []
        for i, env in enumerate(self.envs):
            obs, rew, done, trunc, _ = env.step(actions[i])
            if done or trunc:
                obs, _ = env.reset()
                done = True
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(float(done))
        return (np.stack(obs_list),
                np.array(rew_list, dtype=np.float32),
                np.array(done_list, dtype=np.float32))

    def close(self):
        for env in self.envs:
            env.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PPO TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_ppo(cfg):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[PPO] Device         : {device}")
    if device == "cuda":
        print(f"[PPO] GPU            : {torch.cuda.get_device_name(0)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    if cfg["init_checkpoint"] and os.path.exists(cfg["init_checkpoint"]):
        print(f"[PPO] Loading IL checkpoint: {cfg['init_checkpoint']}")
        model, meta = load_model(
            cfg["init_checkpoint"], device=device,
            hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"]
        )
        print(f"[PPO] Loaded metadata: {meta}")
    else:
        print("[PPO] Starting from scratch (no IL checkpoint)")
        model = OrbitWarsNet(
            hidden_dim    = cfg["hidden_dim"],
            num_layers    = cfg["num_layers"],
            dropout       = 0.0,   # disable dropout during RL rollouts
        ).to(device)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-5)
    use_amp   = (device == "cuda")
    scaler    = GradScaler(enabled=use_amp)

    # ── Environments ──────────────────────────────────────────────────────────
    print(f"[PPO] Creating {cfg['n_envs']} environments (opponent={cfg['opponent']})...")
    vec_env = SimpleVecEnv(cfg["n_envs"], opponent=cfg["opponent"])

    # ── Rollout buffer ─────────────────────────────────────────────────────────
    buffer = RolloutBuffer(cfg["n_steps"], cfg["n_envs"], device)

    # ── Tracking ──────────────────────────────────────────────────────────────
    total_steps   = 0
    n_updates     = 0
    ep_rewards    = deque(maxlen=100)
    best_win_rate = 0.0

    obs_np = vec_env.reset()   # (n_envs, FEATURE_DIM)
    obs_t  = torch.FloatTensor(obs_np).to(device)

    print(f"[PPO] Starting training  (total_timesteps={cfg['total_timesteps']:,})\n")
    t_start = time.time()

    while total_steps < cfg["total_timesteps"]:

        # ────────────────────────────────────────────────────────────────────
        # COLLECT ROLLOUT
        # ────────────────────────────────────────────────────────────────────
        buffer.reset()
        ep_reward_buf = np.zeros(cfg["n_envs"])

        for step in range(cfg["n_steps"]):
            with torch.no_grad():
                actions_t, log_probs_t, entropy_t, values_t = model.sample_actions(obs_t)
                # actions_t  : (n_envs, MAX_MY_PLANETS)
                # values_t   : (n_envs, 1)

            actions_np = actions_t.cpu().numpy()   # (n_envs, MAX_MY_PLANETS)

            obs_np_next, rewards_np, dones_np = vec_env.step(actions_np)

            buffer.add(
                obs_t,
                actions_t,
                torch.FloatTensor(rewards_np).to(device),
                torch.FloatTensor(dones_np).to(device),
                values_t.squeeze(-1),
                log_probs_t,
            )

            ep_reward_buf += rewards_np
            for i, d in enumerate(dones_np):
                if d:
                    ep_rewards.append(ep_reward_buf[i])
                    ep_reward_buf[i] = 0.0

            obs_t = torch.FloatTensor(obs_np_next).to(device)
            total_steps += cfg["n_envs"]

        # Compute advantages
        with torch.no_grad():
            _, last_values = model.greedy_actions(obs_t)
        buffer.compute_returns_and_advantages(
            last_values.squeeze(-1),
            torch.FloatTensor(dones_np).to(device),
            cfg["gamma"], cfg["gae_lambda"]
        )

        # ────────────────────────────────────────────────────────────────────
        # PPO UPDATE
        # ────────────────────────────────────────────────────────────────────
        pg_losses, v_losses, ent_losses, clip_fracs = [], [], [], []

        for _ in range(cfg["n_epochs"]):
            for (mb_obs, mb_actions, mb_old_logp,
                 mb_returns, mb_advantages) in buffer.get_batches(cfg["batch_size"]):

                optimizer.zero_grad()

                with autocast(enabled=use_amp):
                    new_logp, entropy, new_values = model.evaluate_actions(
                        mb_obs, mb_actions
                    )
                    # new_logp : (B, MAX_MY_PLANETS)
                    # Aggregate log probs across planet slots (sum = joint prob)
                    new_logp_sum = new_logp.sum(dim=-1)          # (B,)
                    old_logp_sum = mb_old_logp.sum(dim=-1)       # (B,)

                    # PPO ratio and clipped surrogate
                    ratio = torch.exp(new_logp_sum - old_logp_sum)
                    clip_frac = ((ratio - 1.0).abs() > cfg["clip_eps"]).float().mean()

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * ratio.clamp(
                        1.0 - cfg["clip_eps"], 1.0 + cfg["clip_eps"]
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (clipped)
                    v_loss = 0.5 * ((new_values.squeeze(-1) - mb_returns) ** 2).mean()

                    # Entropy bonus (encourage exploration)
                    ent_loss = -entropy.mean()

                    loss = (pg_loss
                            + cfg["value_coef"]   * v_loss
                            + cfg["entropy_coef"] * ent_loss)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())
                clip_fracs.append(clip_frac.item())

        n_updates += 1

        # ────────────────────────────────────────────────────────────────────
        # LOGGING
        # ────────────────────────────────────────────────────────────────────
        if n_updates % 5 == 0:
            elapsed  = time.time() - t_start
            fps      = total_steps / elapsed
            mean_rew = np.mean(ep_rewards) if ep_rewards else 0.0
            print(
                f"steps {total_steps:>8,} | "
                f"updates {n_updates:>5} | "
                f"pg {np.mean(pg_losses):.4f} | "
                f"v {np.mean(v_losses):.4f} | "
                f"ent {np.mean(ent_losses):.4f} | "
                f"clip {np.mean(clip_fracs):.3f} | "
                f"rew {mean_rew:.2f} | "
                f"fps {fps:.0f}"
            )

        # ────────────────────────────────────────────────────────────────────
        # PERIODIC CHECKPOINT
        # ────────────────────────────────────────────────────────────────────
        if total_steps % cfg["save_every"] < cfg["n_envs"] * cfg["n_steps"]:
            ckpt = os.path.join(cfg["out_dir"], f"ppo_{total_steps}.pt")
            save_model(model, ckpt, metadata={"total_steps": total_steps})
            # Also save a "latest" that main.py always loads
            save_model(model, os.path.join(cfg["out_dir"], "ppo_latest.pt"),
                       metadata={"total_steps": total_steps})

        # ────────────────────────────────────────────────────────────────────
        # EVALUATION
        # ────────────────────────────────────────────────────────────────────
        if total_steps % cfg["eval_every"] < cfg["n_envs"] * cfg["n_steps"]:
            try:
                win_rate = evaluate_vs_hardcoded(model, n_games=10, device=device)
                print(f"  ★ Eval vs hardcoded: win_rate={win_rate:.1%}")
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    save_model(model, os.path.join(cfg["out_dir"], "ppo_best.pt"),
                               metadata={"total_steps": total_steps, "win_rate": win_rate})
                    print(f"  ★ New best model saved (win_rate={win_rate:.1%})")
            except Exception as e:
                print(f"  [eval failed: {e}]")

    # Final save
    save_model(model, os.path.join(cfg["out_dir"], "ppo_final.pt"),
               metadata={"total_steps": total_steps, "final": True})
    vec_env.close()

    print(f"\n[PPO] Training complete. Total steps: {total_steps:,}")
    print(f"[PPO] Best win rate vs hardcoded: {best_win_rate:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training for Orbit Wars")
    for k, v in DEFAULTS.items():
        t = type(v) if v is not None else str
        parser.add_argument(f"--{k}", type=t, default=v)
    args = parser.parse_args()
    cfg  = vars(args)
    train_ppo(cfg)
