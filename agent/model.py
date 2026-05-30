"""
model.py
========
The neural network brain for our Orbit Wars agent.

ARCHITECTURE: Actor-Critic (used by PPO)
-----------------------------------------
Two heads share one backbone:

  Observation (618,)
       │
  [Backbone: 3 × Linear + LayerNorm + ReLU]
       │
       ├──► Actor head  → per-planet action logits
       │    Shape: (MAX_MY_PLANETS, NUM_ACTIONS_PER_PLANET)
       │    Softmax over NUM_ACTIONS_PER_PLANET gives a probability
       │    distribution of what to do with each planet.
       │
       └──► Critic head → scalar state value  V(s)
            Shape: (1,)
            Estimates "how good is this position" — used to compute
            advantages during PPO training.

WHY ACTOR-CRITIC?
-----------------
Think of it as two employees:
- The Actor  decides WHAT to do (picks moves).
- The Critic evaluates HOW GOOD the current situation is.
The Critic's feedback helps the Actor improve faster than pure trial-and-error.

WHY LAYERNORM?
--------------
Our features have very different scales (ship counts, angles, binary flags).
LayerNorm normalises within each sample so no single feature dominates.
BatchNorm requires large batches and doesn't work well with variable-length
episodes — LayerNorm doesn't have that problem.

IMITATION LEARNING vs RL
------------------------
This same model is used for both:
- IL phase : train with cross-entropy loss on (state → expert_action) pairs.
             Only the Actor head and backbone are updated.
- RL phase : PPO updates both Actor and Critic using advantage estimates.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from feature_utils import FEATURE_DIM, MAX_PLANETS
from action_utils   import NUM_ACTIONS_PER_PLANET, NUM_TARGETS, NUM_RATIOS

# How many of our own planets the network makes decisions for each turn.
# Matches the max number of planets we can own (bounded by MAX_PLANETS).
MAX_MY_PLANETS = 10   # practical ceiling; padded with zeros if fewer


# ─────────────────────────────────────────────────────────────────────────────
# MLP block: Linear → LayerNorm → ReLU (with optional dropout)
# ─────────────────────────────────────────────────────────────────────────────
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)
        self.drop   = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(F.relu(self.norm(self.linear(x))))


# ─────────────────────────────────────────────────────────────────────────────
# OrbitWarsNet — the full Actor-Critic network
# ─────────────────────────────────────────────────────────────────────────────
class OrbitWarsNet(nn.Module):
    """
    Parameters
    ----------
    hidden_dim   : width of each hidden layer (default 512)
    num_layers   : depth of the shared backbone (default 3)
    dropout      : dropout rate in backbone (0 = disabled, good for inference)
    max_my_planets: max planets we make decisions for (padded if fewer)
    """

    def __init__(
        self,
        hidden_dim    = 512,
        num_layers    = 3,
        dropout       = 0.1,
        max_my_planets= MAX_MY_PLANETS,
    ):
        super().__init__()
        self.max_my_planets = max_my_planets

        # ── Shared backbone ───────────────────────────────────────────────────
        layers = [MLPBlock(FEATURE_DIM, hidden_dim, dropout)]
        for _ in range(num_layers - 1):
            layers.append(MLPBlock(hidden_dim, hidden_dim, dropout))
        self.backbone = nn.Sequential(*layers)

        # ── Actor head ────────────────────────────────────────────────────────
        # Outputs logits for every (planet_slot × action) combination.
        # Shape: (batch, max_my_planets * NUM_ACTIONS_PER_PLANET)
        self.actor = nn.Linear(hidden_dim, max_my_planets * NUM_ACTIONS_PER_PLANET)

        # ── Critic head ───────────────────────────────────────────────────────
        self.critic = nn.Linear(hidden_dim, 1)

        # ── Weight initialisation ─────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        # Actor output layer — smaller init for stable early training
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs_features, action_mask=None):
        """
        Parameters
        ----------
        obs_features : Tensor (batch, FEATURE_DIM)
        action_mask  : Tensor (batch, max_my_planets, NUM_ACTIONS_PER_PLANET)
                       bool — True = valid action. Optional.

        Returns
        -------
        logits : Tensor (batch, max_my_planets, NUM_ACTIONS_PER_PLANET)
        value  : Tensor (batch, 1)
        """
        batch = obs_features.shape[0]

        h = self.backbone(obs_features)   # (batch, hidden_dim)

        # Actor
        raw_logits = self.actor(h)        # (batch, max_my_planets * NUM_ACTIONS)
        logits = raw_logits.view(batch, self.max_my_planets, NUM_ACTIONS_PER_PLANET)

        # Apply mask: set invalid actions to -1e9 so softmax → ~0 probability
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        # Critic
        value = self.critic(h)            # (batch, 1)

        return logits, value

    # ── Convenience: get action probabilities ─────────────────────────────────
    def action_probs(self, obs_features, action_mask=None):
        """Returns softmax probabilities — shape (batch, max_my, NUM_ACTIONS)."""
        logits, value = self.forward(obs_features, action_mask)
        return F.softmax(logits, dim=-1), value

    # ── Convenience: sample actions stochastically (training) ─────────────────
    def sample_actions(self, obs_features, action_mask=None):
        """
        Sample one action per planet slot from the policy distribution.

        Returns
        -------
        actions     : Tensor (batch, max_my_planets)   — sampled action codes
        log_probs   : Tensor (batch, max_my_planets)   — log prob of each action
        entropy     : Tensor (batch,)                  — mean entropy over planets
        value       : Tensor (batch, 1)
        """
        logits, value = self.forward(obs_features, action_mask)
        dist_obj = torch.distributions.Categorical(logits=logits)
        actions   = dist_obj.sample()           # (batch, max_my_planets)
        log_probs = dist_obj.log_prob(actions)  # (batch, max_my_planets)
        entropy   = dist_obj.entropy().mean(-1) # (batch,)
        return actions, log_probs, entropy, value

    # ── Convenience: greedy deterministic actions (inference / evaluation) ─────
    def greedy_actions(self, obs_features, action_mask=None):
        """Return argmax actions — shape (batch, max_my_planets)."""
        logits, value = self.forward(obs_features, action_mask)
        return logits.argmax(dim=-1), value

    # ── Convenience: evaluate log_probs of given actions (PPO update step) ─────
    def evaluate_actions(self, obs_features, actions, action_mask=None):
        """
        Parameters
        ----------
        actions : Tensor (batch, max_my_planets) — previously sampled actions

        Returns
        -------
        log_probs : Tensor (batch, max_my_planets)
        entropy   : Tensor (batch,)
        value     : Tensor (batch, 1)
        """
        logits, value = self.forward(obs_features, action_mask)
        dist_obj  = torch.distributions.Categorical(logits=logits)
        log_probs = dist_obj.log_prob(actions)
        entropy   = dist_obj.entropy().mean(-1)
        return log_probs, entropy, value


# ─────────────────────────────────────────────────────────────────────────────
# Model I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, path, metadata=None):
    """Save model weights + optional metadata dict."""
    payload = {"state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)
    print(f"[model] Saved → {path}")


def load_model(path, device="cpu", **model_kwargs):
    """Load model from checkpoint. Returns (model, metadata_or_None)."""
    payload  = torch.load(path, map_location=device)
    model    = OrbitWarsNet(**model_kwargs).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    meta = payload.get("metadata", None)
    print(f"[model] Loaded ← {path}  |  metadata={meta}")
    return model, meta


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = OrbitWarsNet(hidden_dim=512, num_layers=3, dropout=0.1).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total_params:,}")

    # Forward pass
    batch_size  = 4
    fake_feats  = torch.randn(batch_size, FEATURE_DIM).to(device)
    fake_mask   = torch.ones(batch_size, MAX_MY_PLANETS, NUM_ACTIONS_PER_PLANET,
                             dtype=torch.bool).to(device)

    logits, value = model(fake_feats, fake_mask)
    print(f"Logits shape     : {logits.shape}")   # (4, 10, 200)
    print(f"Value  shape     : {value.shape}")    # (4, 1)

    # Sample actions
    actions, log_probs, entropy, value2 = model.sample_actions(fake_feats, fake_mask)
    print(f"Actions shape    : {actions.shape}")     # (4, 10)
    print(f"Log_probs shape  : {log_probs.shape}")   # (4, 10)
    print(f"Entropy          : {entropy.mean().item():.4f}")

    # Evaluate actions (PPO update path)
    lp2, ent2, val3 = model.evaluate_actions(fake_feats, actions, fake_mask)
    print(f"Evaluate log_probs shape: {lp2.shape}")  # (4, 10)

    # Save / load round-trip
    save_model(model, "/tmp/test_model.pt", metadata={"step": 0})
    model2, meta = load_model("/tmp/test_model.pt", device=device)
    print(f"Load metadata    : {meta}")

    print("\nAll model checks PASSED")
