"""
main.py
=======
Kaggle Orbit Wars submission file.

SUBMISSION RULES
----------------
  - The last `def` in this file must accept an observation and return an action.
  - Multiple files can be submitted as a .tar.gz with main.py at the root.

HOW THIS WORKS AT SUBMISSION TIME
----------------------------------
When Kaggle runs our agent, it:
  1. Imports this file.
  2. Calls agent(obs) every turn.
  3. Expects a list of moves: [[from_planet_id, angle, num_ships], ...]

We load the neural network weights ONCE at module load time (outside agent()),
so we don't reload them every single turn (that would be very slow).

FALLBACK LOGIC
--------------
If the model weights file isn't found (e.g. first submission test without
training), we fall back to the hardcoded rule-based bot automatically.
This means you can submit main.py alone at any stage and it will work.

BUNDLE STRUCTURE (for full RL submission)
------------------------------------------
  tar -czf submission.tar.gz \
      main.py \
      model.py \
      feature_utils.py \
      action_utils.py \
      hardcoded_bot.py \
      checkpoints/ppo_best.pt

  kaggle competitions submit orbit-wars -f submission.tar.gz -m "PPO v1"
"""

import os
import math
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
# When Kaggle extracts the tar.gz, files land in the working directory.
# We check a few possible locations for the weights file.
_WEIGHT_CANDIDATES = [
    "./checkpoints/ppo_best.pt",
    "./checkpoints/ppo_final.pt",
    "./checkpoints/ppo_latest.pt",
    "./checkpoints/il_best.pt",
    "./ppo_best.pt",
]

# ── Try to load the neural network ───────────────────────────────────────────
_model       = None
_device      = "cpu"   # Kaggle inference runs on CPU; training uses GPU
_use_rl_bot  = False
_step_counter = 0      # track game step for time-aware features

def _try_load_model():
    """Called once at import time. Returns True if model loaded successfully."""
    global _model, _device, _use_rl_bot

    # Find weights
    weights_path = None
    for candidate in _WEIGHT_CANDIDATES:
        if os.path.exists(candidate):
            weights_path = candidate
            break

    if weights_path is None:
        print("[main.py] No model weights found — using hardcoded bot fallback.")
        return False

    try:
        import torch
        from model import OrbitWarsNet
        from feature_utils import FEATURE_DIM
        from action_utils import MAX_MY_PLANETS

        # Load checkpoint
        payload = torch.load(weights_path, map_location="cpu")
        meta    = payload.get("metadata", {})
        hd      = meta.get("hidden_dim", 512)
        nl      = meta.get("num_layers",  3)

        _model = OrbitWarsNet(hidden_dim=hd, num_layers=nl).to("cpu")
        _model.load_state_dict(payload["state_dict"])
        _model.eval()
        _use_rl_bot = True
        print(f"[main.py] RL model loaded from {weights_path}  meta={meta}")
        return True

    except Exception as e:
        print(f"[main.py] Model load failed ({e}) — using hardcoded bot fallback.")
        return False


_use_rl_bot = _try_load_model()


# ── RL bot inference ──────────────────────────────────────────────────────────
def _rl_agent_moves(obs):
    """Run the neural network and convert outputs to game moves."""
    import torch
    from feature_utils import obs_to_features, MAX_PLANETS, planet_priority
    from action_utils  import decode_action, actions_to_moves, MAX_MY_PLANETS

    global _step_counter
    _step_counter += 1

    # Parse observation
    if isinstance(obs, dict):
        player  = obs.get("player", 0)
        planets = obs.get("planets", [])
        av      = obs.get("angular_velocity", 0.03)
    else:
        player  = obs.player
        planets = list(obs.planets)
        av      = obs.angular_velocity

    my_planets = [p for p in planets if p[1] == player]
    if not my_planets:
        return []

    # Sorted planet list (must match training order)
    sorted_planets = sorted(
        planets,
        key=lambda p: planet_priority(p, player, my_planets),
        reverse=True
    )[:MAX_PLANETS]

    # Feature vector
    features = obs_to_features(obs, step=_step_counter)
    obs_t    = torch.FloatTensor(features).unsqueeze(0)   # (1, FEATURE_DIM)

    # Network inference
    with torch.no_grad():
        actions_t, _ = _model.greedy_actions(obs_t)
    actions_np = actions_t[0].cpu().numpy()   # (MAX_MY_PLANETS,)

    # Decode to game moves
    planet_actions = []
    for i in range(min(len(my_planets), MAX_MY_PLANETS)):
        code = int(actions_np[i])
        planet_actions.append(decode_action(code))

    moves = actions_to_moves(planet_actions, my_planets, sorted_planets, av)
    return moves


# ── Reset step counter on new game ───────────────────────────────────────────
# Kaggle calls agent(obs) each turn. We detect a new game when step appears
# to reset (obs has no step field, so we track it ourselves and reset when
# we detect a new game via fleet/planet counts dropping drastically).
_prev_planet_count = -1

def _maybe_reset_step(obs):
    global _step_counter, _prev_planet_count
    if isinstance(obs, dict):
        n_planets = len(obs.get("planets", []))
    else:
        n_planets = len(list(obs.planets))

    # If planet count jumps up significantly, it's a new game
    if n_planets > _prev_planet_count + 5 and _prev_planet_count > 0:
        _step_counter = 0
    _prev_planet_count = n_planets


# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED FALLBACK (inlined to keep main.py self-contained if needed)
# ─────────────────────────────────────────────────────────────────────────────
def _hardcoded_fallback(obs):
    """Import and run hardcoded_bot. Falls back to minimal logic if import fails."""
    try:
        from hardcoded_bot import agent as hc_agent
        return hc_agent(obs)
    except ImportError:
        pass

    # Ultra-minimal fallback if hardcoded_bot.py is also missing
    try:
        if isinstance(obs, dict):
            player  = obs.get("player", 0)
            planets = obs.get("planets", [])
        else:
            player  = obs.player
            planets = list(obs.planets)

        my_planets  = [p for p in planets if p[1] == player]
        targets     = [p for p in planets if p[1] != player]
        if not my_planets or not targets:
            return []

        moves = []
        for mp in my_planets:
            if mp[5] < 5:
                continue
            best = min(targets, key=lambda t: math.hypot(mp[2]-t[2], mp[3]-t[3]))
            angle= math.atan2(best[3]-mp[3], best[2]-mp[2])
            ships= max(1, mp[5] // 2)
            moves.append([mp[0], angle, ships])
        return moves
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# agent() — THIS MUST BE THE LAST def IN THE FILE
# ─────────────────────────────────────────────────────────────────────────────
def agent(obs):
    """
    Main agent function called by Kaggle every turn.

    Parameters
    ----------
    obs : observation dict (or namedtuple) from the Orbit Wars environment.

    Returns
    -------
    list of [from_planet_id, angle_radians, num_ships]
    """
    try:
        _maybe_reset_step(obs)

        if _use_rl_bot and _model is not None:
            moves = _rl_agent_moves(obs)
            # If RL returns nothing, fall back to hardcoded
            if moves is not None and len(moves) > 0:
                return moves

        return _hardcoded_fallback(obs)

    except Exception as e:
        # Never crash — always return something
        print(f"[agent] Error: {e}")
        try:
            return _hardcoded_fallback(obs)
        except Exception:
            return []
