"""
parse_replays.py
================
Converts Kaggle episode replay JSON files into a supervised training dataset
for imitation learning (IL).

WHAT THIS DOES (Plain English)
-------------------------------
Each replay file is one complete game — 500 turns, all moves made by all bots.
We go through each replay and ask:
  "For each turn the WINNING player took, what was the game state and what
   did they do?"

That (state, action) pair becomes one training example. We save thousands
of these to disk as numpy arrays. The imitation learning trainer (train_il.py)
then loads these and teaches our neural network to mimic good play.

WHY ONLY THE WINNER?
--------------------
We only learn from the winner because:
  1. Winners made better decisions (on average).
  2. Learning from losers could teach bad habits.

We also apply a QUALITY FILTER — we skip games where the winner's final score
is less than 1.5× the loser's score, because close games are noisier.

REPLAY FORMAT
-------------
Kaggle replays look like this (from kaggle-environments):

{
  "steps": [
    [                           ← one step (list of per-player entries)
      {
        "observation": { "player": 0, "planets": [...], "fleets": [...], ... },
        "action": [[from_id, angle, num_ships], ...],    ← can be null
        "reward": null,         ← only set at game end
        "status": "ACTIVE"
      },
      { ... }                   ← player 1's entry
    ],
    ...                         ← 500 steps total
  ],
  "rewards": [score0, score1]   ← final scores
}

HOW ACTIONS ARE ENCODED FOR TRAINING
-------------------------------------
The neural net predicts one action code per planet slot (up to MAX_MY_PLANETS).
We reverse-engineer the replay's move list into action codes using:
  - For each move [from_id, angle, ships], find which planet it came from and
    which target planet the angle is closest to.
  - Determine the ratio (ships_sent / available_ships).
  - Encode as (target_idx, ratio_idx) → action_code.

USAGE
-----
  python parse_replays.py --replay_dir ./replays --out_dir ./il_data

Output
------
  il_data/
    states.npy     shape (N, FEATURE_DIM)   float32
    actions.npy    shape (N, MAX_MY_PLANETS) int32
    metadata.json  { "n_samples": N, "n_replays": M, "quality_filtered": K }
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from tqdm import tqdm

# Allow imports from agent/ whether this script is run from repo root,
# from training/, or directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AGENT_DIR = os.path.join(_REPO_ROOT, "agent")
for _p in [_AGENT_DIR, _REPO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from feature_utils  import (
    obs_to_features, FEATURE_DIM, MAX_PLANETS,
    dist, planet_priority
)
from action_utils   import (
    encode_action, NUM_TARGETS, NUM_RATIOS, SHIP_RATIOS,
    MAX_MY_PLANETS, NUM_ACTIONS_PER_PLANET
)

# ── Config ────────────────────────────────────────────────────────────────────
QUALITY_RATIO    = 1.3    # winner must have >= 1.3× the loser's final score
MAX_ANGLE_ERR    = 0.4    # radians — max angle diff to match a move to a target
MIN_SHIPS_SENT   = 2      # ignore moves that send fewer ships than this


# ─────────────────────────────────────────────────────────────────────────────
# find_best_target
#
# Given a launch angle from a source planet, find which target planet in
# sorted_planets the fleet was most likely aiming at.
# ─────────────────────────────────────────────────────────────────────────────
def find_best_target(src_x, src_y, launch_angle, sorted_planets):
    """
    Return (target_idx, angle_error) for the most likely target.
    Returns (None, inf) if no target is within MAX_ANGLE_ERR.
    """
    best_idx = None
    best_err = float('inf')

    for idx, p in enumerate(sorted_planets):
        tx, ty = p[2], p[3]
        expected_angle = math.atan2(ty - src_y, tx - src_x)
        err = abs((launch_angle - expected_angle + math.pi) % (2 * math.pi) - math.pi)
        if err < best_err:
            best_err = err
            best_idx = idx

    if best_err > MAX_ANGLE_ERR:
        return None, best_err
    return best_idx, best_err


# ─────────────────────────────────────────────────────────────────────────────
# infer_ratio_idx
#
# Given ships_sent and ships_available, return the closest ratio index.
# ─────────────────────────────────────────────────────────────────────────────
def infer_ratio_idx(ships_sent, ships_available):
    if ships_available <= 0:
        return 0
    ratio = ships_sent / ships_available
    # Find closest SHIP_RATIOS entry (skip 0.0 = hold)
    best_idx = 1
    best_err = abs(SHIP_RATIOS[1] - ratio)
    for i in range(2, NUM_RATIOS):
        err = abs(SHIP_RATIOS[i] - ratio)
        if err < best_err:
            best_err = err
            best_idx = i
    return best_idx


# ─────────────────────────────────────────────────────────────────────────────
# parse_step
#
# Given the observation at one step and the list of moves the winning player
# made, return:
#   features  : np.ndarray (FEATURE_DIM,)
#   actions   : np.ndarray (MAX_MY_PLANETS,)  action codes (0 = hold)
#
# Returns (None, None) if the step should be skipped.
# ─────────────────────────────────────────────────────────────────────────────
def parse_step(obs_dict, moves, step_idx):
    """
    obs_dict : the raw observation dict from the replay
    moves    : list of [from_id, angle, ships] or None/[]
    step_idx : current step number (for features)
    """
    try:
        player  = obs_dict.get("player", 0)
        planets = obs_dict.get("planets", [])
        av      = obs_dict.get("angular_velocity", 0.03)

        if not planets:
            return None, None

        my_planets = [p for p in planets if p[1] == player]
        if not my_planets:
            return None, None

        # Sorted planet list (same order as feature vector)
        sorted_planets = sorted(
            planets,
            key=lambda p: planet_priority(p, player, my_planets),
            reverse=True
        )[:MAX_PLANETS]

        # Build planet_id → index in sorted_planets for fast lookup
        planet_id_to_sorted_idx = {p[0]: i for i, p in enumerate(sorted_planets)}

        # Build planet_id → planet row
        planet_map = {p[0]: p for p in planets}

        # ── Feature vector ────────────────────────────────────────────────────
        features = obs_to_features(obs_dict, step=step_idx)

        # ── Action vector — default all zeros (hold) ──────────────────────────
        # Position in action vector = position of src planet in MY_PLANETS list
        # (sorted by priority, same as feature extraction)
        my_planets_sorted = sorted(
            my_planets,
            key=lambda p: planet_priority(p, player, my_planets),
            reverse=True
        )[:MAX_MY_PLANETS]

        my_planet_id_to_slot = {p[0]: i for i, p in enumerate(my_planets_sorted)}

        action_codes = np.zeros(MAX_MY_PLANETS, dtype=np.int32)

        if moves:
            for move in moves:
                if move is None or len(move) < 3:
                    continue
                from_id, angle, ships_sent = move[0], move[1], move[2]

                if ships_sent < MIN_SHIPS_SENT:
                    continue

                # Which slot does this source planet occupy in our action vector?
                slot = my_planet_id_to_slot.get(from_id, None)
                if slot is None:
                    continue   # planet not in our tracked slots

                src = planet_map.get(from_id, None)
                if src is None:
                    continue

                # Find which target planet this angle points at
                target_idx, angle_err = find_best_target(
                    src[2], src[3], angle, sorted_planets
                )
                if target_idx is None:
                    continue

                # Infer ratio
                ships_available = src[5]
                ratio_idx = infer_ratio_idx(ships_sent, ships_available)
                if ratio_idx == 0:
                    ratio_idx = 1   # at least 25% if something was sent

                action_codes[slot] = encode_action(target_idx, ratio_idx)

        return features, action_codes

    except Exception as e:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# parse_replay_file
#
# Parse one replay JSON file, return lists of (features, actions).
# ─────────────────────────────────────────────────────────────────────────────
def parse_replay_file(filepath):
    """
    Returns
    -------
    list of (features_array, actions_array) tuples
    Empty list if file is invalid, below quality threshold, or errored.
    """
    try:
        with open(filepath, "r") as f:
            replay = json.load(f)
    except Exception:
        return []

    steps   = replay.get("steps",   [])
    rewards = replay.get("rewards", [])

    if len(steps) < 10 or len(rewards) < 2:
        return []

    # ── Quality filter ────────────────────────────────────────────────────────
    # Only learn from dominant wins to reduce noise
    r0, r1 = rewards[0], rewards[1]
    if r0 is None or r1 is None:
        return []

    if r0 > r1:
        winner_player = 0
        if r1 > 0 and r0 / r1 < QUALITY_RATIO:
            return []   # too close
    elif r1 > r0:
        winner_player = 1
        if r0 > 0 and r1 / r0 < QUALITY_RATIO:
            return []
    else:
        return []   # draw — skip

    # ── Parse each step ───────────────────────────────────────────────────────
    samples = []

    for step_idx, step in enumerate(steps):
        if len(step) <= winner_player:
            continue

        entry = step[winner_player]
        obs   = entry.get("observation", None)
        action= entry.get("action", None)

        if obs is None:
            continue

        # Make sure the obs has the right player id
        # (kaggle sometimes gives obs from the environment's perspective)
        obs_copy = dict(obs)
        obs_copy["player"] = winner_player

        features, actions = parse_step(obs_copy, action, step_idx)
        if features is None:
            continue

        samples.append((features, actions))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# parse_all_replays
#
# Walk a directory of .json replay files and build the full IL dataset.
# ─────────────────────────────────────────────────────────────────────────────
def parse_all_replays(replay_dir, out_dir, max_files=None):
    """
    Parameters
    ----------
    replay_dir : directory containing .json replay files
    out_dir    : where to save states.npy, actions.npy, metadata.json
    max_files  : optional cap on number of replays to process (useful for testing)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect all json files
    all_files = [
        os.path.join(replay_dir, f)
        for f in os.listdir(replay_dir)
        if f.endswith(".json")
    ]
    if max_files:
        all_files = all_files[:max_files]

    print(f"Found {len(all_files)} replay files in {replay_dir}")

    all_features = []
    all_actions  = []
    n_replays_used   = 0
    n_replays_skipped= 0

    for fpath in tqdm(all_files, desc="Parsing replays"):
        samples = parse_replay_file(fpath)
        if not samples:
            n_replays_skipped += 1
            continue

        for feat, acts in samples:
            all_features.append(feat)
            all_actions.append(acts)
        n_replays_used += 1

    if not all_features:
        print("No valid samples found. Check replay directory and format.")
        return

    # ── Save as numpy arrays ──────────────────────────────────────────────────
    states_arr  = np.stack(all_features, axis=0)   # (N, FEATURE_DIM)
    actions_arr = np.stack(all_actions,  axis=0)   # (N, MAX_MY_PLANETS)

    states_path  = os.path.join(out_dir, "states.npy")
    actions_path = os.path.join(out_dir, "actions.npy")
    np.save(states_path,  states_arr)
    np.save(actions_path, actions_arr)

    metadata = {
        "n_samples"        : len(all_features),
        "n_replays_used"   : n_replays_used,
        "n_replays_skipped": n_replays_skipped,
        "feature_dim"      : FEATURE_DIM,
        "max_my_planets"   : MAX_MY_PLANETS,
        "quality_ratio"    : QUALITY_RATIO,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved to {out_dir}")
    print(f"  Total samples   : {len(all_features):,}")
    print(f"  Replays used    : {n_replays_used:,}")
    print(f"  Replays skipped : {n_replays_skipped:,}  (low quality / bad format)")
    print(f"  states.npy      : {states_arr.shape}  {states_arr.nbytes/1e6:.1f} MB")
    print(f"  actions.npy     : {actions_arr.shape}  {actions_arr.nbytes/1e6:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Orbit Wars replays → IL dataset")
    parser.add_argument("--replay_dir", default=os.path.join(_REPO_ROOT, "replays"),
                        help="Directory containing .json replay files")
    parser.add_argument("--out_dir",    default=os.path.join(_REPO_ROOT, "il_data"),
                        help="Output directory for states.npy and actions.npy")
    parser.add_argument("--max_files",  type=int, default=None,
                        help="Cap on replay files (for testing)")
    args = parser.parse_args()

    parse_all_replays(args.replay_dir, args.out_dir, args.max_files)
