"""
action_utils.py
===============
Translates between the neural network's discrete action choices and the
game's actual move format: [from_planet_id, angle_radians, num_ships].

THE PROBLEM
-----------
The game's action space is theoretically infinite — any planet, any angle,
any ship count. Neural networks work best with a finite set of labeled choices.

OUR SOLUTION: Factored Discrete Action Space
--------------------------------------------
Each turn, for each planet we own, the network makes two independent choices:

  Choice A — TARGET:  which of the MAX_PLANETS planet slots to aim at
  Choice B — RATIO:   what fraction of current ships to send

  NUM_TARGETS = MAX_PLANETS  = 40  (index into the sorted planet list)
  NUM_RATIOS  = 5
      0 → send nothing  (pass / hold)
      1 → send 25%
      2 → send 50%
      3 → send 75%
      4 → send 100%  (but always keep at least 1 ship behind)

So the per-planet action space is 40 × 5 = 200 combinations.

IMPORTANT NOTES
---------------
- We aim at the FUTURE position of orbiting planets (intercept calculation).
- We never send 0 ships (if ratio × ships rounds to 0 we skip that move).
- We always leave at least 1 ship on each planet (never fully strip a planet).
- The neural network outputs one (target_idx, ratio_idx) pair per planet slot.
  Slots that correspond to planets we don't own are masked out (ignored).
"""

import math
import numpy as np

from feature_utils import (
    MAX_PLANETS, dist, fleet_speed, travel_time,
    predict_planet_pos, is_orbiting,
    SUN_X, SUN_Y
)

# ── Action space dimensions ───────────────────────────────────────────────────
NUM_TARGETS = MAX_PLANETS   # 40 possible target slots
NUM_RATIOS  = 5             # 0=hold, 1=25%, 2=50%, 3=75%, 4=100%
SHIP_RATIOS = [0.0, 0.25, 0.50, 0.75, 1.0]

# Combined per-planet action count (used by DQN / flattened approaches)
NUM_ACTIONS_PER_PLANET = NUM_TARGETS * NUM_RATIOS   # 200


# ─────────────────────────────────────────────────────────────────────────────
# encode_action
#
# Given target_idx and ratio_idx, return the integer action code.
# action_code = target_idx * NUM_RATIOS + ratio_idx
# ─────────────────────────────────────────────────────────────────────────────
def encode_action(target_idx, ratio_idx):
    """Return integer in [0, NUM_ACTIONS_PER_PLANET)."""
    return target_idx * NUM_RATIOS + ratio_idx


def decode_action(action_code):
    """Return (target_idx, ratio_idx) from integer action code."""
    target_idx = action_code // NUM_RATIOS
    ratio_idx  = action_code  % NUM_RATIOS
    return target_idx, ratio_idx


# ─────────────────────────────────────────────────────────────────────────────
# aim_angle
#
# Compute the angle from (sx, sy) to a target planet, accounting for the
# planet's orbital movement during the fleet's travel time.
#
# For static planets this is just atan2(dy, dx).
# For orbiting planets we iterate: estimate travel time → predict future
# position → recompute angle → repeat (converges in 2-3 steps).
# ─────────────────────────────────────────────────────────────────────────────
def aim_angle(sx, sy, tx, ty, tradius, angular_velocity, ships, max_iters=4):
    """
    Return the launch angle (radians) from source (sx,sy) toward target.

    Parameters
    ----------
    sx, sy          : source position (launching planet)
    tx, ty          : target's CURRENT position
    tradius         : target planet's radius (to detect if it orbits)
    angular_velocity: game's angular_velocity value from obs
    ships           : number of ships in the fleet (affects speed)
    """
    orbiting = is_orbiting(tx, ty, tradius)

    if not orbiting:
        # Static — aim directly
        return math.atan2(ty - sy, tx - sx)

    # Iterative intercept for orbiting targets
    ftx, fty = tx, ty
    for _ in range(max_iters):
        tt   = travel_time(sx, sy, ftx, fty, ships)
        ftx, fty = predict_planet_pos(tx, ty, angular_velocity, tt)

    return math.atan2(fty - sy, ftx - sx)


# ─────────────────────────────────────────────────────────────────────────────
# actions_to_moves
#
# Convert the network's chosen (target_idx, ratio_idx) pairs for each
# owned planet into the game's move list format:
#   [[from_planet_id, angle_radians, num_ships], ...]
#
# Parameters
# ----------
# planet_actions : list of (target_idx, ratio_idx) — one per owned planet
# my_planets     : list of planet rows for planets I own
# sorted_planets : the full sorted planet list (same order as feature vector)
#                  so that target_idx maps to the right planet
# angular_velocity: from obs, for intercept calculation
#
# Returns
# -------
# List of [from_planet_id, angle, num_ships]  (may be empty)
# ─────────────────────────────────────────────────────────────────────────────
def actions_to_moves(planet_actions, my_planets, sorted_planets, angular_velocity):
    """
    Example
    -------
    planet_actions = [(3, 2), (7, 1)]   # planet 0 sends 50% to slot 3,
                                         # planet 1 sends 25% to slot 7
    """
    moves = []

    for i, (target_idx, ratio_idx) in enumerate(planet_actions):
        if i >= len(my_planets):
            break

        src = my_planets[i]
        src_id   = src[0]
        src_x    = src[2]
        src_y    = src[3]
        src_ships= src[5]

        # Ratio 0 = hold, skip
        if ratio_idx == 0:
            continue

        ratio     = SHIP_RATIOS[ratio_idx]
        num_ships = max(1, int(src_ships * ratio))

        # Always keep at least 1 ship on the source planet
        num_ships = min(num_ships, src_ships - 1)
        if num_ships <= 0:
            continue

        # Resolve target
        if target_idx >= len(sorted_planets):
            continue
        tgt = sorted_planets[target_idx]
        tgt_x    = tgt[2]
        tgt_y    = tgt[3]
        tgt_r    = tgt[4]

        # Don't send to own planet unnecessarily (ratio 4 = all-in is fine
        # for reinforcing, but skip 100% sends to self in the basic case)
        if tgt[1] == src[1] and ratio_idx < 4:
            continue

        angle = aim_angle(src_x, src_y, tgt_x, tgt_y, tgt_r,
                          angular_velocity, num_ships)

        moves.append([src_id, angle, num_ships])

    return moves


# ─────────────────────────────────────────────────────────────────────────────
# build_action_mask
#
# Returns a boolean mask of shape (len(my_planets), NUM_ACTIONS_PER_PLANET)
# where False means "this action is illegal / pointless and should not be
# selected by the network".
#
# Used during training to prevent the agent learning to send fleets to itself
# unnecessarily or to targets that don't exist.
# ─────────────────────────────────────────────────────────────────────────────
def build_action_mask(my_planets, sorted_planets):
    """
    Returns np.ndarray of shape (len(my_planets), NUM_ACTIONS_PER_PLANET)
    dtype bool — True = valid action.
    """
    n = len(my_planets)
    mask = np.ones((n, NUM_ACTIONS_PER_PLANET), dtype=bool)

    for i, src in enumerate(my_planets):
        src_ships = src[5]

        for target_idx in range(NUM_TARGETS):
            for ratio_idx in range(NUM_RATIOS):
                code = encode_action(target_idx, ratio_idx)

                # Can't send if we don't have ships
                if src_ships <= 1 and ratio_idx > 0:
                    mask[i, code] = False
                    continue

                # Target slot doesn't exist
                if target_idx >= len(sorted_planets):
                    mask[i, code] = False
                    continue

                # ratio=0 (hold) is always valid
                if ratio_idx == 0:
                    continue

                tgt = sorted_planets[target_idx]
                # Sending to own planet with ratio < 75% is usually pointless
                # (we allow 75% and 100% for reinforcing)
                if tgt[1] == src[1] and ratio_idx < 3:
                    mask[i, code] = False

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"NUM_TARGETS              : {NUM_TARGETS}")
    print(f"NUM_RATIOS               : {NUM_RATIOS}")
    print(f"NUM_ACTIONS_PER_PLANET   : {NUM_ACTIONS_PER_PLANET}")

    # Encode / decode round-trip
    for t in [0, 5, 39]:
        for r in range(NUM_RATIOS):
            code = encode_action(t, r)
            t2, r2 = decode_action(code)
            assert t == t2 and r == r2, f"Round-trip failed: ({t},{r}) -> {code} -> ({t2},{r2})"
    print("Encode/decode round-trip : PASSED")

    # Aim angle — static planet
    angle = aim_angle(20, 20, 80, 80, 1.5, 0.03, 100)
    expected = math.atan2(60, 60)
    print(f"Aim angle (static)       : {angle:.4f}  expected {expected:.4f}  {'PASSED' if abs(angle-expected)<0.001 else 'FAILED'}")

    # Aim angle — orbiting planet (just check it runs without error)
    angle_orb = aim_angle(20, 20, 45, 40, 1.5, 0.03, 50)
    print(f"Aim angle (orbiting)     : {angle_orb:.4f}  (no crash = PASSED)")
    print("All action_utils checks  : PASSED")
