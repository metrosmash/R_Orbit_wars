"""
feature_utils.py
================
Converts a raw Orbit Wars observation into a fixed-size numpy feature vector.

WHY THIS FILE EXISTS
--------------------
Neural networks need fixed-size inputs, but the game has a variable number of
planets (20-40) and fleets (0 to hundreds). We solve this by:

  1. Sorting planets by a priority score so the most important ones come first.
  2. Always taking exactly MAX_PLANETS slots, padding with zeros if there are fewer.
  3. Doing the same for fleets with MAX_FLEETS slots.

The final vector fed to the neural network is always the same size, regardless
of what's happening in the game.

FEATURE VECTOR LAYOUT (per call to obs_to_features)
----------------------------------------------------
Global features   :  8 values   (my ships, enemy ships, turn info, etc.)
Planet features   :  MAX_PLANETS * PLANET_FEAT_SIZE values
Fleet features    :  MAX_FLEETS  * FLEET_FEAT_SIZE  values
---------------------------------------------------------------------------
Total             :  8 + 40*10 + 30*7 = 618 values  (FEATURE_DIM constant)
"""

import math
import numpy as np

# ── Sizing constants ──────────────────────────────────────────────────────────
MAX_PLANETS    = 40   # game has 20-40 planets; pad/truncate to this
MAX_FLEETS     = 30   # track the 30 most relevant fleets
PLANET_FEAT_SZ = 10   # features per planet slot
FLEET_FEAT_SZ  = 7    # features per fleet slot
GLOBAL_FEAT_SZ = 8    # global / scalar features

FEATURE_DIM = GLOBAL_FEAT_SZ + MAX_PLANETS * PLANET_FEAT_SZ + MAX_FLEETS * FLEET_FEAT_SZ
# = 8 + 400 + 210 = 618

# ── Sun / board constants (match game config) ─────────────────────────────────
SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS   = 10.0
BOARD_SIZE   = 100.0
MAX_SHIPS    = 1000.0   # normalisation ceiling for ship counts
MAX_STEPS    = 500.0    # normalisation ceiling for turn number


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Euclidean distance
# ─────────────────────────────────────────────────────────────────────────────
def dist(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: predict orbiting planet position N turns ahead
#
# Orbiting planets rotate around the sun at angular_velocity radians/turn.
# We use atan2 to get the current angle, add N * omega, then re-project.
# ─────────────────────────────────────────────────────────────────────────────
def predict_planet_pos(px, py, angular_velocity, turns_ahead):
    """Return (x, y) of an orbiting planet `turns_ahead` turns in the future."""
    dx = px - SUN_X
    dy = py - SUN_Y
    r  = math.hypot(dx, dy)
    if r < 1e-6:
        return px, py
    current_angle = math.atan2(dy, dx)
    future_angle  = current_angle + angular_velocity * turns_ahead
    return SUN_X + r * math.cos(future_angle), SUN_Y + r * math.sin(future_angle)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: fleet speed formula (from game spec)
# ─────────────────────────────────────────────────────────────────────────────
def fleet_speed(ships, max_speed=6.0):
    if ships <= 0:
        return 1.0
    return 1.0 + (max_speed - 1.0) * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Helper: estimate travel time from (x1,y1) to (x2,y2) with `ships` ships
# ─────────────────────────────────────────────────────────────────────────────
def travel_time(x1, y1, x2, y2, ships):
    d = dist(x1, y1, x2, y2)
    s = fleet_speed(ships)
    return d / s if s > 0 else float('inf')


# ─────────────────────────────────────────────────────────────────────────────
# Helper: is a planet orbiting? (orbital radius + planet radius < 50)
# ─────────────────────────────────────────────────────────────────────────────
def is_orbiting(px, py, pradius):
    orbital_r = dist(px, py, SUN_X, SUN_Y)
    return (orbital_r + pradius) < 50.0


# ─────────────────────────────────────────────────────────────────────────────
# PLANET FEATURE VECTOR  (10 values per planet)
# ─────────────────────────────────────────────────────────────────────────────
# Index  Meaning                                Range after normalisation
# -----  -------                                -------------------------
#  0     is_mine      (own planet)               0 or 1
#  1     is_enemy     (enemy planet)             0 or 1
#  2     is_neutral   (unclaimed)                0 or 1
#  3     ships / MAX_SHIPS                       [0, 1]
#  4     production / 5.0                        [0, 1]
#  5     dist_to_sun / 50.0                      [0, 1]
#  6     dist_to_my_nearest_planet / BOARD_SIZE  [0, 1]
#  7     is_orbiting                             0 or 1
#  8     x / BOARD_SIZE                          [0, 1]
#  9     y / BOARD_SIZE                          [0, 1]

def planet_features(p, player, my_planets):
    """
    p          : namedtuple or list [id, owner, x, y, radius, ships, production]
    player     : our player id
    my_planets : list of our planet namedtuples/lists (for nearest-distance calc)
    """
    pid, owner, x, y, radius, ships, production = (
        p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    )
    is_mine    = float(owner == player)
    is_enemy   = float(owner != player and owner != -1)
    is_neutral = float(owner == -1)
    ships_norm = min(ships, MAX_SHIPS) / MAX_SHIPS
    prod_norm  = production / 5.0
    sun_dist   = dist(x, y, SUN_X, SUN_Y) / 50.0
    orbiting   = float(is_orbiting(x, y, radius))
    x_norm     = x / BOARD_SIZE
    y_norm     = y / BOARD_SIZE

    if my_planets:
        nearest_my = min(dist(x, y, mp[2], mp[3]) for mp in my_planets)
    else:
        nearest_my = BOARD_SIZE
    nearest_norm = nearest_my / BOARD_SIZE

    return [is_mine, is_enemy, is_neutral,
            ships_norm, prod_norm, sun_dist,
            nearest_norm, orbiting, x_norm, y_norm]


# ─────────────────────────────────────────────────────────────────────────────
# FLEET FEATURE VECTOR  (7 values per fleet)
# ─────────────────────────────────────────────────────────────────────────────
# Index  Meaning
# -----  -------
#  0     is_mine   (our fleet)
#  1     is_enemy  (enemy fleet)
#  2     ships / MAX_SHIPS
#  3     x / BOARD_SIZE
#  4     y / BOARD_SIZE
#  5     angle / (2*pi)   (direction normalised to [0,1])
#  6     dist to sun / 50.0

def fleet_features(f, player):
    """
    f : namedtuple or list [id, owner, x, y, angle, from_planet_id, ships]
    """
    fid, owner, x, y, angle, from_pid, ships = (
        f[0], f[1], f[2], f[3], f[4], f[5], f[6]
    )
    is_mine  = float(owner == player)
    is_enemy = float(owner != player)
    ships_n  = min(ships, MAX_SHIPS) / MAX_SHIPS
    x_n      = x / BOARD_SIZE
    y_n      = y / BOARD_SIZE
    angle_n  = (angle % (2 * math.pi)) / (2 * math.pi)
    sun_d    = dist(x, y, SUN_X, SUN_Y) / 50.0

    return [is_mine, is_enemy, ships_n, x_n, y_n, angle_n, sun_d]


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL FEATURE VECTOR  (8 values)
# ─────────────────────────────────────────────────────────────────────────────
# 0  my_total_ships / MAX_SHIPS
# 1  enemy_total_ships / MAX_SHIPS
# 2  my_planet_count / MAX_PLANETS
# 3  enemy_planet_count / MAX_PLANETS
# 4  neutral_planet_count / MAX_PLANETS
# 5  my_total_production / (MAX_PLANETS * 5)
# 6  step / MAX_STEPS         (requires step to be passed in, default 0)
# 7  ship_ratio               (my / (my + enemy + 1))

def global_features(planets, fleets, player, step=0):
    my_ships     = 0.0
    enemy_ships  = 0.0
    my_planets   = 0
    enemy_planets= 0
    neutral_pl   = 0
    my_prod      = 0.0

    for p in planets:
        owner, ships, prod = p[1], p[5], p[6]
        if owner == player:
            my_ships   += ships
            my_planets += 1
            my_prod    += prod
        elif owner == -1:
            neutral_pl += 1
        else:
            enemy_ships   += ships
            enemy_planets += 1

    for f in fleets:
        if f[1] == player:
            my_ships += f[6]
        else:
            enemy_ships += f[6]

    total = my_ships + enemy_ships + 1.0
    return [
        min(my_ships, MAX_SHIPS)     / MAX_SHIPS,
        min(enemy_ships, MAX_SHIPS)  / MAX_SHIPS,
        my_planets   / MAX_PLANETS,
        enemy_planets/ MAX_PLANETS,
        neutral_pl   / MAX_PLANETS,
        my_prod      / (MAX_PLANETS * 5.0),
        step         / MAX_STEPS,
        my_ships     / total,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# PLANET PRIORITY SCORE
#
# We have MAX_PLANETS slots but might have fewer actual planets — that's fine,
# we pad. If we have more, we keep the most strategically relevant ones.
#
# Priority (higher = more important to include):
#   - My own planets: always include (high base score)
#   - Neutral/enemy: closer to my planets = higher priority
#   - Higher production = higher priority
# ─────────────────────────────────────────────────────────────────────────────
def planet_priority(p, player, my_planets):
    owner, x, y, ships, prod = p[1], p[2], p[3], p[5], p[6]
    if owner == player:
        return 1e9   # always include own planets first

    if my_planets:
        d = min(dist(x, y, mp[2], mp[3]) for mp in my_planets)
    else:
        d = BOARD_SIZE

    # Closer + higher production = higher priority
    return prod * 10.0 - d


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT: obs_to_features
# ─────────────────────────────────────────────────────────────────────────────
def obs_to_features(obs, step=0):
    """
    Convert a raw Orbit Wars observation dict into a fixed-size numpy array
    of shape (FEATURE_DIM,) = (618,).

    Parameters
    ----------
    obs  : dict  — the observation passed to agent(obs)
    step : int   — current game step (0-500), used for time-awareness

    Returns
    -------
    np.ndarray of shape (FEATURE_DIM,) dtype float32
    """
    # ── Parse raw observation ─────────────────────────────────────────────────
    if isinstance(obs, dict):
        player  = obs.get("player", 0)
        planets = obs.get("planets", [])
        fleets  = obs.get("fleets",  [])
    else:
        player  = obs.player
        planets = list(obs.planets)
        fleets  = list(obs.fleets)

    my_planets = [p for p in planets if p[1] == player]

    # ── Global features ───────────────────────────────────────────────────────
    g_feats = global_features(planets, fleets, player, step)

    # ── Planet features (sorted by priority, padded to MAX_PLANETS) ───────────
    sorted_planets = sorted(
        planets,
        key=lambda p: planet_priority(p, player, my_planets),
        reverse=True
    )[:MAX_PLANETS]

    p_feats = []
    for p in sorted_planets:
        p_feats.extend(planet_features(p, player, my_planets))
    # Pad with zeros if fewer than MAX_PLANETS
    pad_count = MAX_PLANETS - len(sorted_planets)
    p_feats.extend([0.0] * (pad_count * PLANET_FEAT_SZ))

    # ── Fleet features (our fleets first, then enemy, truncate to MAX_FLEETS) ─
    my_fleets  = [f for f in fleets if f[1] == player]
    opp_fleets = [f for f in fleets if f[1] != player]
    # Sort each group: larger fleets first (more strategically relevant)
    my_fleets  = sorted(my_fleets,  key=lambda f: f[6], reverse=True)
    opp_fleets = sorted(opp_fleets, key=lambda f: f[6], reverse=True)
    combined   = (my_fleets + opp_fleets)[:MAX_FLEETS]

    f_feats = []
    for f in combined:
        f_feats.extend(fleet_features(f, player))
    pad_count = MAX_FLEETS - len(combined)
    f_feats.extend([0.0] * (pad_count * FLEET_FEAT_SZ))

    # ── Concatenate everything ────────────────────────────────────────────────
    full = g_feats + p_feats + f_feats
    assert len(full) == FEATURE_DIM, f"Feature dim mismatch: {len(full)} != {FEATURE_DIM}"

    return np.array(full, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check (run this file directly to verify)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build a fake minimal observation to verify shapes
    fake_obs = {
        "player": 0,
        "planets": [
            [0,  0, 20.0, 20.0, 2.0, 30, 3],   # mine
            [1, -1, 80.0, 80.0, 1.0, 10, 1],   # neutral
            [2,  1, 75.0, 25.0, 1.0, 15, 2],   # enemy
        ],
        "fleets": [
            [0, 0, 22.0, 22.0, 0.785, 0, 50],  # my fleet
            [1, 1, 78.0, 28.0, 3.14,  2, 20],  # enemy fleet
        ],
        "angular_velocity": 0.03,
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
    }

    features = obs_to_features(fake_obs, step=42)
    print(f"Feature vector shape : {features.shape}")
    print(f"Expected             : ({FEATURE_DIM},)")
    print(f"Non-zero values      : {(features != 0).sum()}")
    print(f"Min / Max            : {features.min():.3f} / {features.max():.3f}")
    print("PASSED" if features.shape == (FEATURE_DIM,) else "FAILED")
