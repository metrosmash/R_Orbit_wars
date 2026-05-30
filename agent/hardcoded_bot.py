"""
hardcoded_bot.py
================
A strong rule-based bot that beats the example "nearest planet sniper".
Used as:
  1. The baseline to beat with RL
  2. The training opponent during early PPO self-play

STRATEGY (in priority order each turn)
---------------------------------------
1. DEFEND   — if an enemy fleet is heading toward one of my planets and will
               arrive before I can send reinforcements, send ships to defend.
2. EXPAND   — send ships to capture the best neutral planet (score =
               production / (distance + ships_needed)). Prefer high-production,
               close, low-garrison neutrals.
3. ATTACK   — if I'm stronger overall, attack the weakest reachable enemy planet.
4. REINFORCE— if I have a planet that's very rich but lightly defended, and I
               have another planet with surplus ships, reinforce it.
5. IDLE     — do nothing if none of the above apply.

INTERCEPT CALCULATION
---------------------
For orbiting planets we aim at where they WILL BE when our fleet arrives,
not where they are now. This avoids the sniper bot's biggest weakness —
fleets sailing into empty space.
"""

import math
from feature_utils import (
    dist, fleet_speed, travel_time,
    predict_planet_pos, is_orbiting,
    SUN_X, SUN_Y
)
from action_utils import aim_angle

# ── Tuning knobs ──────────────────────────────────────────────────────────────
DEFEND_BUFFER        = 1.15   # send 15% extra ships when defending
ATTACK_THRESHOLD     = 1.20   # only attack if we have 20% more ships than needed
EXPAND_SHIP_FRACTION = 0.55   # send 55% of a planet's ships when expanding
MIN_SHIPS_TO_SEND    = 3      # never send fewer than this many ships
REINFORCE_THRESHOLD  = 0.4    # reinforce if planet has < 40% of "expected" garrison


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_obs(obs):
    """Return (player, planets_list, fleets_list, angular_velocity)."""
    if isinstance(obs, dict):
        player = obs.get("player", 0)
        planets = obs.get("planets", [])
        fleets  = obs.get("fleets",  [])
        av      = obs.get("angular_velocity", 0.03)
    else:
        player  = obs.player
        planets = list(obs.planets)
        fleets  = list(obs.fleets)
        av      = obs.angular_velocity
    return player, planets, fleets, av


def _planet(p):
    """Unpack planet row."""
    return p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    # id, owner, x, y, radius, ships, production


def _fleet(f):
    """Unpack fleet row."""
    return f[0], f[1], f[2], f[3], f[4], f[5], f[6]
    # id, owner, x, y, angle, from_planet_id, ships


def _future_pos(p, av, turns):
    """Future (x,y) of planet p after `turns` turns."""
    pid, owner, x, y, r, ships, prod = _planet(p)
    if is_orbiting(x, y, r):
        return predict_planet_pos(x, y, av, turns)
    return x, y


def _travel_turns(src, tgt, ships, av):
    """Estimated turns for `ships` to travel from src planet to tgt planet."""
    sx, sy = src[2], src[3]
    tx, ty = tgt[2], tgt[3]
    tr     = tgt[4]
    # Iterative intercept
    ftx, fty = tx, ty
    for _ in range(4):
        tt = travel_time(sx, sy, ftx, fty, ships)
        ftx, fty = _future_pos(tgt, av, tt)
    return travel_time(sx, sy, ftx, fty, ships)


def _make_move(src, tgt, ships, av):
    """Return a [from_id, angle, ships] move tuple."""
    sx, sy = src[2], src[3]
    tx, ty = tgt[2], tgt[3]
    tr     = tgt[4]
    angle = aim_angle(sx, sy, tx, ty, tr, av, ships)
    return [src[0], angle, ships]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DEFEND
# detect enemy fleets heading toward my planets; send reinforcements
# ─────────────────────────────────────────────────────────────────────────────

def _defend_moves(my_planets, all_planets, fleets, av, used_ships):
    """
    For each enemy fleet that appears to be heading toward one of my planets,
    if the planet will fall, send ships from the nearest other planet I own.

    used_ships : dict {planet_id: ships_already_committed_this_turn}
    """
    moves = []
    planet_map = {p[0]: p for p in all_planets}

    for f in fleets:
        fid, fowner, fx, fy, fangle, from_pid, fships = _fleet(f)
        if fowner < 0:
            continue   # neutral fleet (shouldn't exist but just in case)

        # Find which of my planets this fleet is most likely targeting
        best_mine = None
        best_score = float('inf')
        for mp in my_planets:
            mx, my_ = mp[2], mp[3]
            # Does the fleet's heading roughly point at this planet?
            dx, dy = mx - fx, my_ - fy
            target_angle = math.atan2(dy, dx)
            angle_diff = abs((fangle - target_angle + math.pi) % (2*math.pi) - math.pi)
            if angle_diff < 0.35:   # within ~20 degrees
                d = dist(fx, fy, mx, my_)
                if d < best_score:
                    best_score = d
                    best_mine  = mp

        if best_mine is None:
            continue

        # Estimate when the fleet arrives
        spd   = fleet_speed(fships)
        eta   = best_score / spd if spd > 0 else 999
        # Ships on the planet at arrival (production each turn)
        pid, owner, px, py, pr, pships, pprod = _planet(best_mine)
        ships_at_arrival = pships + int(eta * pprod)
        ships_already_committed = used_ships.get(pid, 0)
        effective_garrison = ships_at_arrival - ships_already_committed

        if fships <= effective_garrison:
            continue   # we'll survive without help

        # How many extra ships do we need?
        deficit = int((fships - effective_garrison) * DEFEND_BUFFER) + 1

        # Find the nearest planet I own that can send help
        for mp2 in my_planets:
            if mp2[0] == pid:
                continue
            available = mp2[5] - used_ships.get(mp2[0], 0) - 1
            if available < MIN_SHIPS_TO_SEND:
                continue
            to_send = min(available, deficit)
            if to_send < MIN_SHIPS_TO_SEND:
                continue
            # Will it arrive in time?
            tt = _travel_turns(mp2, best_mine, to_send, av)
            if tt < eta + 2:   # +2 turns buffer
                moves.append(_make_move(mp2, best_mine, to_send, av))
                used_ships[mp2[0]] = used_ships.get(mp2[0], 0) + to_send
                deficit -= to_send
                if deficit <= 0:
                    break

    return moves


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPAND
# capture neutral planets, prioritising high production / low garrison / close
# ─────────────────────────────────────────────────────────────────────────────

def _expand_score(src, tgt, av):
    """Higher = better neutral to expand to."""
    prod  = tgt[6]
    ships = tgt[5]
    d     = _travel_turns(src, tgt, max(ships+2, 10), av)
    # Avoid sending fleets into the sun's shadow
    sun_d = dist(tgt[2], tgt[3], SUN_X, SUN_Y)
    sun_penalty = 0.5 if sun_d < 15 else 1.0
    return (prod * 10.0) / (d + ships + 1.0) * sun_penalty


def _expand_moves(my_planets, neutrals, av, used_ships):
    moves = []
    claimed = set()   # neutral planet ids already targeted this turn

    for src in sorted(my_planets, key=lambda p: p[5], reverse=True):
        available = src[5] - used_ships.get(src[0], 0) - 1
        if available < MIN_SHIPS_TO_SEND:
            continue

        candidates = [n for n in neutrals if n[0] not in claimed]
        if not candidates:
            break

        best = max(candidates, key=lambda n: _expand_score(src, n, av))
        ships_needed = int(best[5] * DEFEND_BUFFER) + 1
        to_send = min(available, int(available * EXPAND_SHIP_FRACTION))
        to_send = max(to_send, ships_needed)   # send at least enough to capture

        if to_send > available or to_send < MIN_SHIPS_TO_SEND:
            continue

        moves.append(_make_move(src, best, to_send, av))
        used_ships[src[0]] = used_ships.get(src[0], 0) + to_send
        claimed.add(best[0])

    return moves


# ─────────────────────────────────────────────────────────────────────────────
# 3. ATTACK
# attack the weakest reachable enemy planet when we have the advantage
# ─────────────────────────────────────────────────────────────────────────────

def _attack_moves(my_planets, enemies, av, used_ships, my_total, enemy_total):
    moves = []
    if my_total < enemy_total * 0.9:   # don't attack if we're behind
        return moves

    targeted = set()

    for src in sorted(my_planets, key=lambda p: p[5], reverse=True):
        available = src[5] - used_ships.get(src[0], 0) - 1
        if available < MIN_SHIPS_TO_SEND:
            continue

        candidates = [e for e in enemies if e[0] not in targeted]
        if not candidates:
            break

        # Pick enemy with best attack score (weak garrison, high production, close)
        def attack_score(e):
            d    = _travel_turns(src, e, max(e[5]+2, 10), av)
            return e[6] / (d + e[5] + 1.0)

        best = max(candidates, key=attack_score)
        ships_needed = int(best[5] * ATTACK_THRESHOLD) + 1
        if available < ships_needed:
            continue

        to_send = min(available, ships_needed + 5)
        moves.append(_make_move(src, best, to_send, av))
        used_ships[src[0]] = used_ships.get(src[0], 0) + to_send
        targeted.add(best[0])

    return moves


# ─────────────────────────────────────────────────────────────────────────────
# 4. REINFORCE
# move surplus ships from safe planets to frontline planets
# ─────────────────────────────────────────────────────────────────────────────

def _reinforce_moves(my_planets, all_planets, av, used_ships):
    moves = []
    if len(my_planets) < 2:
        return moves

    planet_map = {p[0]: p for p in all_planets}

    # Identify planets that are "rich but thin" vs "safe surplus"
    avg_ships = sum(p[5] for p in my_planets) / len(my_planets)

    thin_planets = [p for p in my_planets
                    if p[5] < avg_ships * REINFORCE_THRESHOLD and p[6] >= 3]
    rich_planets = [p for p in my_planets
                    if p[5] > avg_ships * 1.5]

    for dst in thin_planets:
        for src in sorted(rich_planets, key=lambda p: dist(p[2], p[3], dst[2], dst[3])):
            if src[0] == dst[0]:
                continue
            available = src[5] - used_ships.get(src[0], 0) - 1
            to_send   = available // 3   # send a third of surplus
            if to_send < MIN_SHIPS_TO_SEND:
                continue
            moves.append(_make_move(src, dst, to_send, av))
            used_ships[src[0]] = used_ships.get(src[0], 0) + to_send
            break   # one reinforcement source per thin planet

    return moves


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT FUNCTION
# This must be the last `def` in the file when used as a standalone submission
# ─────────────────────────────────────────────────────────────────────────────

def agent(obs):
    """
    Strong hardcoded Orbit Wars agent.

    Priorities: Defend → Expand → Attack → Reinforce
    Handles orbiting planets with intercept aiming.
    """
    player, planets, fleets, av = _parse_obs(obs)

    my_planets  = [p for p in planets if p[1] == player]
    neutrals    = [p for p in planets if p[1] == -1]
    enemies     = [p for p in planets if p[1] not in (-1, player) and p[1] >= 0]
    enemy_fleets= [f for f in fleets  if f[1] != player]

    if not my_planets:
        return []

    my_total    = sum(p[5] for p in my_planets) + sum(f[6] for f in fleets if f[1] == player)
    enemy_total = sum(p[5] for p in enemies)    + sum(f[6] for f in fleets if f[1] != player)

    used_ships = {}   # tracks ships committed per planet this turn
    moves = []

    # 1. Defend threatened planets
    moves += _defend_moves(my_planets, planets, enemy_fleets, av, used_ships)

    # 2. Expand to neutral planets
    if neutrals:
        moves += _expand_moves(my_planets, neutrals, av, used_ships)

    # 3. Attack enemy planets if we're ahead
    if enemies:
        moves += _attack_moves(my_planets, enemies, av, used_ships, my_total, enemy_total)

    # 4. Reinforce thin frontline planets
    moves += _reinforce_moves(my_planets, planets, av, used_ships)

    return moves


# ─────────────────────────────────────────────────────────────────────────────
# Quick offline test (no kaggle-environments needed)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fake_obs = {
        "player": 0,
        "angular_velocity": 0.03,
        "planets": [
            [0,  0, 20.0, 20.0, 2.0, 80, 3],   # mine — lots of ships
            [1, -1, 60.0, 60.0, 1.0, 10, 2],   # neutral
            [2,  1, 75.0, 25.0, 1.5, 20, 2],   # enemy
            [3,  0, 25.0, 75.0, 1.0, 30, 1],   # mine — secondary
        ],
        "fleets": [
            # enemy fleet heading roughly toward planet 0
            [0, 1, 40.0, 30.0, math.atan2(20-30, 20-40), 2, 50],
        ],
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
    }

    import math
    result = agent(fake_obs)
    print(f"Moves returned: {len(result)}")
    for m in result:
        print(f"  Planet {m[0]}  →  angle={m[1]:.3f}  ships={m[2]}")
    print("hardcoded_bot smoke test PASSED")
