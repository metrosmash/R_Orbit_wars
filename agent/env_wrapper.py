"""
env_wrapper.py
==============
Wraps the Orbit Wars kaggle-environment into a Gymnasium-compatible interface
so that standard RL libraries (Stable-Baselines3, custom PPO, etc.) can use it.

HOW GYMNASIUM WORKS (Plain English)
-------------------------------------
Gymnasium is the standard interface for RL environments. Every env has:

  obs, info  = env.reset()          ← start a new game, get first observation
  obs, reward, done, trunc, info = env.step(action)  ← take an action

Our wrapper does two things:
  1. Calls the kaggle orbit_wars engine under the hood.
  2. Converts obs ↔ our fixed feature vector,  action codes ↔ game moves.

REWARD SHAPING
--------------
Raw reward (win/lose at step 500) is too sparse for learning. We add:

  +0.5  per neutral planet captured this turn
  +1.0  per enemy planet captured this turn
  -1.0  per own planet lost this turn
  +0.02 per net ship increase this turn (production momentum)
  +5.0  for winning the episode
  -5.0  for losing (or being eliminated)

These are SHAPED rewards — they guide learning but don't change who wins.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from feature_utils  import obs_to_features, FEATURE_DIM, MAX_PLANETS
from action_utils   import (
    decode_action, actions_to_moves, build_action_mask,
    NUM_ACTIONS_PER_PLANET, MAX_MY_PLANETS
)
from hardcoded_bot  import agent as hardcoded_agent

# ── Reward shaping weights ────────────────────────────────────────────────────
R_NEUTRAL_CAPTURE  =  0.5
R_ENEMY_CAPTURE    =  1.0
R_PLANET_LOST      = -1.0
R_SHIP_DELTA       =  0.02   # per 1 net ship gained this turn
R_WIN              =  5.0
R_LOSE             = -5.0


# ─────────────────────────────────────────────────────────────────────────────
# OrbitWarsEnv
# ─────────────────────────────────────────────────────────────────────────────
class OrbitWarsEnv(gym.Env):
    """
    Single-agent wrapper around Orbit Wars.

    The "opponent" is configurable:
      - "hardcoded" : play against our hardcoded_bot (default, good for early training)
      - "self"      : play against a copy of the current policy (self-play)
      - "random"    : play against the built-in random agent

    Observation space : Box(FEATURE_DIM,) float32  in [-2, 2]
    Action space      : MultiDiscrete([NUM_ACTIONS_PER_PLANET] * MAX_MY_PLANETS)
                        One action code per planet slot (padded if < MAX_MY_PLANETS)

    Parameters
    ----------
    opponent        : "hardcoded" | "random" | callable agent function
    seed            : int or None
    max_steps       : episode length (default 500)
    render_mode     : None (no render) or "human" (prints basic stats each step)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        opponent    = "hardcoded",
        seed        = None,
        max_steps   = 500,
        render_mode = None,
    ):
        super().__init__()
        self.opponent    = opponent
        self.seed_val    = seed
        self.max_steps   = max_steps
        self.render_mode = render_mode

        # ── Spaces ────────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(FEATURE_DIM,),
            dtype=np.float32
        )
        # One discrete action per planet slot
        self.action_space = spaces.MultiDiscrete(
            [NUM_ACTIONS_PER_PLANET] * MAX_MY_PLANETS
        )

        # Internal state
        self._env        = None
        self._obs        = None
        self._step_count = 0
        self._prev_state = None   # for reward shaping delta

    # ─────────────────────────────────────────────────────────────────────────
    # reset
    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        from kaggle_environments import make

        seed_val = seed if seed is not None else self.seed_val
        cfg = {"seed": seed_val} if seed_val is not None else {}

        self._env = make("orbit_wars", configuration=cfg, debug=False)

        # Determine opponent agent string / callable
        if self.opponent == "hardcoded":
            opp = hardcoded_agent
        elif self.opponent == "random":
            opp = "random"
        elif callable(self.opponent):
            opp = self.opponent
        else:
            opp = "random"

        # We always play as player 0 for simplicity
        self._env.reset()
        self._raw_envs = self._env.train([None, opp])
        raw_obs = self._raw_envs.reset()

        self._obs        = raw_obs
        self._step_count = 0
        self._prev_state = self._snapshot(raw_obs)

        features = obs_to_features(raw_obs, step=0)
        return features, {}

    # ─────────────────────────────────────────────────────────────────────────
    # step
    # ─────────────────────────────────────────────────────────────────────────
    def step(self, action):
        """
        action : np.ndarray of shape (MAX_MY_PLANETS,)
                 Each element is an integer action code in [0, NUM_ACTIONS_PER_PLANET)
        """
        # ── Parse current observation ─────────────────────────────────────────
        raw_obs = self._obs
        if isinstance(raw_obs, dict):
            player  = raw_obs.get("player", 0)
            planets = raw_obs.get("planets", [])
            av      = raw_obs.get("angular_velocity", 0.03)
        else:
            player  = raw_obs.player
            planets = list(raw_obs.planets)
            av      = raw_obs.angular_velocity

        my_planets = [p for p in planets if p[1] == player]

        # Sort planets the same way feature_utils does (priority order)
        from feature_utils import planet_priority
        sorted_planets = sorted(
            planets,
            key=lambda p: planet_priority(p, player, my_planets),
            reverse=True
        )[:MAX_PLANETS]

        # ── Decode network actions → game moves ───────────────────────────────
        planet_actions = []
        for i in range(min(len(my_planets), MAX_MY_PLANETS)):
            code = int(action[i])
            planet_actions.append(decode_action(code))

        moves = actions_to_moves(planet_actions, my_planets, sorted_planets, av)

        # ── Step the environment ──────────────────────────────────────────────
        raw_obs_next, raw_reward, done, info = self._raw_envs.step(moves)

        self._step_count += 1
        self._obs = raw_obs_next

        # ── Reward shaping ────────────────────────────────────────────────────
        shaped_reward = self._shape_reward(
            raw_obs, raw_obs_next, raw_reward, done, player
        )

        # ── Features for next obs ─────────────────────────────────────────────
        features_next = obs_to_features(raw_obs_next, step=self._step_count)

        truncated = (self._step_count >= self.max_steps)

        if self.render_mode == "human":
            self._render_step(shaped_reward, done)

        return features_next, shaped_reward, bool(done), truncated, info or {}

    # ─────────────────────────────────────────────────────────────────────────
    # Reward shaping helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _snapshot(self, obs):
        """Capture the state we need for reward delta calculations."""
        if isinstance(obs, dict):
            player  = obs.get("player", 0)
            planets = obs.get("planets", [])
            fleets  = obs.get("fleets", [])
        else:
            player  = obs.player
            planets = list(obs.planets)
            fleets  = list(obs.fleets)

        my_planets = [p for p in planets if p[1] == player]
        my_ships   = sum(p[5] for p in my_planets) + \
                     sum(f[6] for f in fleets if f[1] == player)
        my_planet_ids = {p[0] for p in my_planets}

        return {
            "player"        : player,
            "my_planet_ids" : my_planet_ids,
            "my_ships"      : my_ships,
        }

    def _shape_reward(self, obs_before, obs_after, raw_reward, done, player):
        reward = 0.0

        try:
            snap_before = self._prev_state
            snap_after  = self._snapshot(obs_after)

            # Planets gained / lost
            gained = snap_after["my_planet_ids"] - snap_before["my_planet_ids"]
            lost   = snap_before["my_planet_ids"] - snap_after["my_planet_ids"]

            # Distinguish neutral captures from enemy captures
            if isinstance(obs_before, dict):
                planets_before = obs_before.get("planets", [])
            else:
                planets_before = list(obs_before.planets)

            owner_map = {p[0]: p[1] for p in planets_before}
            for pid in gained:
                prev_owner = owner_map.get(pid, -1)
                if prev_owner == -1:
                    reward += R_NEUTRAL_CAPTURE
                else:
                    reward += R_ENEMY_CAPTURE
            reward += len(lost) * R_PLANET_LOST

            # Ship delta momentum
            ship_delta = snap_after["my_ships"] - snap_before["my_ships"]
            reward += ship_delta * R_SHIP_DELTA

            self._prev_state = snap_after

        except Exception:
            pass   # never crash the env on reward shaping errors

        # Terminal reward
        if done:
            if raw_reward is not None:
                if raw_reward > 0:
                    reward += R_WIN
                elif raw_reward < 0:
                    reward += R_LOSE

        return float(reward)

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering (basic text)
    # ─────────────────────────────────────────────────────────────────────────
    def _render_step(self, reward, done):
        snap = self._prev_state
        print(f"  Step {self._step_count:3d} | "
              f"planets={len(snap['my_planet_ids'])} | "
              f"ships={snap['my_ships']:.0f} | "
              f"reward={reward:+.3f} | "
              f"{'DONE' if done else ''}")

    def render(self):
        if self.render_mode == "human":
            snap = self._prev_state
            print(f"[OrbitWarsEnv] Step={self._step_count}  "
                  f"planets={len(snap['my_planet_ids'])}  "
                  f"ships={snap['my_ships']:.0f}")

    def close(self):
        self._env = None


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check — runs one episode with random actions
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing OrbitWarsEnv with random actions...")
    try:
        env  = OrbitWarsEnv(opponent="random", seed=42)
        obs, info = env.reset()
        print(f"Obs shape after reset: {obs.shape}")   # should be (618,)

        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 50:   # only run 50 steps for speed
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        print(f"Steps run     : {steps}")
        print(f"Total reward  : {total_reward:.3f}")
        print(f"Done          : {done}")
        print(f"Obs shape     : {obs.shape}")
        print("env_wrapper smoke test PASSED")
        env.close()

    except ImportError as e:
        print(f"kaggle-environments not installed yet: {e}")
        print("This is expected before setup — env_wrapper is ready to use once installed.")
