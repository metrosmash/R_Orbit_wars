# Orbit Wars RL

Reinforcement learning agent for the [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) competition.

## Strategy

Two-phase training pipeline:

1. **Imitation Learning (IL)** — supervised training on Kaggle replay dataset. The model learns to copy what winning bots did before touching RL.
2. **PPO Fine-tuning** — self-play reinforcement learning starting from the IL checkpoint. The model learns to go beyond imitation and win.

A strong hardcoded rule-based bot (`agent/hardcoded_bot.py`) serves as both the training opponent and automatic fallback if the model weights aren't found at submission time.

---

## Repo Structure

```
orbit-wars-rl/
├── setup_env.sh            ← run this first on a new machine
│
├── agent/                  ← all source code (submitted to Kaggle)
│   ├── main.py             ← submission entry point (agent() is last def)
│   ├── feature_utils.py    ← obs → fixed-size feature vector (618 values)
│   ├── action_utils.py     ← action encoding / intercept angle calculation
│   ├── hardcoded_bot.py    ← rule-based bot: Defend → Expand → Attack → Reinforce
│   ├── model.py            ← actor-critic neural network (PPO + IL compatible)
│   └── env_wrapper.py      ← Gymnasium wrapper with shaped rewards
│
├── training/               ← training scripts (not submitted)
│   ├── parse_replays.py    ← Kaggle replay JSON → numpy IL dataset
│   ├── train_il.py         ← Phase 1: imitation learning
│   └── train_ppo.py        ← Phase 2: PPO RL fine-tuning
│
├── checkpoints/            ← saved model weights (gitignored)
├── replays/                ← Kaggle replay JSON files (gitignored)
├── il_data/                ← parsed numpy training data (gitignored)
└── submissions/            ← packaged .tar.gz files for upload
```

---

## Setup

Tested on **Ubuntu 25.10 + AMD MI300X (ROCm 6.2)**.

```bash
git clone https://github.com/YOUR_USERNAME/orbit-wars-rl.git
cd orbit-wars-rl
bash setup_env.sh
```

The setup script:
- Installs PyTorch with ROCm 6.2
- Installs all RL dependencies
- Installs the Kaggle CLI
- Downloads the competition replay dataset (requires Kaggle auth)
- Parses replays into the IL training dataset automatically

**Kaggle authentication** (required for dataset download and submission):
```bash
# Download kaggle.json from https://www.kaggle.com/settings/api
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Training Pipeline

Activate the environment first in every terminal:
```bash
source ~/orbit_rl_env/bin/activate
cd orbit-wars-rl
```

### Step 1 — Parse replay dataset
Done automatically by `setup_env.sh` if replays are present. To run manually:
```bash
python training/parse_replays.py
```
Output: `il_data/states.npy` and `il_data/actions.npy`

### Step 2 — Imitation learning
```bash
python training/train_il.py
```
- Learns from winning moves in the replay dataset
- Saves best checkpoint to `checkpoints/il_best.pt`
- Typical runtime on MI300X: 2–4 hours

### Step 3 — PPO RL fine-tuning
```bash
python training/train_ppo.py --init_checkpoint checkpoints/il_best.pt
```
- Starts from the IL checkpoint (much faster convergence than from scratch)
- Saves best checkpoint to `checkpoints/ppo_best.pt`
- Monitor with TensorBoard: `tensorboard --logdir=checkpoints/logs --host=0.0.0.0 --port=6006`

### Step 4 — Submit to Kaggle
```bash
tar -czf submissions/submission_v1.tar.gz \
    agent/main.py \
    agent/model.py \
    agent/feature_utils.py \
    agent/action_utils.py \
    agent/hardcoded_bot.py \
    checkpoints/ppo_best.pt

kaggle competitions submit orbit-wars \
    -f submissions/submission_v1.tar.gz \
    -m "PPO v1"
```

---

## How the Submission Works

`agent/main.py` is the entry point. Per Kaggle's rules, `agent()` is the **last `def`** in the file.

At import time it tries to load model weights from `checkpoints/ppo_best.pt`. If weights aren't found it automatically falls back to the hardcoded rule-based bot — so you can submit at any stage and it will always work.

```
Kaggle calls agent(obs) each turn
         ↓
    weights found?
    ┌─── YES ──────────────────────────────────────────┐
    │  obs → feature vector (618,)                     │
    │  → neural network inference                      │
    │  → greedy action per planet slot                 │
    │  → decode to [from_id, angle, ships] moves       │
    └──────────────────────────────────────────────────┘
    ┌─── NO (fallback) ────────────────────────────────┐
    │  hardcoded_bot: Defend → Expand → Attack         │
    └──────────────────────────────────────────────────┘
```

---

## Architecture

### Feature Vector (618 values)
| Section | Size | What it encodes |
|---|---|---|
| Global | 8 | Ship counts, planet counts, production totals, turn number |
| Planets | 40 × 10 = 400 | Ownership, ships, production, distance, orbit status, position |
| Fleets | 30 × 7 = 210 | Ownership, ships, position, heading |

### Neural Network
```
Input (618,)
    │
    ├─ 3× [Linear(512) → LayerNorm → ReLU]
    │
    ├─► Actor:  Linear → (10 planets × 200 actions) logits
    └─► Critic: Linear → scalar value V(s)
```

### Action Space
Per owned planet: choose a **target** (40 slots) × **ship ratio** (0%, 25%, 50%, 75%, 100%) = 200 actions per planet, up to 10 planets = 2,000 combinations per turn.

### Reward Shaping
| Event | Reward |
|---|---|
| Neutral planet captured | +0.5 |
| Enemy planet captured | +1.0 |
| Own planet lost | −1.0 |
| Net ship gain per turn | +0.02 per ship |
| Win episode | +5.0 |
| Lose episode | −5.0 |

---

## Hardcoded Bot Strategy

`agent/hardcoded_bot.py` runs four priority-ordered rules each turn:

1. **Defend** — detect enemy fleets heading toward my planets by angle, send reinforcements if needed
2. **Expand** — capture neutral planets scored by `production / (distance + garrison)`
3. **Attack** — attack weakest enemy planet when total ship count favours us
4. **Reinforce** — move surplus ships from safe planets to thin frontline planets

All fleet targeting uses **intercept calculation** for orbiting planets — the fleet aims at where the planet *will be* when it arrives, not where it is now.

---

## Sanity Checks

Run individual modules to verify they work before training:
```bash
cd orbit-wars-rl
python agent/feature_utils.py    # → Feature vector shape: (618,) PASSED
python agent/action_utils.py     # → All action_utils checks PASSED
python agent/hardcoded_bot.py    # → hardcoded_bot smoke test PASSED
python agent/model.py            # → All model checks PASSED
python agent/env_wrapper.py      # → env_wrapper smoke test PASSED
```
