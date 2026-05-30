#!/bin/bash
# ============================================================
#  Orbit Wars — Full Environment Setup
#  AMD MI300X (ROCm 6.2) | Safe to run on every reboot
#
#  Repo structure this script expects / creates:
#    orbit-wars-rl/
#    ├── setup_env.sh        ← this file (run from repo root)
#    ├── agent/              ← main.py + all importable modules
#    ├── training/           ← train_il.py, train_ppo.py, parse_replays.py
#    ├── checkpoints/        ← saved model weights (gitignored)
#    ├── replays/            ← kaggle replay JSON files (gitignored)
#    ├── il_data/            ← parsed numpy training data (gitignored)
#    └── submissions/        ← packaged .tar.gz files for upload
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✔] $1${NC}"; }
warn() { echo -e "${YELLOW}[!] $1${NC}"; }
fail() { echo -e "${RED}[✘] $1${NC}"; exit 1; }

# Resolve repo root (directory containing this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log "Repo root: $REPO_ROOT"

echo ""
echo "============================================================"
echo "   Orbit Wars — Full Setup on AMD MI300X"
echo "============================================================"
echo ""

# ------------------------------------------------------------
# 1. System update
# ------------------------------------------------------------
log "Refreshing package lists..."
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git wget curl pigz || true
log "System packages ready"

# ------------------------------------------------------------
# 2. Virtual environment
# ------------------------------------------------------------
VENV_DIR="$HOME/orbit_rl_env"

if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at $VENV_DIR — skipping creation"
else
    log "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
log "Virtual environment activated: $VENV_DIR"

# ------------------------------------------------------------
# 3. Upgrade pip
# ------------------------------------------------------------
log "Upgrading pip..."
pip install --upgrade pip --quiet

# ------------------------------------------------------------
# 4. PyTorch with ROCm 6.2 (MI300X requires 6.2+)
# ------------------------------------------------------------
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    warn "PyTorch already installed ($TORCH_VER) — skipping"
else
    log "Installing PyTorch with ROCm 6.2 support..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.2
    log "PyTorch installed"
fi

log "Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f'  GPUs found   : {n}')
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram  = props.total_memory / 1e9
        print(f'  GPU {i}        : {props.name}  ({vram:.1f} GB VRAM)')
    x = torch.randn(512, 512, device='cuda')
    _ = x @ x.T
    print('  Smoke test   : PASSED')
else:
    print('  WARNING: No GPU detected — check ROCm installation')
    print('  Try: rocm-smi')
"

# ------------------------------------------------------------
# 5. kaggle-environments (Orbit Wars requires >= 1.28.0)
# ------------------------------------------------------------
REQUIRED_KE="1.28.0"
INSTALLED_KE=$(pip show kaggle-environments 2>/dev/null | grep ^Version | awk '{print $2}')

if [ -z "$INSTALLED_KE" ]; then
    log "Installing kaggle-environments..."
    pip install "kaggle-environments>=${REQUIRED_KE}"
else
    MEETS=$(python3 -c "
from packaging.version import Version
print('yes' if Version('${INSTALLED_KE}') >= Version('${REQUIRED_KE}') else 'no')
" 2>/dev/null || echo "no")
    if [ "$MEETS" = "yes" ]; then
        warn "kaggle-environments ${INSTALLED_KE} already installed — skipping"
    else
        log "Upgrading kaggle-environments..."
        pip install --upgrade "kaggle-environments>=${REQUIRED_KE}"
    fi
fi

# ------------------------------------------------------------
# 6. RL & data science stack
# ------------------------------------------------------------
log "Installing RL and data science packages..."
pip install --upgrade --quiet \
    numpy \
    pandas \
    matplotlib \
    gymnasium \
    stable-baselines3 \
    tensorboard \
    wandb \
    tqdm \
    scipy \
    scikit-learn \
    packaging

log "RL stack installed"

# ------------------------------------------------------------
# 7. Kaggle CLI
# ------------------------------------------------------------
if command -v kaggle &>/dev/null; then
    warn "Kaggle CLI already installed — skipping"
else
    log "Installing Kaggle CLI..."
    pip install --quiet kaggle
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ ! -f "$HOME/.kaggle/access_token" ]; then
    warn "Kaggle API token not found at ~/.kaggle/"
    warn "Download from https://www.kaggle.com/settings/api"
    warn "Then: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
fi

# ------------------------------------------------------------
# 8. Ensure repo directories exist (safe if already present)
# ------------------------------------------------------------
log "Ensuring repo directories exist..."
mkdir -p \
    "$REPO_ROOT/agent" \
    "$REPO_ROOT/training" \
    "$REPO_ROOT/checkpoints" \
    "$REPO_ROOT/replays" \
    "$REPO_ROOT/il_data" \
    "$REPO_ROOT/submissions"

# Create .gitkeep files for empty tracked dirs (idempotent)
for dir in checkpoints replays il_data submissions; do
    touch "$REPO_ROOT/$dir/.gitkeep"
done
log "Repo directories ready"

# ------------------------------------------------------------
# 9. Download Kaggle dataset (replay files for imitation learning)
#    Only runs if replays/ is empty AND Kaggle CLI is authenticated.
# ------------------------------------------------------------
REPLAY_DIR="$REPO_ROOT/replays"
REPLAY_COUNT=$(ls "$REPLAY_DIR"/*.json 2>/dev/null | wc -l)

if [ "$REPLAY_COUNT" -gt "0" ]; then
    warn "Replays already present ($REPLAY_COUNT files) — skipping download"
else
    log "Checking Kaggle authentication..."
    if kaggle competitions list -s "orbit wars" &>/dev/null; then
        log "Downloading Orbit Wars dataset (replay files)..."
        kaggle competitions download orbit-wars -p "$REPLAY_DIR" --quiet || \
            warn "Download failed — check Kaggle token and that you have joined the competition"

        # Extract zip archives
        if ls "$REPLAY_DIR"/*.zip &>/dev/null; then
            log "Extracting .zip archives..."
            for z in "$REPLAY_DIR"/*.zip; do
                unzip -q "$z" -d "$REPLAY_DIR" && rm "$z"
            done
        fi

        # Extract tar.gz archives
        if ls "$REPLAY_DIR"/*.tar.gz &>/dev/null; then
            log "Extracting .tar.gz archives..."
            for z in "$REPLAY_DIR"/*.tar.gz; do
                tar -xzf "$z" -C "$REPLAY_DIR" && rm "$z"
            done
        fi

        REPLAY_COUNT=$(ls "$REPLAY_DIR"/*.json 2>/dev/null | wc -l)
        log "Replays ready: $REPLAY_COUNT JSON files"
    else
        warn "Kaggle CLI not authenticated — skipping dataset download"
        warn "Authenticate first: kaggle auth login"
        warn "Then re-run this script, or manually:"
        warn "  kaggle competitions download orbit-wars -p $REPLAY_DIR"
    fi
fi

# ------------------------------------------------------------
# 10. Parse replays → IL dataset
#     Only runs if replays exist and il_data/states.npy is missing.
# ------------------------------------------------------------
IL_STATES="$REPO_ROOT/il_data/states.npy"
REPLAY_COUNT=$(ls "$REPLAY_DIR"/*.json 2>/dev/null | wc -l)

if [ -f "$IL_STATES" ]; then
    warn "IL dataset already exists — skipping parsing"
elif [ "$REPLAY_COUNT" -gt "0" ]; then
    log "Parsing $REPLAY_COUNT replay files into IL dataset (this takes a few minutes)..."
    python3 "$REPO_ROOT/training/parse_replays.py" \
        --replay_dir "$REPLAY_DIR" \
        --out_dir    "$REPO_ROOT/il_data"
    log "IL dataset ready at $REPO_ROOT/il_data"
else
    warn "No replay files found — skipping IL dataset creation"
    warn "After downloading replays, run:"
    warn "  python training/parse_replays.py"
fi

# ------------------------------------------------------------
# 11. Full verification
# ------------------------------------------------------------
echo ""
log "Running full verification..."
python3 - <<'PYEOF'
import sys, os

# Add agent/ to path for verification
repo_root = os.environ.get("REPO_ROOT", os.getcwd())
agent_dir = os.path.join(repo_root, "agent")
for p in [agent_dir, repo_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

checks = []

# PyTorch
try:
    import torch
    checks.append(('PyTorch',    torch.__version__, True))
    checks.append(('CUDA/ROCm',  str(torch.cuda.is_available()), torch.cuda.is_available()))
except ImportError:
    checks.append(('PyTorch', 'NOT FOUND', False))

# NumPy
try:
    import numpy
    checks.append(('NumPy', numpy.__version__, True))
except ImportError:
    checks.append(('NumPy', 'NOT FOUND', False))

# kaggle-environments
try:
    import kaggle_environments
    checks.append(('kaggle-environments', kaggle_environments.__version__, True))
except ImportError:
    checks.append(('kaggle-environments', 'NOT FOUND', False))

# Orbit Wars env
try:
    from kaggle_environments import make
    env = make('orbit_wars', configuration={'seed': 1}, debug=False)
    checks.append(('orbit_wars env', 'OK', True))
except Exception as e:
    checks.append(('orbit_wars env', f'FAILED: {e}', False))

# Our modules
for mod in ['feature_utils', 'action_utils', 'hardcoded_bot', 'model', 'env_wrapper']:
    try:
        __import__(mod)
        checks.append((f'agent/{mod}', 'OK', True))
    except Exception as e:
        checks.append((f'agent/{mod}', f'FAILED: {e}', False))

# feature_utils sanity check
try:
    from feature_utils import obs_to_features, FEATURE_DIM
    fake = {'player': 0, 'planets': [[0, 0, 20., 20., 2., 30, 3]],
            'fleets': [], 'angular_velocity': 0.03,
            'initial_planets': [], 'comet_planet_ids': [], 'comets': []}
    f = obs_to_features(fake, step=1)
    assert f.shape == (FEATURE_DIM,)
    checks.append(('feature vector shape', f'({FEATURE_DIM},) ✔', True))
except Exception as e:
    checks.append(('feature vector shape', f'FAILED: {e}', False))

# IL dataset
il_path = os.path.join(repo_root, 'il_data', 'states.npy')
if os.path.exists(il_path):
    import numpy as np
    s = np.load(il_path)
    checks.append(('IL dataset', f'{len(s):,} samples', True))
else:
    checks.append(('IL dataset', 'not yet created (ok)', None))

# wandb + stable-baselines3
for lib in ['wandb', 'stable_baselines3']:
    try:
        m = __import__(lib)
        checks.append((lib, m.__version__, True))
    except ImportError:
        checks.append((lib, 'NOT FOUND', False))

print()
all_ok = True
for name, version, ok in checks:
    if ok is None:
        sym, color = '~', '\033[1;33m'
    elif ok:
        sym, color = '✔', '\033[0;32m'
    else:
        sym, color = '✘', '\033[0;31m'
        all_ok = False
    print(f'  {color}[{sym}]\033[0m {name:<32}: {version}')

print()
if all_ok:
    print('\033[0;32m  All checks passed — ready to train!\033[0m')
else:
    print('\033[0;31m  Some checks failed — review above.\033[0m')
    sys.exit(1)
PYEOF

# ------------------------------------------------------------
# 12. Done
# ------------------------------------------------------------
echo ""
echo "============================================================"
log "Setup complete!"
echo ""
warn "Activate your environment in every new terminal:"
echo "    source ~/orbit_rl_env/bin/activate"
echo "    cd $REPO_ROOT"
echo ""
warn "TRAINING PIPELINE (run from repo root):"
echo ""
echo "  Step 1 — Parse replays (done automatically above if replays exist):"
echo "    python training/parse_replays.py"
echo ""
echo "  Step 2 — Imitation learning (~2-4 hrs on MI300X):"
echo "    python training/train_il.py"
echo ""
echo "  Step 3 — RL fine-tuning (start from IL checkpoint):"
echo "    python training/train_ppo.py --init_checkpoint checkpoints/il_best.pt"
echo ""
echo "  Step 4 — Package and submit:"
echo "    tar -czf submissions/submission_v1.tar.gz \\"
echo "        agent/main.py agent/model.py agent/feature_utils.py \\"
echo "        agent/action_utils.py agent/hardcoded_bot.py \\"
echo "        checkpoints/ppo_best.pt"
echo "    kaggle competitions submit orbit-wars \\"
echo "        -f submissions/submission_v1.tar.gz -m 'PPO v1'"
echo ""
warn "Monitor training with TensorBoard:"
echo "    tensorboard --logdir=checkpoints/logs --host=0.0.0.0 --port=6006"
echo "============================================================"
echo ""
