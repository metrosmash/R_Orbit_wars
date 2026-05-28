#!/bin/bash
# ============================================================
#  Orbit Wars — DQN Environment Setup
#  AMD MI300X (ROCm 6.2) | Safe to run on every reboot
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✔] $1${NC}"; }
warn() { echo -e "${YELLOW}[!] $1${NC}"; }
fail() { echo -e "${RED}[✘] $1${NC}"; exit 1; }

echo ""
echo "============================================================"
echo "   Orbit Wars — DQN Setup on AMD MI300X"
echo "============================================================"
echo ""

# ------------------------------------------------------------
# 1. System update (update only — upgrade is manual)
#    Full upgrade is intentionally excluded here because it
#    can break ROCm kernel modules and takes 5-10 min.
#    Run 'sudo apt-get upgrade -y' manually when needed.
# ------------------------------------------------------------
log "Refreshing package lists..."
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git wget curl || true
log "System packages ready"

# ------------------------------------------------------------
# 2. Virtual environment (skips if already exists)
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
#    Skipped if already installed to save reboot time.
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

# Verify GPU detection
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
    # Quick smoke test
    x = torch.randn(512, 512, device='cuda')
    _ = x @ x.T
    print('  Smoke test   : PASSED')
else:
    print('  WARNING: No GPU detected — check ROCm installation')
    print('  Try: rocm-smi')
"

# ------------------------------------------------------------
# 5. Kaggle environments (Orbit Wars requires >= 1.28.0)
#    Skipped if correct version already present.
# ------------------------------------------------------------
REQUIRED_KE="1.28.0"
INSTALLED_KE=$(pip show kaggle-environments 2>/dev/null | grep ^Version | awk '{print $2}')

if [ -z "$INSTALLED_KE" ]; then
    log "Installing kaggle-environments..."
    pip install "kaggle-environments>=${REQUIRED_KE}"
else
    # Simple version check — reinstall if below minimum
    MEETS=$(python3 -c "
from packaging.version import Version
installed = '${INSTALLED_KE}'
required  = '${REQUIRED_KE}'
print('yes' if Version(installed) >= Version(required) else 'no')
" 2>/dev/null || echo "no")

    if [ "$MEETS" = "yes" ]; then
        warn "kaggle-environments $INSTALLED_KE already installed — skipping"
    else
        log "Upgrading kaggle-environments ($INSTALLED_KE → $REQUIRED_KE+)..."
        pip install --upgrade "kaggle-environments>=${REQUIRED_KE}"
    fi
fi

# ------------------------------------------------------------
# 6. RL & data science stack
#    Uses --upgrade so versions stay current across reboots.
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
# 7. Kaggle CLI (for submitting agents)
# ------------------------------------------------------------
if command -v kaggle &>/dev/null; then
    warn "Kaggle CLI already installed — skipping"
else
    log "Installing Kaggle CLI..."
    pip install --quiet kaggle
fi

# Remind about Kaggle API token if not present
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ ! -f "$HOME/.kaggle/access_token" ]; then
    warn "Kaggle API token not found at ~/.kaggle/"
    warn "Download it from https://www.kaggle.com/settings/api"
    warn "Then: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
fi

# ------------------------------------------------------------
# 8. Project directory setup
# ------------------------------------------------------------
PROJECT_DIR="$HOME/orbit_wars"

if [ -d "$PROJECT_DIR" ]; then
    warn "Project directory already exists at $PROJECT_DIR — skipping"
else
    log "Creating project directory structure..."
    mkdir -p "$PROJECT_DIR"/{agent,training,utils,checkpoints,logs,submissions,docs}
    log "Project directory created at $PROJECT_DIR"
fi

# If git repo URL is set, clone it
# Uncomment and set your repo URL after you create it:
# GIT_REPO="https://github.com/YOUR_USERNAME/orbit-wars-rl.git"
# if [ -n "$GIT_REPO" ] && [ ! -d "$PROJECT_DIR/.git" ]; then
#     log "Cloning repo..."
#     git clone "$GIT_REPO" "$PROJECT_DIR"
# fi

# ------------------------------------------------------------
# 9. Verify everything
# ------------------------------------------------------------
echo ""
log "Running full verification..."
python3 -c "
import sys

checks = []

# PyTorch
try:
    import torch
    checks.append(('PyTorch',        torch.__version__, True))
    checks.append(('CUDA/ROCm',      str(torch.cuda.is_available()), torch.cuda.is_available()))
except ImportError as e:
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

# wandb
try:
    import wandb
    checks.append(('wandb', wandb.__version__, True))
except ImportError:
    checks.append(('wandb', 'NOT FOUND', False))

# stable-baselines3
try:
    import stable_baselines3
    checks.append(('stable-baselines3', stable_baselines3.__version__, True))
except ImportError:
    checks.append(('stable-baselines3', 'NOT FOUND', False))

print()
all_ok = True
for name, version, ok in checks:
    status = '✔' if ok else '✘'
    color  = '\033[0;32m' if ok else '\033[0;31m'
    reset  = '\033[0m'
    print(f'  {color}[{status}]{reset} {name:<25}: {version}')
    if not ok:
        all_ok = False

print()
if all_ok:
    print('\033[0;32m  All checks passed — ready to train!\033[0m')
else:
    print('\033[0;31m  Some checks failed — review above.\033[0m')
    sys.exit(1)
"

# ------------------------------------------------------------
# 10. Done
# ------------------------------------------------------------
echo ""
echo "============================================================"
log "Setup complete!"
echo ""
warn "Activate your environment in every new terminal:"
echo "    source ~/orbit_rl_env/bin/activate"
echo ""
warn "Run verification anytime:"
echo "    python ~/orbit_wars/verify_setup.py"
echo ""
warn "Start training:"
echo "    cd ~/orbit_wars && python training/train_dqn.py"
echo ""
warn "Monitor with TensorBoard:"
echo "    tensorboard --logdir=~/orbit_wars/logs --host=0.0.0.0 --port=6006"
echo ""
warn "Submit to Kaggle:"
echo "    kaggle competitions submit orbit-wars -f submissions/main.py -m 'v1'"
echo "============================================================"
echo ""
