#!/bin/bash
# ============================================================
#  Lux AI Season 3 — DQN Environment Setup
#  AMD MI300X (ROCm) | Runs on every reboot
# ============================================================

set -e  # Exit immediately on any error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[✔] $1${NC}"; }
warn() { echo -e "${YELLOW}[!] $1${NC}"; }
fail() { echo -e "${RED}[✘] $1${NC}"; exit 1; }

echo ""
echo "============================================================"
echo "   Lux AI S3 — DQN Setup on AMD MI300X"
echo "============================================================"
echo ""

# ------------------------------------------------------------
# 1. System update
# ------------------------------------------------------------
log "Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y python3-venv python3-pip git wget

# ------------------------------------------------------------
# 2. Create virtual environment (skips if already exists)
# ------------------------------------------------------------
VENV_DIR="$HOME/lux_rl_env"

if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at $VENV_DIR — skipping creation"
else
    log "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
log "Virtual environment activated"

# ------------------------------------------------------------
# 3. Upgrade pip
# ------------------------------------------------------------
log "Upgrading pip..."
pip install --upgrade pip

# ------------------------------------------------------------
# 4. Install PyTorch with ROCm (AMD GPU support)
# ------------------------------------------------------------
log "Installing PyTorch with ROCm support for AMD MI300X..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# Verify GPU is detected
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU detected — running on CPU')
"

# ------------------------------------------------------------
# 5. Install Kaggle Environments + Lux AI S3
# ------------------------------------------------------------
log "Installing kaggle-environments..."
pip install --upgrade "kaggle-environments>=1.28.0"

log "Installing Lux AI Season 3 package..."
pip install --upgrade luxai-s3

# ------------------------------------------------------------
# 6. Install RL & Data Science stack
# ------------------------------------------------------------
log "Installing RL and data science packages..."
pip install \
    numpy \
    pandas \
    matplotlib \
    gymnasium \
    stable-baselines3 \
    tensorboard \
    tqdm \
    scipy \
    scikit-learn

# ------------------------------------------------------------
# 7. Verify key installs
# ------------------------------------------------------------
echo ""
log "Verifying installations..."
python3 -c "
import torch, numpy, kaggle_environments, luxai_s3
print(f'  PyTorch     : {torch.__version__}')
print(f'  NumPy       : {numpy.__version__}')
print(f'  Kaggle Envs : {kaggle_environments.__version__}')
"
LUXAI_VERSION=$(pip show luxai-s3 2>/dev/null | grep Version | awk '{print $2}')
echo "  LuxAI S3    : ${LUXAI_VERSION:-not found}"

# ------------------------------------------------------------
# 8. Print activation reminder
# ------------------------------------------------------------
echo ""
echo "============================================================"
log "Setup complete! Environment is ready."
echo ""
warn "Remember to activate your venv in every new terminal:"
echo "    source ~/lux_rl_env/bin/activate"
echo ""
warn "To start TensorBoard for training monitoring:"
echo "    tensorboard --logdir=./logs --host=0.0.0.0"
echo "============================================================"
echo ""
