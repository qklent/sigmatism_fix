#!/bin/bash
set -uo pipefail

echo "=== sigmatism_fix RunPod startup script ==="

# ── 1. System packages ──────────────────────────────────────────────────────
# espeak-ng: required by the phonemizer (text-side hard-S detection in gigaam mode).
# ffmpeg/sox/libsndfile1: audio I/O. libasound2: audio runtime.
echo "[1/6] Installing system packages..."
apt-get update -qq && apt-get install -y -qq espeak-ng ffmpeg sox libsndfile1 libasound2 git > /dev/null 2>&1

# ── 2. Clone repo ───────────────────────────────────────────────────────────
echo "[2/6] Cloning repo..."
REPO_DIR=/workspace/sigmatism_fix
GITHUB_URL="https://oauth2:${GITHUB_TOKEN}@github.com/qklent/sigmatism_fix.git"
if [ ! -d "$REPO_DIR" ]; then
    git clone "$GITHUB_URL" "$REPO_DIR"
else
    cd "$REPO_DIR"
    git pull --ff-only || true
fi
cd "$REPO_DIR"
git remote set-url origin "$GITHUB_URL"
git config --global user.name "qklent"
git config --global user.email "qklentkz@gmail.com"

# ── 3. Miniforge + mamba env ─────────────────────────────────────────────────
echo "[3/6] Setting up mamba environment..."
MINIFORGE_DIR=/root/miniforge3
if [ ! -f "$MINIFORGE_DIR/bin/mamba" ]; then
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$MINIFORGE_DIR"
    rm /tmp/miniforge.sh
fi
$MINIFORGE_DIR/bin/conda init bash > /dev/null 2>&1
eval "$($MINIFORGE_DIR/bin/conda shell.bash hook)"

if ! conda env list | grep -q "^sigmatism "; then
    $MINIFORGE_DIR/bin/mamba env create -f "$REPO_DIR/environment.yml" -y
else
    $MINIFORGE_DIR/bin/mamba env update -f "$REPO_DIR/environment.yml" --prune -y
fi
conda activate sigmatism

# ── 4. Install runtime extras (inference + resynthesis + app + serving + dev) ─
# NOT silenced — let install errors surface so we never end up with a broken
# ASR / TTS path that goes unnoticed.
echo "[4/6] Installing runtime extras..."
pip install -e ".[dev,inference,resynthesis,app,serving]"

# ── 5. Download MFA Russian models (only needed for aligner: "mfa") ──────────
echo "[5/6] Downloading MFA russian models..."
mfa model download acoustic russian_mfa 2>/dev/null || true
mfa model download dictionary russian_mfa 2>/dev/null || true

# ── 6. Environment variables ────────────────────────────────────────────────
echo "[6/6] Setting environment variables..."
if ! grep -q "# sigmatism_fix env" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'ENVEOF'

# sigmatism_fix env
export HF_HOME=/root/.cache/huggingface
export TORCH_HOME=/root/.cache/torch
export PATH="/root/miniforge3/envs/sigmatism/bin:$PATH"
export CONDA_DEFAULT_ENV=sigmatism
export CONDA_PREFIX=/root/miniforge3/envs/sigmatism

source /root/miniforge3/etc/profile.d/conda.sh
conda activate sigmatism
ENVEOF
fi

echo "=== Setup complete! Run 'source ~/.bashrc' or start a new shell. ==="
