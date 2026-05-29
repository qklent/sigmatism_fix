#!/bin/bash
# Bootstrap the sigmatism_fix dev environment ON a RunPod pod.
#
# Design goal: everything that is expensive to recreate lives under /workspace,
# which survives pod stop/start. The container disk (/, /root) is WIPED on stop,
# so miniforge, the repo, and the HF cache all go under /workspace.
#
# Idempotent: safe to re-run after a stop/start (re-installs only what's missing).
# Requires env: GITHUB_TOKEN (clone), HF_TOKEN (model download), HF_HOME.
set -euo pipefail

WS=/workspace
REPO_DIR="$WS/sigmatism_fix"
MF="$WS/miniforge3"
export HF_HOME="${HF_HOME:-$WS/.cache/huggingface}"

echo "[1/5] System packages (container disk; cheap to reinstall after a stop)..."
apt-get update -qq && apt-get install -y -qq \
  espeak-ng ffmpeg sox libsndfile1 libasound2 git tmux curl >/dev/null

echo "[2/5] Repo under /workspace..."
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "https://oauth2:${GITHUB_TOKEN}@github.com/qklent/sigmatism_fix.git" "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --ff-only || true
fi

echo "[3/5] Miniforge under /workspace (persists across stop)..."
if [ ! -x "$MF/bin/conda" ]; then
  curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/mf.sh
  bash /tmp/mf.sh -b -p "$MF" && rm -f /tmp/mf.sh
fi
# shellcheck disable=SC1091
source "$MF/etc/profile.d/conda.sh"

echo "[4/5] Conda env 'sigmatism' (create or update) + editable install..."
if ! conda env list | grep -q "^sigmatism "; then
  "$MF/bin/mamba" env create -f "$REPO_DIR/environment.yml" -y
else
  "$MF/bin/mamba" env update -f "$REPO_DIR/environment.yml" --prune -y
fi
conda activate sigmatism
pip install -e "$REPO_DIR"[dev,inference,resynthesis,app]

echo "[5/5] MFA russian models (only used by aligner: \"mfa\")..."
mfa model download acoustic russian_mfa 2>/dev/null || true
mfa model download dictionary russian_mfa 2>/dev/null || true

echo "=== Bootstrap complete. HF_HOME=$HF_HOME, env=sigmatism, repo=$REPO_DIR ==="
echo "Next: bash $REPO_DIR/deploy/start_services.sh"
