#!/bin/bash
# Start sigmatism_fix services ON the pod, detached in tmux, with logs to files
# so they can be tailed/grepped over SSH (this MCP can't fetch pod logs).
#
# Today: the Gradio app (single PyTorch process). Later: a Triton session is added
# here as a second tmux window, with the app pointed at localhost Triton.
set -euo pipefail

WS=/workspace
REPO_DIR="$WS/sigmatism_fix"
LOG_DIR="$WS/logs"
mkdir -p "$LOG_DIR"

# shellcheck disable=SC1091
source "$WS/miniforge3/etc/profile.d/conda.sh"
conda activate sigmatism

# --- app (Gradio) ---
tmux kill-session -t app 2>/dev/null || true
tmux new-session -d -s app \
  "cd $REPO_DIR && python scripts/run_gradio_app.py --host 0.0.0.0 --port 7860 2>&1 | tee $LOG_DIR/app.log"

echo "Gradio app launching in tmux session 'app'."
echo "  logs:    tail -f $LOG_DIR/app.log"
echo "  attach:  tmux attach -t app"
echo "  UI:      https://<POD_ID>-7860.proxy.runpod.net   (first run downloads models)"
