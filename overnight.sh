#!/usr/bin/env bash
# Starts both overnight training jobs in a tmux session with two side-by-side panes.
# Usage: bash overnight.sh
# Reattach later: tmux attach -t overnight

SESSION="overnight"
WORKDIR="$(cd "$(dirname "$0")/Liars poker" && pwd)"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"

# Kill any existing session with this name
tmux kill-session -t "$SESSION" 2>/dev/null

# Create new session, first pane: CFR+
tmux new-session -d -s "$SESSION" -x 220 -y 50 \
  -c "$WORKDIR" \
  "$PYTHON -m agent.baseline.cfr_1v1_overnight \
    --name cfr_plus_mb4_hh \
    --max-bids 4 \
    --batch 100 \
    --total-iters 50000 \
    --eval-every 2; echo 'CFR+ DONE — press any key'; read"

# Split vertically, second pane: R-NaD
tmux split-window -h -t "$SESSION" -c "$WORKDIR" \
  "$PYTHON -m agent.rnad.trainer --stage B --iterations 50000; \
   echo 'R-NaD DONE — press any key'; read"

# Even up the pane sizes
tmux select-layout -t "$SESSION" even-horizontal

# Add status bar labels
tmux select-pane -t "$SESSION:0.0" -T "CFR+ (n=2 Nash)"
tmux select-pane -t "$SESSION:0.1" -T "R-NaD (Stage B)"

# Attach
tmux attach -t "$SESSION"
