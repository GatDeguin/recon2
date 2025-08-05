#!/usr/bin/env bash
set -e

# Script to install the RAFT optical flow dependency
# Tested with commit 3fa0bb0a9c633ea0a9bb8a79c576b6785d4e6a02

git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
pip install -r requirements.txt
bash download_models.sh

RAFT_DIR="$(pwd)"
RAFT_CHECKPOINT="$RAFT_DIR/models/raft-sintel.pth"

if [[ ! -d "$RAFT_DIR" ]]; then
  echo "RAFT directory not found: $RAFT_DIR" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "$RAFT_CHECKPOINT" ]]; then
  echo "Checkpoint not found: $RAFT_CHECKPOINT" >&2
  return 1 2>/dev/null || exit 1
fi

export RAFT_DIR RAFT_CHECKPOINT

cat <<EOM

RAFT installed in $RAFT_DIR
Environment variables set:
  RAFT_DIR=$RAFT_DIR
  RAFT_CHECKPOINT=$RAFT_CHECKPOINT
EOM
