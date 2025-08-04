#!/usr/bin/env bash
set -e

# Script to install the RAFT optical flow dependency
# Tested with commit 3fa0bb0a9c633ea0a9bb8a79c576b6785d4e6a02

git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
pip install -r requirements.txt
bash download_models.sh

cat <<EOM

RAFT installed in $(pwd)
Set the following environment variables:
  export RAFT_DIR=$(pwd)
  export RAFT_CHECKPOINT=$(pwd)/models/raft-sintel.pth
EOM
