#!/usr/bin/env bash
set -e

# Script to build OpenFace from source (tested with tag OpenFace_2.2.0)

sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libboost-all-dev libopencv-dev

git clone https://github.com/TadasBaltrusaitis/OpenFace.git --branch OpenFace_2.2.0 --depth 1
cd OpenFace
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

OPENFACE_BIN="$(pwd)/bin/FeatureExtraction"
if [[ ! -x "$OPENFACE_BIN" ]]; then
  echo "FeatureExtraction binary not found: $OPENFACE_BIN" >&2
  return 1 2>/dev/null || exit 1
fi

export OPENFACE_BIN

cat <<EOM

OpenFace built in $(pwd)
Environment variable set:
  OPENFACE_BIN=$OPENFACE_BIN
EOM
