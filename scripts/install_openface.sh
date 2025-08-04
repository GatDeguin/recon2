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

cat <<EOM

OpenFace built in $(pwd)
Set the following environment variable:
  export OPENFACE_BIN=$(pwd)/bin/FeatureExtraction
EOM
