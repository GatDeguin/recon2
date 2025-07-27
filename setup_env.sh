#!/usr/bin/env bash
# Setup Python virtual environment for the project
# Usage: source setup_env.sh
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Extra packages not listed in requirements
pip install ultralytics spacy tqdm pillow onnxruntime
# Download small Spanish model for spaCy
python -m spacy download es_core_news_sm

echo "Environment ready. Activate with: source .venv/bin/activate"
