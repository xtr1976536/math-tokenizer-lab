#!/usr/bin/env bash
set -euo pipefail
EQGPT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$EQGPT_ROOT"
python3 -c 'import sys; assert (3,11)<=sys.version_info[:2]<(3,13), "Use Python 3.11 or 3.12"'
python3 -m venv .venv-eqgpt
.venv-eqgpt/bin/python -m pip install --upgrade pip
.venv-eqgpt/bin/python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
.venv-eqgpt/bin/python -m pip install -r docs/eqgpt/local_environment.lock.txt
.venv-eqgpt/bin/python -m pip install --no-deps -e .
export PYTHONPATH="$EQGPT_ROOT/src"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-eqgpt/bin/python scripts/eqgpt/preflight.py --require-cuda
