#!/usr/bin/env bash
set -euo pipefail
EQGPT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$EQGPT_ROOT"
EQGPT_PYTHON="${EQGPT_PYTHON:-python3}"
"$EQGPT_PYTHON" -c 'import sys; assert sys.version_info[:2]==(3,11), "Select a Python 3.11 image or set EQGPT_PYTHON to a Python 3.11 executable"'
command -v cc >/dev/null || { echo 'C compiler required; choose an AutoDL development image with gcc.'; exit 1; }
"$EQGPT_PYTHON" -m venv .venv-eqgpt
.venv-eqgpt/bin/python -m pip install --upgrade pip
.venv-eqgpt/bin/python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
.venv-eqgpt/bin/python -m pip install -r requirements-eqgpt.lock.txt
export PYTHONPATH="$EQGPT_ROOT/src"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-eqgpt/bin/python scripts/eqgpt/verify_round1_bundle.py
.venv-eqgpt/bin/python scripts/eqgpt/preflight.py --require-cuda --output preflight_cuda.json
.venv-eqgpt/bin/python -m pytest tests/test_eqgpt.py tests/test_eqgpt_latex.py tests/test_eqgpt_round1.py -q
