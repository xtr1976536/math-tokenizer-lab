#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_prefix="${1:-results/v2_reproduce}"
.venv/bin/python -m pytest tests -q
.venv/bin/python -m math_tokenizer.experiment_v2 --output "${run_prefix}_512" --capacity 512 --seeds 17 29 43
.venv/bin/python -m math_tokenizer.experiment_v2 --output "${run_prefix}_1024" --capacity 1024 --seeds 17 29 43
.venv/bin/python scripts/retime_v2.py "${run_prefix}_512"
.venv/bin/python scripts/retime_v2.py "${run_prefix}_1024"
.venv/bin/python scripts/audit_v2.py "${run_prefix}_512"
.venv/bin/python scripts/audit_v2.py "${run_prefix}_1024"
