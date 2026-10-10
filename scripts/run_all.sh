#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_prefix="${1:-results/new_run}"
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/fetch_baselines.py
.venv/bin/python -m math_tokenizer.run --output "${run_prefix}_384" --capacity 384 --seeds 17 29 43
.venv/bin/python -m math_tokenizer.run --output "${run_prefix}_512" --capacity 512 --seeds 17 29 43
.venv/bin/python -m math_tokenizer.audit "${run_prefix}_384"
.venv/bin/python -m math_tokenizer.audit "${run_prefix}_512"
