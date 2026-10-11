#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_prefix="${1:-results/v3_reproduce}"
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/experiment_v3.py --output "${run_prefix}_512" --capacity 512
.venv/bin/python scripts/experiment_v3.py --output "${run_prefix}_1024" --capacity 1024
.venv/bin/python scripts/retime_v3.py "${run_prefix}_512"
.venv/bin/python scripts/retime_v3.py "${run_prefix}_1024"
.venv/bin/python scripts/finalize_v3_artifacts.py "${run_prefix}_512"
.venv/bin/python scripts/finalize_v3_artifacts.py "${run_prefix}_1024"
.venv/bin/python scripts/audit_v3.py "${run_prefix}_512"
.venv/bin/python scripts/audit_v3.py "${run_prefix}_1024"
