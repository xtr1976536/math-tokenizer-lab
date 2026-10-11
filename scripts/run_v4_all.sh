#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_prefix="${1:-results/v4_reproduce}"
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/experiment_v4.py --capacity 512 --output "${run_prefix}_512"
.venv/bin/python scripts/experiment_v4.py --capacity 1024 --output "${run_prefix}_1024"
.venv/bin/python scripts/add_initial_control_v4.py "${run_prefix}_512"
.venv/bin/python scripts/add_initial_control_v4.py "${run_prefix}_1024"
.venv/bin/python scripts/retime_v4.py "${run_prefix}_512"
.venv/bin/python scripts/retime_v4.py "${run_prefix}_1024"
.venv/bin/python scripts/replay_v4.py "${run_prefix}_512"
.venv/bin/python scripts/replay_v4.py "${run_prefix}_1024"
.venv/bin/python scripts/audit_v4.py "${run_prefix}_512"
.venv/bin/python scripts/audit_v4.py "${run_prefix}_1024"
