#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_output="${1:-results/v5_reproduce1024}"
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/experiment_v5.py --capacity 1024 --output "$run_output"
.venv/bin/python scripts/retime_v5.py "$run_output"
.venv/bin/python scripts/replay_v5.py "$run_output"
.venv/bin/python scripts/audit_v5.py "$run_output"
.venv/bin/python scripts/report_v5.py "$run_output"
