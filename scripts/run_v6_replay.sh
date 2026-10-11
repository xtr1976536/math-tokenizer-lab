#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
.venv/bin/python -m pytest -q
.venv/bin/python scripts/fetch_pretrained_v6.py
.venv/bin/python scripts/audit_preflight_v6.py
.venv/bin/python scripts/stress_benchmark_v6.py
.venv/bin/python scripts/evaluate_full_v6.py
.venv/bin/python scripts/retime_full_v6.py
.venv/bin/python scripts/audit_full_v6.py
.venv/bin/python scripts/summarize_full_v6.py
