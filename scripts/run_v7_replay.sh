#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
.venv/bin/python scripts/fetch_pretrained_v6.py
.venv/bin/python -m pytest -q > results/v7/final_tests.log 2>&1
cat results/v7/final_tests.log
# Reuse sealed data and frozen models; v7 test is now exposed. Do not refreeze
# development selection or describe reruns as a new unseen-test experiment.
.venv/bin/python scripts/evaluate_v7.py
.venv/bin/python scripts/retime_v7.py
.venv/bin/python scripts/audit_results_v7.py
.venv/bin/python scripts/summarize_v7.py
.venv/bin/python scripts/verify_release_v7.py
.venv/bin/python scripts/report_v7.py
