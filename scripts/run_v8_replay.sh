#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
.venv/bin/python scripts/fetch_pretrained_v6.py
.venv/bin/python -m pytest -q > results/v8/code_review_round2_tests.log 2>&1
cat results/v8/code_review_round2_tests.log
# Reuse frozen models and sealed inputs. Test is now exposed: never refreeze,
# retrain/select variants, or describe this replay as fresh unseen evidence.
.venv/bin/python scripts/evaluate_v7.py --release v8
.venv/bin/python scripts/retime_v7.py --release v8
.venv/bin/python scripts/audit_results_v7.py --release v8
.venv/bin/python scripts/summarize_v7.py --release v8
.venv/bin/python scripts/multimetric_statistics_v8.py
.venv/bin/python scripts/finalize_intrinsic_v8.py
.venv/bin/python scripts/verify_release_v8.py
.venv/bin/python scripts/plot_final_v8.py
.venv/bin/python scripts/report_final_v8.py
