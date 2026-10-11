#!/bin/sh
set -eu
current_output="${1:-results/current_reproduction_new}"
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/train_current.py --output "$current_output"
.venv/bin/python scripts/replay_current.py "$current_output"
