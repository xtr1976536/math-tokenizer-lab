# Current release reproduction

The active registry is `models/registry.json`. Load with `math_tokenizer.final_models.load_final`. Only the two selected routes and five baselines are accepted. Legacy source modules remain as implementation dependencies, not discoverable model candidates. Historical recipes and tests are in `history/`; use commit 5217a5a for an exact replay of original paths.

Use Python 3.11 and `requirements-v8.lock.txt`; install the package editable with `pip install -e .`. C/C++ cores compile from source into `.native_cache` and require a compiler and Python headers. Original Meta tokenizer source, license and byte-rank files are pinned in `models/baselines/Meta-Llama-3-8B/`; Qwen's selected configuration is stored in its baseline directory. tiktoken uses its pinned package and encoding data. A package installation/network fetch may be needed in a new environment.

Run:

```sh
python scripts/audit_current.py
python scripts/summarize_current.py
python -m pytest -q
python scripts/train_current.py --scenario broad --capacity 1024 --output /tmp/math-tokenizer-retrain
```

The training recipe uses only allowed fit and auxiliary IDs, derives mathematical primitives from fit only and uses original HF BPE. The Boundary AST uses fixed .25 tree and .30 boundary costs. It verifies the new vocabulary against frozen IDs. It never loads test feedback or trains an RL policy.

Build the presentation with `python scripts/plot_presentation.py`, `python scripts/build_presentation.py`, then XeLaTeX/latexmk from the source directory. Use `presentations/math_tokenizer/source/mathematical_tokenizer.tex`; the GBFR template and all three vector figures are included. The cover's October 10, 2026 date is retained by explicit user request; research audit/search dates are October 13, 2026.

The full catalog must not be used as a training set. Refer to its immutable ID manifests. Automated checks validate source syntax and provenance, not the truth of equations or universal mathematical semantics. Current evidence is intrinsic and single-machine; downstream capability remains an experiment to perform.
