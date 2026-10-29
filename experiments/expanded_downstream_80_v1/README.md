# Expanded-corpus downstream experiment, 80-token capacity

This is a separate, completed experiment. It uses the same eight evaluation cases and four model arms as `experiments/downstream/`, while changing the training corpus, augmentation, shared sequence capacity, Hard input spelling and Hard LaTeX parsing fallback. See [`reports/expanded_downstream_80_v1/comparison_previous_zh.md`](../../reports/expanded_downstream_80_v1/comparison_previous_zh.md) before interpreting differences: this is not a corpus-size-only ablation.

## Reproduction record

- Common source pool: 1,855 formulas (1,643 external benchmark + all 212 original EqGPT non-target formulas); 59,360 paired augmented sequences.
- Arms: EqGPT GRE 57 ID and Hard AST-BPE 280/315/350 ID.
- Shared Transformer: 3 layers, 25 epochs, batch 128, Adam at 1e-4, seed 525; 80 positions and search generation limit.
- Augmentation: 32 sampled equivalent permutations per source, fixed seed, EOS at the end.
- Search: eight cases × four arms; three cycles × 200 attempts, two continuation epochs per cycle.
- Existing MLP caches were read-only reused; no PINN or independent coefficient fit was run.

The final state is `COMPLETE`. `RUN.json` binds the run to corpus hashes, approved settings, runtime-source hashes, and device. `shared_pretrain/` contains the four 25-epoch training reports and the common-weight manifest. Each `cases/<case>/<arm>/` contains the search input binding, full three-cycle candidate history, and the same per-arm `fast_summary.json` artifact used by the older experiment; each case has a `complete.json`. `status.json` records orchestration completion. Audit reports and summary tables are in `reports/expanded_downstream_80_v1/`. The original verbose terminal progress logs were not retained in the final run package; epoch losses and candidate-by-candidate outcomes are preserved in the structured training and search JSON files.

## Large checkpoints

The 36 `.pt` checkpoint files (four shared pretrained weights and 32 searched weights; about 2.78 GB total) follow the same distribution pattern as the older downstream run: they are published as individual assets in the public GitHub Release [`expanded-downstream-80-v1-weights`](https://github.com/xtr1976536/math-tokenizer-lab/releases/tag/expanded-downstream-80-v1-weights), while remaining excluded from Git history. `weight_manifest.json` records each relative path, release asset, byte size, and SHA-256. Download and verify all checkpoints from the repository root with:

```bash
PYTHONPATH=src python scripts/eqgpt/download_expanded_80_weights.py
```

Verify already downloaded local checkpoints without network access:

```bash
PYTHONPATH=src python scripts/eqgpt/download_expanded_80_weights.py --verify-only
```

Search JSON records contain the tensor hashes of the corresponding weights. The manifest records the serialized-file hash and size.

## The two formula benchmarks

The older benchmark is the 221-formula EqGPT handbook collection. The new external benchmark is separate and has 2,045 screened formulas before the run-specific length filter. This run selects 1,643 of those formulas and combines them with the 212 old, non-target EqGPT formulas. It does not overwrite either benchmark; the eight case datasets remain the evaluation set. Dataset provenance and filtering are documented in [`data/README.md`](../../data/README.md), [`data/formulas/gre_benchmark_filtered/README.md`](../../data/formulas/gre_benchmark_filtered/README.md), and the per-pass audit files under `data/formulas/expanded_downstream_80_v1/`.

## Direct model use

For inspect/generate/score and new-output search commands, see the [study-2 usage guide](../../scripts/eqgpt/README_方案二模型使用.md). This utility loads the 80-position model and the study-2 tokenizer; it does not change the archived experiment runtime. Install dependencies as described in [ENVIRONMENT.md](../../ENVIRONMENT.md).
