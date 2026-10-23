# Math tokenizer: current PDE benchmark and hard AST-BPE downstream results

This curated branch contains the current four-tokenizer upstream comparison, the 1,500-PDE formula benchmark, and the original eight-case hard AST-BPE downstream experiment from 2026-10-22. It excludes the later compact-alphabet/early-stopping pilot and the provisional real-data discovery benchmark.

## Upstream

- Four methods: hard AST-BPE, Qwen2.5, Llama 3, and `o200k_base`.
- Fixed 1,500-PDE benchmark: `data/benchmark_pde1500_20261022/` (490 source papers; 450/600/450 syntax-complexity mix). The accompanying non-PDE records are preserved in `formulas.jsonl.gz`.
- Latest three-fold and simple-to-complex results: `experiments/pde1500_main_final_20261022/` and `reports/upstream/main_summary.csv`.
- Hard-only 200-450 ID sweep and actual token-use distribution: `experiments/pde1500_sweep_200_450_20261022/` and `reports/upstream/`.

These are tokenizer metrics, not PDE-discovery outcomes. The 1,500 formulas were subjected to reproducible automated checks and semantic spot checks; they were not each expert-certified.

## Downstream

- Original eight EqGPT benchmark cases, hard AST-BPE at 280, 315, 350 IDs: `experiments/downstream/` and `reports/downstream/hard_results.csv`.
- Each case retains its target-excluded training source, fixed protocol, frozen hard tokenizer, training record, complete candidate search trace, filtered hard-only log, and shared fitted MLP surrogate cache.
- The original least-squares reward and Top-10 continuation search are preserved. Final PINN coefficient fitting is omitted; structure classification is the outcome.
- The full hard-only archive with all eight cases' pretrained and searched model weights is distributed separately as `hard_ast_bpe_8case_20261022.zip`; it is intentionally not committed because its size is about 3.5 GB. The archive manifest includes per-file SHA-256 hashes.

The EqGPT source and the eight original numerical-case datasets are under `third_party/eqgpt/`. The benchmark's 1,500 PDEs are formula text for upstream segmentation, not previously unknown physical laws. No new effective-equation discovery is claimed here.
