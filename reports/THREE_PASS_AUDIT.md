# Three-pass release audit (2026-10-23)

## Pass 1: hard-only archive

- Local deliverable: `math_tokenizer_project/deliverables/hard_ast_bpe_8case_20261022.zip`.
- ZIP size: 3,492,917,864 bytes; SHA-256: `d4b3ec52fc987b7b4c28a378c64ccb77c1e476787c2a5a4a51661c7728be6ac5`.
- ZIP structural check passed with `unzip -tqq`. All 1,248 archived files matched the SHA-256 inventory embedded in `ARCHIVE_MANIFEST.json`.
- Eight cases and hard vocabularies 280/315/350 are present. No soft-arm result directory, GRE-arm result directory, provisional discovery benchmark, or abandoned compact/early-stopping pilot appears in the ZIP.
- Full hard checkpoints are in the local ZIP. The 3.5 GB ZIP is not in GitHub because it exceeds normal Git hosting limits.

## Pass 2: repository scope

- The private GitHub `main` tree contains 600 files. `experiments/pde1500_main_final_20261022/results.json` contains exactly hard AST-BPE, Qwen2.5, Llama 3, and `o200k_base`; the 200-450-ID sweep contains only hard AST-BPE.
- The downstream tree contains eight cases, three hard sizes per case, and 24 complete search traces. `reports/downstream/hard_results.csv` has the same 24 arms. The target-exclusion audit has eight cases and 24 hard training streams.
- No `soft_ast_bpe`, `eqgpt_reference`, `discovery_benchmark`, or `compact_es` path is in the Git tree. Shared EqGPT author source is retained as a numerical dependency, not as a GRE result.
- The original working directory and its uncommitted files were left intact. Cleanup was committed from a separate clean worktree and pushed to the private repository without force-push.

## Pass 3: data and runtime

- The frozen upstream dataset loads as 1,500 PDEs from 490 source groups, with 450 simple, 600 moderate, and 450 complex syntax bins.
- The downstream 350-ID hard tokenizer round-trips a held-out training text; the Wave case source contains 6,784 augmented sequences.
- The curated EqGPT adapter and hard tokenizer modules import using the project environment. All committed files are below 90 MB.
- `git -c core.whitespace=cr-at-eol diff --cached --check` passed before commit. The CRLF in archived CSV results is preserved as source data.
- Remote private `main` was verified at commit `d4caf8ea661dff1c51141ae34df1e378099de9de` after push.

The retained benchmark is for upstream formula tokenization and the original eight known-equation downstream tests. A benchmark for previously unknown effective equations is **not** part of this release.
