# Fresh ZIP AST-BPE experiment — four intrinsic metrics only

The final Chinese report is [REPORT_ZH.md](REPORT_ZH.md). Machine-readable results are in `all_results.csv` and `all_results.json`. See `completion.json` for completed-run counts and audit status; an absent completion file means the study is not finalized.

Candidates are the ZIP's **Hard AST-BPE** (`ast_bpe`) and **Soft AST-BPE** (`soft_ast_bpe`), each independently trained from scratch at 320, 384, 448, 512, 640, 768, 896 and 1024 total IDs. Only these two research models are candidates. Old research implementations occur in the byte-identical source snapshot solely as import dependencies; old learned vocabularies are never loaded.

The three fixed pretrained comparisons are Qwen2.5, Llama 3 and o200k. Their original full vocabularies are retained. The main five-system comparison uses 1024-ID new models; the expanded CSV contains the same five-system comparison at all eight new-model budgets.

Metrics: byte-normalized fragmentation (neither too high nor too low is desirable), command integrity, **Cross preservation** (100 minus the previous partial-crossing percentage), and numeric integrity (the latter three higher is better). Fragmentation is a segmentation-density measure, not a semantic-damage oracle. No algebraic inverse relationship between these four measures is assumed.

The catalog is unchanged. PDE-only experiments use the 356 strictly classified PDE strings. Full-benchmark experiments use the 56,000-formula candidate catalog, with source/formula/template decontamination. Protocols include three source splits for PDE, three for the full catalog, three domain holdouts, and three simple-to-complex splits. Detailed split membership is frozen in `protocol.json`. Split seed is not a stochastic training seed.

Reproduction scripts are under `scripts/zip_ast_20261021/` at the repository root. Install the project's package in the environment (`pip install -e .`); the extra pinned packages used here are listed beside the scripts. The C backends require an ordinary C compiler and Python development headers. `vendor/src` and `vendor/native` are exact ZIP copies, hashed in `zip_provenance.json`; no bundled pretrained vocabulary, EqGPT dataset, or downstream model is used. The original ZIP is optional when the checked source snapshot is present.

Audits:

1. Data/source provenance, source/canonical/template isolation, independent interval tests and manually computed metric examples.
2. Native hard-AST encoding vs. its Python reference, native soft-AST objective vs. an independent dynamic program, and deterministic fresh retraining at both endpoint capacities.
3. Independent aggregation of every result and common evaluation IDs/denominators, capacity checks and checkpoint/training-ID hashes.

Per-formula counts are stored in compressed `.jsonl.gz` files, individual source-bootstrap intervals in each result JSON, paired differences in `paired_comparisons_1024.csv`, and PNG/PDF/SVG plots in `figures/`. Overlapping repeated splits are not treated as independent samples, and the pretrained systems' historical training exposure is unknown.

Presentation update: fragmentation has no directional arrow or monotonic ranking. Formula-level fragmentation SD (percentage points) is supplied in `granularity_dispersion.csv` as a supplementary dispersion diagnostic, not evidence of token reusability. Frozen historical protocols and paired delta sign conventions are retained for reproducibility; lower fragmentation is not interpreted as automatically better.
