# Attribution and redistribution scope

The root MIT license covers original project code. It does not relicense upstream tokenizer assets or extracted source formulas.

- The active mathematical formula catalog carries per-record author/title/version/license/source attribution. arXiv records are fixed pre-2020 CC BY 3.0 / CC BY 4.0 / CC0 versions. Full raw manuscript archives remain excluded from Git.
- OpenAI math extractions: Apache-2.0, possible_mixed provenance. FunGrim: MIT, Fredrik Johansson, 1,889 accepted strings. SymPy: BSD-3-Clause, 72 accepted strings. Source formulas are not unconditional mathematical truth labels.
- EqGPT workbook-derived structures and simple assistant-generated templates remain historical supplementary material, outside the accepted current complete-formula catalog. Original historical attribution is preserved in history/NOTICE.md and per-record metadata.
- Hugging Face tokenizers and SentencePiece: Apache-2.0. tiktoken: MIT. Installed official library source is not modified.
- The selected Qwen2.5-0.5B tokenizer configuration is now included under models/baselines/Qwen2.5-0.5B/, with its pinned SHA-256 in the registry. Upstream model assets are Apache-2.0; see the official Qwen/Qwen2.5-0.5B repository.
- Llama 3 reference tokenizer source, license, official-identical configuration and deterministic byte-rank conversion are in models/baselines/Meta-Llama-3-8B/. Meta's Llama 3 license and notices apply. Pinned hashes and provenance are in models/baselines/llama3_provenance.json.
- Original historical metric and baseline code has its license snapshots in references/ and remains traceable through history/. The current mathematical source metrics are project-defined diagnostics, distinct from third-party metrics.
- The presentation uses the user-provided local GBFR Beamer template. Bibliographic references link to original papers; their PDF manuscripts are not republished. All three result figures are generated from this project's audited data. The included PowerPoint is an exact page rendition; editable content is in the Beamer source bundle.

Formula-only files inherit their records' licenses. The unified catalog and directory names do not override upstream terms. Historical logs retain original paths; original reproducibility is available at Git commit 5217a5a.
