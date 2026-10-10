# License and attribution scope

The root MIT license applies to original project code, not blanket relicensing of upstream material.

- EqGPT workbook-derived coefficient-free structures: upstream woshixuhao/EqGPT repository declares Apache-2.0. Original mathematical handbook attribution remains in each record and the cited paper; automatic conversion is explicitly documented. Repository declaration is recorded, not asserted as independent verification of all handbook rights.
- openai/math source formula extractions: upstream Apache-2.0; model-produced manuscripts with some human editing. Correctness of claims is not established by this corpus.
- SymPy 1.10 documentation extractions: BSD-3-Clause, SymPy Development Team attribution retained in references/sympy_LICENSE.
- SamD770/bitter-lesson-tokenization: MIT; upstream source not modified, not redistributed in the main tree. Exact reference commit in references/upstream_code.json.
- Explicit assistant-authored formula templates and mechanical parameter expansions: CC0-1.0. They are not independent model samples.
- HuggingFace tokenizers (Apache-2.0), SentencePiece (Apache-2.0), tiktoken (MIT): installed official distributions, source code unchanged.
- Qwen/Qwen2.5-0.5B tokenizer asset: official model repository license Apache-2.0; retrieved by immutable revision. Asset not included in this tree; SHA-256 and download adapter provided.

The quarantine directory is rejected source material retained for audit, not a training corpus. License snapshots for primary source repositories are in references/. References to user-supplied papers are research evidence; the supplied PDF files are not republished in this repository.

Second release additions:

- Pinned arXiv papers: only article-version pages and OAI metadata jointly verified as CC BY 4.0 / CC0 (other permissive variants recorded if present). Formula attribution includes authors, title, exact version, licence URL, source URL and source location. No nonexclusive-distribution manuscript is added to the published formula corpus. Raw source archives are cached locally and excluded from Git. CC BY attribution is inherited per record.
- Fungrim symbolic formula database and official LaTeX renderer: MIT, Fredrik Johansson; unmodified official renderer at the revision in data/fungrim_source.json. Preserve references/fungrim_LICENSE. Reuse is not CC0; formula conditions can remain in source context, so extracted formula strings must not be used as unconditional mathematical truth labels.
- TokEval intrinsic metrics: MIT, Clara Meister; unmodified official package at the revision in references/upstream_code.json and requirements-v2.lock.txt. Preserve references/tokeval_LICENSE. All formula-specific metrics written here are distinguished from upstream metrics.
