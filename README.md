# Specialized Tokenizers for Mathematical Text

This repository contains **two active mathematical tokenizer routes** and **five baseline tokenizers**, a classified formula-only benchmark, audited intrinsic experiments, and an academic Beamer presentation. Downstream language-model training has not started.

| Current system | Role | Vocabulary |
|---|---|---|
| Boundary AST | Source structure and parameter-boundary segmentation | 1,024 / 2,048 IDs |
| Math-atom BPE | Fit-derived mathematical alphabet + original HF BPE | 1,024 / 2,048 IDs |
| Byte BPE | Original HF byte BPE baseline | 1,024 / 2,048 IDs |
| SentencePiece Unigram | Original probabilistic subword baseline | 1,024 / 2,048 IDs |
| Qwen2.5 | Fixed general tokenizer | 151,665 IDs |
| Llama 3 | Original Meta tokenizer interface | 128,256 IDs |
| o200k | Original tiktoken encoding | 200,019 IDs |

The math-atom route is a domain-specific training recipe using established BPE; the AST route changes segmentation over exactly the same dictionary. Neither production route uses RL. Full general vocabularies are deployment references, not equal-parameter causal comparisons.

## Repository map

- [`data/benchmark/`](data/benchmark/README.md): **56,000 formula strings**, 13 domains, source-provenance and two syntax-complexity classes. Formula text, IDs and metadata are separate. Frozen source partitions are preserved.
- [`models/registry.json`](models/registry.json): the only active model registry; 7 methods / 19 configurations. `models/proposed/` contains the two routes; `models/baselines/` contain their pinned controls.
- [`experiments/current/`](experiments/current/README.md): all retained metrics, 234,556 formula-result rows and three new audit rounds. Numerical results are an unchanged subset of the original v8 experiment.
- [`presentations/math_tokenizer/`](presentations/math_tokenizer/README.md): English academic presentation using the supplied GBFR Beamer design, exact cover identity and date, 26 primary references and 3 data figures.
- [`docs/`](docs/reproduction.md): current reproduction and downstream protocol notes.
- [`history/`](history/README.md): superseded models, datasets, experiment records, scripts and tests. Embedded historical paths refer to Git commit `5217a5a`; check out that commit for original replay.

## Loading and training

```python
from math_tokenizer.final_models import load_final
model = load_final('ast_boundary', scenario='broad', capacity=1024)
formula = r'\frac{\partial u}{\partial t}=\Delta u'
assert model.decode(model.encode(formula)) == formula
```

The other proposed method is `official_atomic_bpe`. Exact baseline method names are listed in the registry. Historical model names are rejected by the active loader. Trained methods have both `broad` and strict source-category `specialized` versions; ordinary downstream work should begin with `broad`.

```sh
.venv/bin/python scripts/audit_current.py
.venv/bin/python scripts/summarize_current.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/train_current.py --scenario broad --capacity 1024 --output /tmp/math-tokenizer-retrain
```

Training reads only the preserved fit and allowed auxiliary IDs. **Do not train on the entire 56,000-string catalog:** it includes test records. Test membership and metadata have not been changed by repository organization.

## Main intrinsic result

On the final v8 test, Boundary AST reduces paper-macro token/byte by **1.09%–2.64%** and encodes **2.65–3.00 times faster** than the same-dictionary math-atom BPE across the four budget/scenario conditions. Scope recall rises in three conditions and falls by 0.28 percentage points in one. Family-adjusted paper-cluster intervals support compression improvement in all four conditions, but do not establish universal structural gains.

The three new release audits passed over **56,000 catalog formulas, 19 configurations, 109 metric records and 234,556 result rows**. Earlier v8 algorithm selection preceded new-test scoring; the current seven-method selection is retrospective. Historical RL did not pass the prespecified same-reward search-control promotion gate and is archived.

Quality assurance consists of automated source/license, syntax, macro-fidelity, duplicate, classification and serialization checks. There is no pending expert-annotation workflow or certification claim. The benchmark is source syntax, not mathematical theorem truth. Domains are uneven; only **356** catalog formulas meet the strict PDE-equation rule. All current tests are now exposed. Speed evidence is from one machine and no downstream-task benefit has yet been measured.
