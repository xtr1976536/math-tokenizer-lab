# Classified mathematical formula catalog

**56,000 strings** collected from the frozen v6/v7/v8 releases. The catalog consolidates access; it does not change any training or test boundary.

| Axis | Categories / counts |
|---|---|
| Formula partition | Strict PDE equations: 356; all other mathematics: 55,644 |
| Formula domain | 13 categories; detailed counts in manifest.json |
| Provenance | Historical human: 51,229; possible_mixed: 2,810; unknown: 1,961 |
| Syntax complexity | simple: 37,400; complex: 18,600 |
| Fully AI-generated | Zero accepted main-catalog records; historical templates are supplementary material only |

## Files

- `formula_only/pde/<origin>/<complexity>.jsonl`: one JSON string per line, no prose or metadata.
- `formula_only/other/<domain>/<origin>/<complexity>.jsonl`: the same format, by non-PDE domain.
- `ids/`: parallel ID files for tracing each string back to its record.
- `metadata/formulas.jsonl.gz`: source/revision/license/position, raw LaTeX, recorded transformations, hashes, classifications and reference syntax.
- `splits.json`: immutable formula ID lists for original train/dev and test scopes. Extreme-domain IDs are subsets, not additional independent data.
- `manifest.json`: complete file catalog, category counts and content hashes.
- `stress/formulas.jsonl.gz`: 521 exposed synthetic/long-input diagnostic formulas outside the natural catalog.

Only nonempty categories have a formula file. Missing combinations contain zero accepted records; do not invent examples to fill a category. `unknown` remains separate from possible AI involvement. Pre-2020 provenance supports a historical human-origin inference, not per-author attestation.

## How classification works

Formula domain and source-paper subject are separate fields. A paper about PDEs can contain algebraic expressions; these are not all classified as PDE equations. The strict PDE class requires equation/derivative evidence and a compatible source category. Other syntax tags include integral, sum/product, matrix, multirow, differential relation and system.

Complexity uses a fixed weighted score combining lexeme length (.40), command variety (.20), syntax nesting (.15), symbol variety (.15) and operator count (.10). Each normalized component is capped at 2; `complex` means score at least .65. This is a syntax proxy, not mathematical difficulty.

## Sources and quality

51,229 records come from 283 contributing fixed pre-2020 arXiv sources with CC BY/CC0 provenance. 2,810 come from the Apache-2.0 OpenAI math repository and are labeled possible_mixed. FunGrim contributes 1,889 MIT records and SymPy 72 BSD records, both origin unknown. Per-record source links, revisions and notices remain in metadata. EqGPT structure strings and simple AI templates remain in historical supplements, outside the accepted complete-formula benchmark.

Checks are automated: source/version/license hashes; formula-only extraction; bounded macros and independent syntax; macro rendering fidelity; exact/template/rendered-duplicate exclusion for new holdouts; category consistency; stored-string fidelity. The current release rechecks every catalog record. No expert-review queue or human certification is part of the released process.

## Leakage and recommended use

Use `v6_train`/`v6_dev` or the scenario scopes in `experiments/current/`, not all records for training. v6/v7 tests are historical regression sets. v8 was sealed before scoring; it is now exposed. Source categories can be disjoint while mathematical concepts still overlap. Domain imbalance, sparse PDE/linear-algebra coverage and source-syntax labels limit generalization claims.

Original raw dataset snapshots and rejection logs are in `history/data/releases/`. Raw entire paper source archives are cached locally and not redistributed here. Upstream per-record licenses apply to formulas and source content; the repository code license does not override them.
