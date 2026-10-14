# Academic research presentation

**Design of a Specialized Tokenizer for Mathematical Text**

42 slides in the supplied 4:3 GBFR Warsaw Beamer design. Presenter Tongren Xiao, email, affiliation and the original October 10, 2026 cover date are retained. Research, repository selection and literature verification were updated on October 13, 2026.

The two added future-research slides explain autonomous boundary learning, fast
signal guidance and verified mathematical transitions. Stage 2 is explicitly
paused; no joint success or mathematical understanding is claimed. Current
priority is downstream evaluation of the two Stage 1 routes and five baselines.

## Deliverables

- `output/Design_of_a_Specialized_Tokenizer_for_Mathematical_Text.pdf`: vector Beamer presentation.
- `output/Design_of_a_Specialized_Tokenizer_for_Mathematical_Text.pptx`: exact high-resolution slide-page rendition for PowerPoint; source text is in speaker notes. Slide elements are images; edit the Beamer master for content/layout changes.
- `output/Mathematical_Tokenizer_Beamer_Source.zip`: complete editable .tex master, supplied template, vector/raster figures, source-reference records and build instructions.

Updated deliverables have `_Updated.pdf` and `_Updated.pptx` suffixes; the canonical
PDF/PPTX paths also point to this revision. PowerPoint remains an exact page image
rendition with speaker notes, now exported through Artifact Tool. The editable
master is the Beamer source bundle.

The talk covers the research question, classical/general/math-specific/adaptive related work, source dataset construction and taxonomy, both selected methods, controlled settings and metrics, seven-method numerical results and uncertainty, historical RL removal, and four planned downstream tasks plus compute/adaptation constraints. It uses exactly **3 original data figures** and **26 primary references**. Related work is a targeted narrative search, not a claimed exhaustive systematic review.

## Rebuild and verification

Run `scripts/plot_presentation.py`, then `scripts/build_presentation.py`. Compile `source/mathematical_tokenizer.tex` with XeLaTeX/latexmk. Use the bundled document Python for `scripts/export_presentation.py`, which needs python-pptx and pypdf. Run `scripts/check_presentation.py` for evidence, layout and export-identity checks.

`qa/three_round_review.json` records three deliverable checks. Every slide was rendered and inspected; overflowing boxes and plot-label overlaps were corrected. PDF table cells are generated from experiments/current/rankings.json, and the original 144-family statistical intervals are retained. The PowerPoint export is byte-verified against every rendered source page. The editable source bundle contains all dependencies other than the TeX installation.

No downstream accuracy is presented as completed. The seven-method subset is retrospective; original v8 algorithm settings were frozen before its new-paper scores. Source syntax metrics do not establish mathematical semantic correctness.

The October 14 edit changes only the opening content: concise research question and overview, three baseline pages (source links, SP Unigram workflow, two-round EM example). Original content frame9onward remains unchanged; frame numbers update automatically.
