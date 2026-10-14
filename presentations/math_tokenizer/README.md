# Academic research presentation

**Design of a Specialized Tokenizer for Mathematical Text**

30 pages in the supplied 4:3 GBFR Warsaw Beamer design. Cover identity, affiliation,
email and October 10, 2026 date are unchanged. The October 14 revision consolidates
Dataset into three pages (sources+pipeline,domain chart without panel (b),classification+
complexity+split counts). The old content frame 18 and 31–38 are removed; one
Limitations and Next Steps frame describes PDE/ODE/reasoning tasks,from-scratch
training,benchmark selection and compute needs. Other retained content is
unchanged apart from replacing reference footers with numeric citation links.

Only 13 used references appear in two reference pages. Numeric citations jump internally
in both PDF and PowerPoint. Source/code/paper links are external hyperlinks.

## Deliverables

- output/Design_of_a_Specialized_Tokenizer_for_Mathematical_Text.pptx and.pdf:
  canonical latest versions. Dated/dataset-revision copies identify revisions.
- output/Mathematical_Tokenizer_Beamer_Source.zip: editable source/template/assets.
- PowerPoint retains exact Beamer page images with transparent native clickable
  regions. Edit content in the Beamer master; diagram PDF embeds editable drawioXML.

## Rebuild

Use scripts/build_presentation.py,then compile source/mathematical_tokenizer.tex
with XeLaTeX. Export with bundledPython scripts/export_presentation.py and Artifact
Tool; scripts/add_presentation_links.py preserves PDF links before finalization.
Run scripts/check_presentation.py. Source/reference/scope and hyperlink reviews are
in qa/. The original three data figures are retained; Dataset uses only panel (a), and
adds one simple construction flowchart. No downstream accuracy is claimed completed.
