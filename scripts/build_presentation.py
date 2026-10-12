"""Beamer source with evidence tables generated directly from reviewed JSON."""
import hashlib,json,re
from pathlib import Path
from math_tokenizer.corpus import ROOT

OUT=ROOT/'presentations/math_tokenizer/source'
METHODS=['ast_boundary','official_atomic_bpe','hf_byte_bpe','sentencepiece_unigram','Qwen2.5-0.5B','meta-llama/Meta-Llama-3-8B','tiktoken_o200k_base']
LABELS=['Boundary AST (ours)','Math-atom BPE (ours)','Byte BPE','SP Unigram','Qwen2.5','Llama 3','o200k']


def escape(text):
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:text=text.replace(a,b)
    return text


def table(headers,rows,widths=None):
    spec=widths or ('l'+'r'*(len(headers)-1))
    spec=spec.replace('X',r'>{\raggedright\arraybackslash}X')
    def cell(value):return re.sub(r'(?<!\\)%',r'\\%',str(value))
    kind='tabularx' if 'X' in spec else 'tabular*'
    spacing='@{}' if kind=='tabularx' else r'@{\extracolsep{\fill}}'
    return r'\par\noindent{\scriptsize\tbl\begin{'+kind+r'}{\linewidth}{'+spacing+spec+r'@{}}\toprule'+'\n'+' & '.join(map(cell,headers))+r'\\\midrule'+'\n'+''.join(' & '.join(map(cell,row))+r'\\'+'\n' for row in rows)+r'\bottomrule\end{'+kind+r'}}\par\vspace{2pt}'+'\n'


def block(title,body,kind='definitionbox'):return '\\begin{'+kind+'}{'+title+'}\n'+body+'\n\\end{'+kind+'}\n'
def bullets(items):return '\\begin{itemize}\n'+''.join('\\item '+v+'\n' for v in items)+'\\end{itemize}\n'


def main():
    assert json.loads((ROOT/'experiments/current/three_round_audit.json').read_text())['all_passed']
    ranks=json.loads((ROOT/'experiments/current/rankings.json').read_text());boards=ranks['boards']
    manifest=json.loads((ROOT/'data/benchmark/manifest.json').read_text())
    stats=json.loads((ROOT/'experiments/current/paired_effects.json').read_text())
    refs=json.loads((OUT/'references.json').read_text());refmap={r['id']:r for r in refs}
    frames=[]
    def frame(section,title,body,cites=(),note=''):
        if isinstance(cites,str):cites=(cites,)
        footer=''
        if cites:footer='\\src{'+ '; '.join(r'\href{'+refmap[k]['url']+'}{'+escape(refmap[k]['authors'])+' ('+str(refmap[k]['year'])+')} ' for k in cites)+'}\n'
        frames.append({'section':section,'title':title,'body':body+footer,'note':note})
    frame('Research Question','Can Mathematical Tokenization Improve Model Capability?',
        block('Research question',r'Can we train tokenizers specifically for mathematical text that improve both \textbf{intrinsic tokenization} and \textbf{downstream mathematical capability}?')+
        table(['Question','What would count as evidence?'],[
            ['RQ1: tokenizer quality','Lossless compression, structural boundary preservation, robustness and full-interface speed.'],
            ['RQ2: model capability','Better symbolic manipulation, PDE solving/discovery and mathematical reasoning under controlled training budgets.'],
            ['Current status','RQ1 has completed experiments; RQ2 is a planned, separate evaluation.']],r'L{2.7cm}X')+
        block('Two final research routes',r'\textbf{Math-atom BPE}: learn a mathematical vocabulary using a reversible lexical alphabet.\\\textbf{Boundary AST}: use the same vocabulary with source-structure-aware segmentation.','remarkbox'),('pathpiece','tokeval'))
    frame('Research Question','Why Mathematical Source Text Is Different',
        block('A source string carries both symbols and scope',r'\[\frac{\partial u}{\partial t}=\nu\Delta u+f(x,t).\]'+'Commands, indices and grouped arguments carry distinct roles. Command splits can hide recurring units; cross-argument merges can hide useful boundaries.')+
        table(['Property','Example','Design consequence'],[
            ['Command identity',r'\texttt{\textbackslash partial}, \texttt{\textbackslash frac}','Protect observed complete commands.'],
            ['Argument scope',r'\texttt{\{x+y\}} vs. \texttt{\{x\}+y}','Track source endpoints without rewriting the formula.'],
            ['Numerical form',r'$1.25$, $10^{-6}$, $x_{12}$','Measure digit and number integrity separately.'],
            ['Open vocabulary','Unseen commands and Unicode','Retain a complete 256-byte fallback.']],r'L{2.1cm}L{3.7cm}X')+
        block('Important distinction','Lossless source reconstruction is necessary. It does not prove that the tokens preserve all mathematical semantics.','remarkbox'),('texbleu','xval'))
    frame('Research Question','What This Study Establishes',
        table(['Component','Completed evidence','What remains untested'],[
            ['Data','56,000 classified formula strings; frozen source-based partitions.','General mathematical truth or complete semantic equivalence.'],
            ['Methods','Two frozen routes; original third-party training and baseline interfaces.','Benefits from model-specific embedding adaptation.'],
            ['Evaluation','Seven retained systems; three new audits of the current release.','Downstream accuracy and scaling to multiple hardware platforms.'],
            ['Selection','Two proposed systems and five baselines explicitly registered.','Whether the best intrinsic trade-off is the best task representation.']],r'L{1.7cm}X X')+
        block('Evidence boundary','The original v8 model/weight freeze preceded its test scores. The present seven-system selection is retrospective and user specified; it is not a new unseen-test experiment.','remarkbox')+
        block('Experimental separation','Tokenizer learning, string-to-ID inference and language-model adaptation are separate interventions. A gain in one stage must be checked rather than assumed to transfer to the next.'),('pathpiece','compute'))
    frame('Related Work','Classical Subword Tokenization',
        table(['Family','Learning and inference','Relevance to mathematics'],[
            ['BPE','Repeated pair-frequency merges; apply a learned merge order.','Simple, fast and data driven; syntax is not the training objective.'],
            ['WordPiece','Vocabulary learning followed by subword matching.','Widely used; unknown handling and pretokenization affect raw fidelity.'],
            ['Unigram','Learn piece probabilities; choose a likely segmentation.','Separates vocabulary probability from one fixed merge history.'],
            ['SentencePiece','A training/runtime framework supporting BPE and Unigram.','Identity normalization and byte fallback must be set explicitly.']],r'L{2.0cm}X X')+
        block('Our use of established algorithms','The math-atom route uses the original Hugging Face BPE trainer and Rust runtime. Its contribution is the mathematical alphabet, corpus and controlled design, not a new BPE optimization algorithm.','remarkbox'),('bpe','unigram','sentencepiece'))
    frame('Related Work','General-Purpose Tokenizers as Deployment References',
        table(['Retained tokenizer','Actual ID space','Interface used'],[
            ['Qwen2.5','151,665','Pinned official tokenizer configuration; HF tokenizers.'],
            ['Llama 3','128,256','Pinned Meta tokenizer code and byte-rank file.'],
            ['o200k','200,019','Original tiktoken o200k\_base.'],
            ['Our trained comparisons','1,024 / 2,048','Every primitive and reserved ID counted.']],r'L{2.5cm}L{2.3cm}X')+
        block('What is held fixed','All systems receive the same raw formulas and metric definitions. Mature tokenizers retain their full vocabularies; their source pretraining data and possible contamination are unknown.')+
        block('What this comparison cannot establish','Different vocabulary sizes imply different embedding/softmax costs. These rows are practical deployment references, not equal-parameter causal comparisons.','remarkbox'),('qwen','llama','tiktoken'))
    frame('Related Work','Specialized Mathematical and Scientific Representations',
        table(['Work','Relevant mechanism','Difference from our study'],[
            ['TeXBLEU (2024)','LaTeX-trained BPE and embeddings for a formula-generation metric.','Evaluates generated expressions, rather than a common standalone tokenizer benchmark.'],
            ['CMERNet (AAAI 2026)','Explicit LaTeX command/environment units plus specialized BPE.','Targets expression recognition; its symbol inventory is curated.'],
            ['MathBERT (2021)','Operator-tree substructure prediction during formula/context pretraining.','Changes model pretraining objectives, rather than only segmentation.'],
            ['MathGPT (ACL 2023)','Operator-tree tokens, tree positions and constrained generation.','Uses a tree representation and model modifications.']],r'L{2.5cm}X X')+
        block('Positioning','Preserving mathematical units and using formula trees are established ideas. Our claim is a specific raw-byte segmentation trade-off with matched-vocabulary controls.','remarkbox'),('texbleu','cmer','mathbert','mathgpt'))
    frame('Related Work','Compression, Structure and Intrinsic Evaluation',
        table(['Work','Finding / design principle','Consequence for our experiment'],[
            ['PathPiece (EMNLP 2024)','Shortest segmentation need not yield the best downstream model.','Separate vocabulary learning from segmentation; keep same-dictionary controls.'],
            ['Greed Is All You Need (ACL 2024)','Inference choices affect intrinsic properties even with fixed vocabularies.','Report lexical boundaries and information measures, not just token count.'],
            ['TokEval (COLM 2026)','Structural properties such as Unicode and digit boundaries matter.','Measure integrity and robustness alongside compression.'],
            ['Compute Optimal Tokenization (2026)','Granularity interacts with model size, data and compute.','Downstream budgets must include vocabulary and sequence costs.']],r'L{2.6cm}X X')+
        block('Our structural metrics','KaTeX partial-crossing and pylatexenc argument endpoints are syntax proxies. Neither is an established gold measure of mathematical meaning.','remarkbox'),('pathpiece','greed','tokeval','compute'))
    frame('Related Work','Adaptive Tokenization and the Role of RL',
        table(['Approach','What is learned','Why it is not the same experiment'],[
            ['Dauncey \& Wattenhofer (ICML 2026)','Discrete byte boundaries using score-function estimates and LM loss.','Requires downstream model feedback; our terminal vocabulary reward is an adaptation.'],
            ['H-Net (2025)','Content-dependent hierarchical chunks jointly with a sequence model.','Changes the model architecture and latent representation.'],
            ['AdaptBPE (2026)','Adapt general tokenization to specialized domains.','Motivates domain adaptation; does not establish formula-tree optimality.'],
            ['xVal (2023)','Continuous numerical encoding inside scientific LMs.','Requires a numerical channel and modified model handling.']],r'L{2.9cm}X X')+
        block('Design decision','Standalone multi-objective RL did not pass our repeatability gate. The final production routes contain no policy updates or online policy inference.','remarkbox'),('rl','hnet','adaptbpe','xval'))
    frame('Related Work','Our Two Routes: Contribution and Prior Art',
        table(['Dimension','Math-atom BPE','Boundary AST'],[
            ['Vocabulary','Fit-derived complete-command/Unicode atoms; original BPE.','Exactly the same byte vocabulary and ID budget.'],
            ['Segmentation','Original learned BPE merge order.','Minimum additive token/tree/boundary cost on a legal source graph.'],
            ['Mathematical prior','Lexical units observed in training.','Lexical units plus a restricted source parser.'],
            ['Deployment','Native reversible codec + original Rust BPE.','Fused native trie traversal and dynamic programming.'],
            ['Novelty boundary','A reproducible domain-specific training recipe.','A tested structural trade-off; shortest paths and AST ideas are not new.']],r'L{2.2cm}X X')+
        block('Central comparison','Holding the dictionary fixed isolates the effect of the structural segmentation rule. Comparing against generic tokenizers instead measures the complete specialized system.','remarkbox'),('cmer','pathpiece','mathgpt'))
    frame('Dataset','Corpus Sources and Their Roles',
        table(['Source','Accepted formulas','Provenance / role'],[
            ['arXiv, fixed pre-2020 versions','51,229','CC BY / CC0; 283 contributing source papers; historical human-origin inference.'],
            ['OpenAI math repository','2,810','Apache-2.0; conservatively labeled possible\_mixed.'],
            ['FunGrim','1,889','MIT; formula library, origin unknown.'],
            ['SymPy','72','BSD-3-Clause; formula fixtures, origin unknown.'],
            ['EqGPT structures / AI templates','Supplement only','Not counted as complete research-level LaTeX formulas in the 56,000-string catalog.']],r'L{3.1cm}L{1.8cm}X')+
        block('Source-specific distinction','EqGPT reports 221 handbook PDE structures and an operator vocabulary. Its structure strings are useful context, but are not automatically valid complete LaTeX equations.','remarkbox')+
        block('Record-level traceability','Each accepted string retains its source revision, license and location. Formula text is stored separately from this metadata; redistribution follows the upstream record license.'),('eqgpt','datasheets'),note='Counts are computed from the active catalog. Do not confuse EqGPT 221 structures with the 356 strict PDE equations in our catalog.')
    frame('Dataset','Construction: From Licensed Source to Formula Record',
        table(['Step','Operation','Retained evidence'],[
            ['1. Acquire','Official arXiv URLs; cache first, one connection, at least 4 s between requests.','Version date, license URL, source archive hash.'],
            ['2. Extract','Find inline/display/multirow math; bounded source-macro expansion.','Original LaTeX, member path and source span.'],
            ['3. Filter','Reject prose, isolated labels, executable commands and damaged/trivial fragments.','Explicit rejection reason and quarantine record.'],
            ['4. Validate','Independent KaTeX parse and source/expanded rendering fidelity.','Reference spans and rendered-equivalence hashes.'],
            ['5. Partition','Exact/template/render-equivalent exclusion against earlier exposures.','Frozen formula IDs, source groups and split hashes.']],r'L{1.7cm}X X')+
        block('No hidden mathematical rewriting','The encoder is evaluated on the accepted stored string. Formula normalization is a recorded dataset operation, not a model-specific test preprocessing step.','remarkbox'),('datasheets','cmer'))
    domainrows=[]
    domains=manifest['domain']
    domain_order=['analysis','geometry_topology','combinatorics','algebra','number_theory','mathematical_physics','optimization','probability','statistics','logic','pde','ode','linear_algebra']
    for i in range(0,len(domain_order),2):
        a=domain_order[i];b=domain_order[i+1] if i+1<len(domain_order) else None
        domainrows.append([escape(a.replace('_',' / ')),f'{domains[a]:,}',escape(b.replace('_',' / ')) if b else '',f'{domains[b]:,}' if b else ''])
    frame('Dataset','Taxonomy: Formula Type Is Not Paper Topic',
        table(['Formula domain','Count','Formula domain','Count'],domainrows,r'X r X r')+
        block('Separate axes',r'\textbf{Formula domain}: strict PDE vs. other mathematics.\\\textbf{Paper domain}: source subject, stored independently.\\\textbf{Types}: integral, sum/product, matrix, differential relation and system.')+
        block('PDE limitation','Only 356 strings meet the strict PDE rule. Other formulas from PDE papers retain their structural category; the catalog is not yet a large PDE-solver training set.','remarkbox'))
    frame('Dataset','Provenance Labels and Two Complexity Classes',
        table(['Origin label','Count','Interpretation'],[
            ['Historical human','51,229','Fixed pre-2020 manuscripts; inference from source history.'],
            ['Possible mixed','2,810','Model-produced / edited manuscript sources; no per-formula attribution.'],
            ['Unknown','1,961','Library provenance does not establish who typed each formula.'],
            ['Fully AI generated','0 in main catalog','Synthetic templates remain separate supplements.']],r'L{3.0cm}r X')+
        block('Fixed syntax-complexity score',r'\[s=.40\bar L+.20\bar C+.15\bar D+.15\bar S+.10\bar O,\quad \bar z=\min(z/z_0,2).\]'+
            r'$L$: lexeme length; $C$: command types; $D$: nesting; $S$: symbol types; $O$: operators. Normalizers: $(60,6,3,18,12)$. Complex iff $s\ge .65$.')+
        block('Class balance','37,400 simple and 18,600 complex formulas. These are reproducible syntax strata, not theorem difficulty levels.','remarkbox'))
    frame('Dataset','Frozen Partitions and Generalization Protocol',
        table(['Partition / scenario','Formulas','Use'],[
            ['v6 train / dev','36,615 / 7,693','Source-separated pools; active scenario subsets below.'],
            ['v6 new-paper test','3,465','Previously exposed; regression check only.'],
            ['v7 new-paper / extreme','3,933 / 1,961','Previously exposed; regression and domain checks.'],
            ['v8 new-paper / extreme','4,294 / 1,428','Final sealed source test, scored after model freeze.'],
            ['Synthetic / long-input stress','521','Explicit stress diagnostics, outside the natural catalog.']],r'L{4.0cm}L{2.0cm}X')+
        bullets([r'\textbf{Broad}: fit 30,100; auxiliary training/reward pool 512; dev 965.',
                 r'\textbf{Specialized}: fit 8,124; auxiliary pool 512; dev 249. Target and adjacent source categories excluded.',
                 'Extreme subsets overlap their parent test. They are not additional independent samples.'])+
        block('Do not train on the whole catalog','56,000 is an organized library size. Its ID-only split manifests preserve the training/test boundary.','remarkbox'))
    frame('Dataset','Corpus Composition and Coverage',
        r'\begin{center}\includegraphics[width=.98\linewidth,height=5.5cm,keepaspectratio]{assets/dataset_composition.pdf}\end{center}'+
        block('What the composition implies','Analysis and geometry dominate. PDE, ODE and linear algebra are sparse. Report paper/domain/complexity strata and avoid interpreting an overall average as uniform mathematical coverage.','remarkbox'))
    frame('Math-Atom BPE','Route 1: Learning a Mathematical Lexical Alphabet',
        block('Training alphabet',r'\[A=\{\text{all 256 bytes}\}\cup\operatorname{Top}_{384}\{\text{observed complete commands / Unicode units}\}.\]'+
              'Atoms are derived from fit records only, ranked by frequency with deterministic tie rules. The actual primitive counts are 596 (broad) and 495 (specialized).')+
        table(['Stage','Implementation','Budget accounting'],[
            ['Reversible codec','Map observed lexical units to private-use symbols.','One primitive ID per atom; unknown strings retain byte coverage.'],
            ['Vocabulary training','Original HF BPE on fit + the allowed 512-record auxiliary pool.','Total IDs capped at 1,024 or 2,048.'],
            ['Inference','Native lexical codec, then original Rust BPE.','No AST path search and no policy network.']],r'L{2.3cm}X X')+
        block('Why this route remains','It is the simple, fast mathematical reference against which extra structural machinery must justify itself. No manually enumerated command vocabulary is required.','remarkbox'),('bpe','cmer'))
    frame('Math-Atom BPE','Route 1: Exact Source Fidelity and Its Limits',
        block('Raw-string contract',r'For every evaluated $x$, $\operatorname{decode}(\operatorname{encode}(x))=x$. The ordered token byte payloads also concatenate to $\operatorname{UTF8}(x)$ exactly. No whitespace normalization is permitted.')+
        table(['Case','Behavior','What is not guaranteed'],[
            ['Seen command / Unicode atom','Codec retains the complete observed unit.','A larger token can still span an argument boundary.'],
            ['Unseen command','Explicit byte coverage; no unknown-token deletion.','Every novel command need not be a single token.'],
            ['Spacing / multiline source','Bytes are preserved exactly.','Layout variants need not receive identical token sequences.'],
            ['Numbers / identifiers','BPE learns recurrent sequences.','Numeric magnitude is not encoded continuously.']],r'L{2.4cm}X X')+
        block('Interpretation','High command integrity reflects observed alphabet coverage, not universal LaTeX grammar coverage or a learned theorem representation.','remarkbox'),('xval','texbleu'))
    frame('Boundary AST','Route 2: Source Structure as a Segmentation Cost',
        block('Same vocabulary, different path',r'An edge $(a,b)$ matches $x_{a:b}\in V$ at permitted lexical endpoints. The restricted parser supplies source spans and matched brace-content endpoints.')+
        block('Additive objective',r'\[z^*=\arg\min_{z\in\mathcal P(x,V)}\left[|z|+\lambda\sum_{e\in z}c_T(e)+\mu\sum_{e\in z}c_B(e)\right].\]'+
              r'$c_T$: token partially crosses a native source span.\\$c_B$: number of brace-content endpoints strictly inside the token.\\Development selection: $\lambda=.25$, $\mu=.30$.')+
        block('A source-tree method','The parser is restricted and falls back to source syntax. It is not a full TeX interpreter or a semantic operator tree for arbitrary mathematics.','remarkbox'),('pathpiece','mathgpt'))
    frame('Boundary AST','Route 2: Exact Dynamic Programming on a Fixed Graph',
        block('Backward recurrence',r'\[D[n]=0,\qquad D[a]=\min_{(a,b)\in E}\{1+\lambda c_T(a,b)+\mu c_B(a,b)+D[b]\}.\]'+
              'A byte trie enumerates matching dictionary edges. A stored best endpoint reconstructs the token sequence. This is exact for the specified legal graph and additive objective.')+
        table(['Design','Mechanism','Reason'],[
            ['Lexical mask','Suppress cuts inside covered commands / codepoints.','Keep complete observed primitive units.'],
            ['Tree cost','Endpoint sweeps detect partial span crossings.','Avoid testing every token against every span.'],
            ['Boundary cost','Prefix sums count hidden brace-content endpoints.','Recover argument boundaries without hard blocking all compression.']],r'L{2.0cm}X X')+
        block('Scope of optimality','The result is not globally optimal vocabulary learning, and a low source-tree cost does not imply low task loss. The shortest-path algorithm itself is established.','remarkbox'),('pathpiece','greed'))
    frame('Boundary AST','Route 2: Development Selection and Runtime Cost',
        table(['Setting','Values / behavior'],[
            ['Prespecified grid',r'$\lambda\in\{0,.25,.75\}$; $\mu\in\{0,.05,.15,.30\}$: 12 candidates.'],
            ['Selection criterion','Common old-dev utility plus .04 times missing parameter-boundary recall.'],
            ['Feasibility guard','Compression regression at most 1%; boundary recall loss at most 1 percentage point vs. math-atom BPE.'],
            ['Selected setting','All four budget/scenario conditions selected .25 / .30 before v8 scoring.'],
            ['Public encoder','C frontend + trie + DP; all source parsing is included in throughput.']],r'L{3.0cm}X')+
        block('Engineering and trade-offs',r'Typical lattice work scales with input bytes, maximum piece length and source spans; the dense byte trie uses more memory than a compact sparse trie.\\Compared with the older length-first path, the boundary term costs some speed but restores much of the parameter-boundary recall.','remarkbox')+
        block('Deployment contract','A frozen dictionary and weights define deterministic IDs. Unknown lexical units use byte coverage; supported source parsing has an explicit one-million-byte input limit.'))
    frame('Experimental Design','Exactly Two Routes and Five Baselines',
        table(['System','Vocabulary / training','Role'],[
            ['Boundary AST (ours)','Common math-atom BPE vocabulary; fixed source-path objective.','Structure-aware route.'],
            ['Math-atom BPE (ours)','Fit-derived alphabet; original BPE on allowed data.','Lexical domain route.'],
            ['Byte BPE','Original HF byte BPE; same allowed training records.','Equal-capacity classical control.'],
            ['SP Unigram','Original SentencePiece; identity normalization, byte fallback.','Equal-capacity probabilistic control.'],
            ['Qwen2.5','Unmodified full official vocabulary.','General deployment reference.'],
            ['Llama 3','Unmodified Meta reference tokenizer.','General deployment reference.'],
            ['o200k','Original tiktoken encoding.','General deployment reference.']],r'L{3.0cm}X L{3.0cm}')+
        block('Current repository contract','Seven method names, 19 frozen configurations. Other models and experiments live only in the historical archive and cannot be selected by the active loader.','remarkbox'))
    frame('Experimental Design','Common Inputs, Training Information and Timing',
        table(['Axis','Controlled implementation'],[
            ['Vocabulary budgets','1,024 / 2,048 for trained methods; all primitive and reserved IDs count.'],
            ['Allowed data','Identical scenario fit/auxiliary pools; generic trainers receive their union.'],
            ['Test input','Original accepted raw strings; no per-model filtering, truncation or normalization.'],
            ['Probability calibration','Same fit IDs for each scenario; mature models recalibrated for broad and specialized separately.'],
            ['Timing','Seven shuffled/interleaved full public encode/decode runs; median throughput; caches cleared.'],
            ['Statistical unit','Paper groups, not individual formulas copied across seeds.']],r'L{3.0cm}X')+
        block('Remaining asymmetry','Historical tuning counts and training wall time differ. Mature vocabulary budgets differ. These differences are disclosed rather than described as fully equal compute.','remarkbox'))
    frame('Evaluation Metrics','Efficiency, Fidelity and Information Measures',
        table(['Metric','Definition / interpretation'],[
            [r'Paper-macro tokens/byte $\downarrow$',r'$\frac1P\sum_p T_p/B_p$; prevents long papers from dominating the primary compression average.'],
            ['Length distribution','Tokens/formula and 50th/95th/99th percentiles; inspect long-tail inflation.'],
            ['Exact fidelity','Both decoded strings and concatenated raw token payloads equal the source.'],
            ['Encode / decode speed','Raw UTF-8 MiB/s through the complete public interface.'],
            [r'Fit unigram BPB $\downarrow$',r'$-B^{-1}\sum_i\log_2[(c_i+\alpha)/(N+\alpha|V|)]$; fit-only count model, not LM perplexity.'],
            ['Information / utilization',r'Renyi efficiency at $\alpha=1,2,2.5,3$; used IDs / actual vocabulary.']],r'L{3.3cm}X')+
        block('Sensitivity','Report smoothing at .01 / .1 / .5 / 1. Information efficiency and token count measure different properties; neither alone proves downstream quality.','remarkbox'),('greed','tokeval'))
    frame('Evaluation Metrics','Structure, Integrity and Robustness',
        table(['Metric','Definition','Evidence boundary'],[
            ['Command / number integrity','Fraction of reference lexemes with no internal token cut.','Does not require isolation as a standalone token.'],
            ['Lexical boundary P/R/F1','Token endpoints against reference lexical endpoints.','A byte tokenizer can obtain high recall.'],
            ['KaTeX crossing rate','Tokens partially overlapping a reference syntax span / all tokens.','A whole-formula token can obtain low crossing.'],
            ['Parameter-boundary recall','pylatexenc group-content endpoints retained as token boundaries.','Independent implementation; syntax proxy, not semantic gold.'],
            ['Robustness','Layout token variation; local symbol edit effects outside a 16-byte neighborhood.','The local edit is not mathematical equivalence.']],r'L{3.0cm}X X')+
        block('Coverage is part of the result','Unicode integrity, parser coverage and probe counts accompany the scores. No formula disappears from the common test merely because a diagnostic parser cannot measure it.','remarkbox'),('tokeval','texbleu'))
    frame('Experimental Audit','Three New Review Rounds of the Retained Release',
        table(['Round','Checks actually executed','Scope'],[
            ['1. Data / provenance','Reclassify every formula; verify file IDs, hash, license/origin rules and source-disjoint pools.','56,000 catalog records.'],
            ['2. Public-encoder replay','Reload every current model; reconstruct original payload; replay length, reference crossings and parameter boundaries.','234,556 formula-result rows.'],
            ['3. Independent aggregation','Recompute paper averages, integrity and fit-calibrated probability bits; check common dictionary and seven timing repeats.','19 configurations, 109 metric records.']],r'L{2.5cm}X L{2.8cm}')+
        block('Status','All three new reviews passed. The earlier v8 freeze/evaluation additionally passed eight protocol/result checks and 201 implementation tests.')+
        block('Reorganization integrity','Selected checkpoints are byte-identical copies of the frozen v8 artifacts. Current results are the seven-method subset, with no new training or test-based parameter changes.','remarkbox'))
    for capacity in (1024,2048):
        values=boards[f'{capacity}/broad']['metrics'];rows=[]
        for name,label in zip(METHODS,LABELS):
            m=values[name];lab=r'\textbf{'+label+'}' if name in METHODS[:2] else label
            rows.append([lab,f"{m['document_macro_tokens_per_byte']:.4f}",f"{100*m['independent_tree_crossing']:.2f}",f"{100*m['independent_scope_boundary_recall']:.2f}",f"{100*m['command_integrity']:.2f}",f"{100*m['number_integrity']:.2f}",f"{m['encode_mib_s']:.2f}"])
        frame('Results',f'New-Paper Results: {capacity:,}-ID Trained Vocabulary',
            r'Common test: 4,294 formulas from 27 new source papers.\par\vspace{2pt}'+
            table(['System',r'T/B $\downarrow$',r'Cross\% $\downarrow$',r'Scope\% $\uparrow$',r'Cmd\% $\uparrow$',r'Num\% $\uparrow$',r'MiB/s $\uparrow$'],rows,r'L{3.0cm}rrrrrr')+
            block('Reading the table','T/B is paper-macro tokens per UTF-8 byte. Crossing and scope recall use different external syntax references; improving one does not imply improving the other.')+
            block('Comparison scope','The two proposed systems share a dictionary. Byte BPE and Unigram share the ID budget and allowed data. Qwen2.5, Llama 3 and o200k retain their full vocabularies.','remarkbox'))
    rows=[]
    for name,label in zip(METHODS,LABELS):
        a=boards['1024/specialized']['metrics'][name];b=boards['2048/specialized']['metrics'][name]
        rows.append([label,f"{a['document_macro_tokens_per_byte']:.4f}",f"{b['document_macro_tokens_per_byte']:.4f}",f"{100*a['independent_scope_boundary_recall']:.2f}",f"{100*b['independent_scope_boundary_recall']:.2f}"])
    frame('Results','Strict Source-Domain Generalization',
        r'Common holdout: 1,428 formulas. Specialized training excludes target and adjacent source categories.\par\vspace{2pt}'+
        table(['System','T/B: 1,024','T/B: 2,048','Scope\%: 1,024','Scope\%: 2,048'],rows,r'L{3.5cm}rrrr')+
        block('What generalizes','Boundary AST compresses more than the same-dictionary BPE in both budgets. Scope recall changes by +0.57 pp / -0.28 pp; not every structural diagnostic improves.')+
        block('What remains uncertain','Source categories are disjoint by construction, but mathematical concepts can overlap. Few source papers and uneven domains produce wide uncertainty intervals.','remarkbox'))
    frame('Results','Compression Across Vocabulary Budgets',
        r'\begin{center}\includegraphics[width=.98\linewidth,height=5.2cm,keepaspectratio]{assets/compression_by_budget.pdf}\end{center}'+
        block('Interpretation','Solid lines: trained 1,024/2,048-ID systems. Dashed lines: fixed full-vocabulary general tokenizers. Lower is shorter; this is not a downstream-accuracy curve.','remarkbox'))
    frame('Results','Runtime and Source-Tree Crossing',
        r'\begin{center}\includegraphics[width=.98\linewidth,height=5.3cm,keepaspectratio]{assets/speed_structure.pdf}\end{center}'+
        block('Interpretation','Boundary AST is about 2.65--3.00 times faster than math-atom BPE across the four principal conditions. All parser/codec work is included. The result is from one Mac and a single-input public interface.','remarkbox'))
    rows=[]
    for c in (1024,2048):
        for s in ('broad','specialized'):
            p=stats['comparisons'][f'{c}/{s}/ast_boundary-vs-official_atomic_bpe']['metrics']
            a=p['document_macro_tokens_per_byte'];b=p['independent_scope_boundary_recall'];ci=a['familywise_ci95'];sj=b['familywise_ci95']
            rows.append([f'{c:,} / '+('broad' if s=='broad' else 'domain'),f"{a['difference_left_minus_right']:+.5f}",f'[{ci[0]:+.5f}, {ci[1]:+.5f}]',f'[{100*sj[0]:+.2f}, {100*sj[1]:+.2f}]'])
    frame('Results','Paired Paper-Cluster Statistical Evidence',
        r'Difference: Boundary AST minus math-atom BPE. Negative compression differences favor AST.\par\vspace{2pt}'+
        table(['Condition','T/B difference','Family 95\% CI','Scope difference CI (pp)'],rows,r'L{2.4cm}r L{3.5cm}X')+
        block('Design','200,000 paper-cluster resamples; Bonferroni intervals retain the original family of 144 comparisons. We did not shrink the family after choosing seven systems.')+
        bullets(['All four adjusted compression intervals exclude zero in favor of Boundary AST.',
                 'Scope and crossing intervals do not all exclude zero; point estimates are not universal structural gains.',
                 'Intervals condition on the realized vocabulary and observed training seeds. They do not cover all possible retraining outcomes.']))
    rows=[]
    for name,label in zip(METHODS,LABELS):
        a=boards['1024/broad']['metrics'][name];b=boards['2048/broad']['metrics'][name]
        rows.append([label,f"{a['fit_predictive_unigram_bits_per_byte']:.3f}",f"{b['fit_predictive_unigram_bits_per_byte']:.3f}",f"{100*a['vocab_utilization']:.1f}",f"{100*b['vocab_utilization']:.1f}"])
    frame('Results','Probability Coding and Vocabulary Utilization',
        table(['System','Fit BPB: 1,024','Fit BPB: 2,048','Used\%: 1,024','Used\%: 2,048'],rows,r'L{3.4cm}rrrr')+
        block('Interpretation','Fit BPB uses the same scenario fit records and actual vocabulary size. The much larger general vocabularies incur a different smoothing term and utilization denominator.')+
        block('Why several measures are necessary','Low token count can coexist with sparse usage or unstable boundaries. These probability readouts are descriptive token statistics; no trained language-model perplexity is reported.','remarkbox'),('greed','tokeval'))
    rows=[]
    for name,label in zip(METHODS,LABELS):
        rows.append([label,str(ranks['combined']['1024']['ranking'].index(name)+1),str(ranks['combined']['2048']['ranking'].index(name)+1),
            f"{ranks['combined']['1024']['scores'][name]:.3f}",f"{ranks['combined']['2048']['scores'][name]:.3f}"])
    frame('Results','Current Seven-System Composite Ranking',
        table(['System','Rank: 1,024','Rank: 2,048','Score: 1,024','Score: 2,048'],rows,r'L{3.4cm}rrrr')+
        block('Frozen metric preferences','Compression 30\%; encode speed 25\%; KaTeX crossing 20\%; command integrity 10\%; number integrity 5\%; fit BPB 10\%. Broad and specialized scenarios each receive 50\%.')+
        block('How to use the ranking','The ranking favors our complete specialized systems under this preference. It is relative to the seven retained methods, omits scope recall from its weighted score and cannot establish downstream superiority.','remarkbox'))
    frame('Results','Why RL Is Not a Final Production Route',
        table(['Same-reward control','RL mean utility gain','Wins','95\% interval crosses zero?'],[
            ['Fixed random search','0.116\%','7 / 15','Yes'],['Cross-entropy search','0.199\%','8 / 15','Yes']],r'L{3.2cm}rr X')+
        block('Controlled evidence','Three paper splits and five seeds per split; shared tasks, features, initialization and 34 actual terminal evaluations. The gate required at least 0.5\% utility improvement, 12/15 wins and core noninferiority.')+
        bullets(['The gate was not met; these results remain in history, outside the current seven-system registry.',
                 'The non-RL Boundary AST replacement preserves core point-estimate tolerances relative to the old v5, with explicit structural trade-offs.',
                 'This does not refute the ICML end-to-end RL paper, whose reward is language-model loss.'])+
        r'\src{Historical v8 controlled-terminal decision; 15 pairs are conditional on three potentially overlapping source splits.}',('rl'))
    frame('Downstream Plan','Four Priority Downstream Tasks',
        table(['Task','Input and output','Primary evaluation'],[
            ['1. PDE solution / solver generation','Equation + initial/boundary conditions; generate an analytic candidate or executable numerical solver.','Residual, boundary error, nRMSE and execution success.'],
            ['2. Governing-equation discovery','Field observations / derivatives; generate a symbolic PDE and fit coefficients.','Structure recovery, coefficient error, held-out prediction and noise robustness.'],
            ['3. Symbolic manipulation','Integration, differentiation, simplification and ODE solving.','Executable/symbolic equivalence, validity and unseen operator combinations.'],
            ['4. Mathematical reasoning','Problem text + formulas; generate derivation and final answer.','Verified final-answer accuracy and length/generalization strata.']],r'L{2.9cm}X X')+
        block('Data and task separation','PDEBench supplies numerical PDE tasks; EqGPT motivates discovery; SymbolicMathematics and MATH motivate symbolic/reasoning tasks. Their test cases must not enter tokenizer or model adaptation training.','remarkbox'),('pdebench','eqgpt','symbolic','math'))
    frame('Downstream Plan','Tokenizer Replacement Does Not Require Full Retraining',
        table(['Option','Procedure','Experimental use'],[
            ['Small models from scratch','Same backbone, raw data and initialization seeds.','Clean comparison; start around 100--300M parameters.'],
            ['Adapt an open backbone','Replace vocabulary; initialize embeddings; continue pretraining.','Retain a no-replacement adaptation control.'],
            ['Align / transplant embeddings','Use old-token decompositions or learned alignment.','Equal adaptation cost against simple initialization.']],r'L{3.0cm}X X')+
        block('What must change','New IDs cannot reuse old embedding rows by ID. Replace input/output vocabulary matrices; other Transformer weights can be retained.')+
        block('Core difficulty','Formula-only data are insufficient for reasoning and solver generation. Additional context/task data and mixed-text coverage are needed.','remarkbox'),('tokalign','compute'))
    frame('Downstream Plan','Compute, Vocabulary Cost and Fair Training Budgets',
        block('Vocabulary-dependent parameters',r'Tied embeddings: $P_{\mathrm{vocab}}=|V|d$. At $d=2{,}048$:\\$|V|=2{,}048\Rightarrow4.2$M; $|V|\approx200{,}000\Rightarrow409.6$M parameters. Untied input/output matrices roughly double this component.')+
        table(['Budget issue','Protocol requirement'],[
            ['Training amount','Use the same raw examples/bytes; also report tokens and measured FLOPs.'],
            ['Context length','Report raw-byte coverage and truncation, not only a common token window.'],
            ['Model size','Report non-embedding and total parameters separately.'],
            ['Compute scale',r'Heuristic $C\approx6ND$: a 1B-parameter, 10B-token run is about $6\times10^{19}$ FLOPs before additional overhead.'],
            ['Replication','Seven systems times three seeds is 21 runs per task configuration.']],r'L{2.8cm}X')+
        block('Staged resource plan','Start with equal-capacity small-model tests. Scale after a reproducible task gain; cost full-vocabulary deployment references separately.','remarkbox'),('compute'))
    frame('Downstream Plan','Implementation Sequence and Decision Criteria',
        table(['Phase','Work','Advance only if'],[
            ['A. Freeze task protocol','Choose equation families, source splits, validation rules and resource accounting.','Data leakage and task evaluator checks pass.'],
            ['B. Small-model pilot','Two routes + five baselines; matched raw training data, same task code, three seeds.','Gain is stable under task accuracy and compute accounting.'],
            ['C. Open-model adaptation','Token/embedding initialization ablations with equal adaptation budgets.','Benefits exceed the no-replacement control.'],
            ['D. Generalization / scale','New PDE families, long formulas, noise and mixed text/formula prompts.','Gain persists across distributions without unacceptable regressions.']],r'L{1.8cm}X X')+
        block('Failure conditions to investigate','No task gain despite better compression; invalid expression generation; failure on unseen domains; adaptation cost larger than inference savings; arithmetic or scope errors hidden by string similarity.','remarkbox'))
    frame('Conclusions','What We Can Claim, and What We Must Test Next',
        block('Supported by current evidence','A classified, source-traceable formula catalog and reproducible frozen experiments are available. Boundary AST improves compression over a same-dictionary BPE control while offering a useful speed/structure trade-off.')+
        table(['Supported statement','Boundary'],[
            ['Two routes are deployable','Standalone tokenizers only; no downstream model training has been completed.'],
            ['Three current audits passed','Automated data, implementation and experiment checks; not external certification.'],
            ['Some core metrics improve','Not all structure metrics or adjusted intervals favor one method.'],
            ['Specialization is promising','Sparse PDE coverage, uneven domains and mixed-text transfer remain open.']],r'L{3.4cm}X')+
        block('Next scientific question','Do the observed intrinsic improvements survive embedding adaptation and produce better symbolic/PDE task performance at a defensible total training cost?','remarkbox'))
    for i in range(0,len(refs),9):
        items=[]
        for r in refs[i:i+9]:
            items.append(r'\item '+escape(r['authors'])+' ('+str(r['year'])+r'). \href{'+r['url']+'}{'+escape(r['title'])+'}. '+escape(r['venue'])+'.')
        frame('References',f'References and Technical Context ({i//9+1}/3)',
            r'{\scriptsize\begin{enumerate}\setlength{\itemsep}{4pt}'+'\n'+'\n'.join(items)+r'\end{enumerate}}'+
            r'\src{Targeted primary-source search; not an exhaustive systematic review. Publication and preprint status are distinguished.}')
    preamble=r'''% !TEX program = xelatex
\documentclass[11pt,aspectratio=43]{beamer}
\input{template.tex}
\usepackage{microtype}
\newcommand{\src}[1]{\par\vspace{2pt}{\fontsize{6.5}{7.5}\selectfont\color{defink}#1\par}}
\newcommand{\major}[1]{\renewcommand{\currentmajor}{#1}}
\newcommand{\tbl}{\renewcommand{\arraystretch}{1.16}\setlength{\tabcolsep}{3pt}}
\hypersetup{colorlinks=true,urlcolor=defink,linkcolor=lecblue,pdfauthor={Tongren Xiao},pdftitle={Design of a Specialized Tokenizer for Mathematical Text}}
\title[Math Tokenization]{\texorpdfstring{Design of a Specialized Tokenizer\\[1ex]for Mathematical Text}{Design of a Specialized Tokenizer for Mathematical Text}}
\author[Tongren Xiao]{\texorpdfstring{Tongren Xiao\\[1ex]\texttt{tongrenxiao7@gmail.com}}{Tongren Xiao}}
\institute{Department of Mathematics, Sun Yat-sen University}
\date[October 10, 2026]{October 10, 2026}
\begin{document}
\footnotesize\tight
\major{Research Presentation}
\begin{frame}[noframenumbering]\titlepage
\note{Cover identity, email, institute and date are retained from the supplied GBFR reference. The research artifacts and literature search were updated on October 13, 2026.}
\end{frame}
'''
    text=preamble
    for f in frames:
        body=f['body']
        if f['title'] in ('Can Mathematical Tokenization Improve Model Capability?','Classical Subword Tokenization','Common Inputs, Training Information and Timing'):
            body=body.replace(r'{\scriptsize\tbl',r'{\footnotesize\tbl')
        if f['title']=='Efficiency, Fidelity and Information Measures':
            body=body.replace(r'{\scriptsize\tbl',r'{\scriptsize\renewcommand{\tbl}{\renewcommand{\arraystretch}{1.27}\setlength{\tabcolsep}{3pt}}\tbl')
        if f['title']=='References and Technical Context (3/3)':body=body.replace(r'\setlength{\itemsep}{4pt}',r'\setlength{\itemsep}{6.5pt}')
        text+='\n\\major{'+f['section']+'}\n\\begin{frame}[t]{'+f['title']+'}\n\\relax\n'+body
        if f['note']:text+='\n\\note{'+f['note']+'}\n'
        text+='\n\\end{frame}\n'
    text+='\\end{document}\n'
    assert not any(ord(c)<32 and c!='\n' for c in text),'Invalid TeX source control character'
    # Python string newline markers in table helper must become actual lines.
    text=text.replace('\\toprule\\n','\\toprule\n')
    (OUT/'mathematical_tokenizer.tex').write_text(text)
    evidence={'body_frames':len(frames),'total_pages':len(frames)+1,'title':'Design of a Specialized Tokenizer for Mathematical Text',
        'cover_information_retained':True,'cover_date':'October 10, 2026','research_date':'2026-10-13',
        'figures':3,'references':len(refs),'methods':METHODS,'table_source_sha256':hashlib.sha256((ROOT/'experiments/current/rankings.json').read_bytes()).hexdigest(),
        'frames':[{'index':i+1,'title':f['title'],'section':f['section']} for i,f in enumerate(frames)]}
    (OUT/'content_manifest.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('Beamer frames',len(frames)+1,'figures 3','references',len(refs))


if __name__=='__main__':main()
