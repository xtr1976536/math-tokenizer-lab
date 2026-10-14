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
        body=re.sub(r'\\src\{[^{}]*\}', '', body)
        frames.append({'section':section,'title':title,'body':body,'cites':tuple(cites),'note':note})
    frame('Research Question','Can Mathematical Tokenization Improve Model Capability?',
        block('Research question',r'Can we train tokenizers specifically for mathematical text that improve both \textbf{intrinsic tokenization} and \textbf{downstream mathematical capability}?')+
        r'\vspace{8pt}'+table(['Question','What would count as evidence?'],[
            ['RQ1: tokenizer quality','Lossless compression, robustness, speed, etc.'],
            ['RQ2: model capability','Mathematical tasks such as PDE/ODE solving.'],
            ['Current status','Tokenizer experiments completed; downstream evaluation to follow.']],r'L{3.0cm}X'),
        note='This page is intentionally limited to the research question and requested short evidence descriptions.')
    frame('Research Question','Study Overview',
        table(['Component','Summary'],[
            ['Dataset','We constructed and classified a dataset of mathematical formulas from multiple mathematical fields.'],
            ['Models','We proposed two specialized mathematical tokenizers.'],
            ['Baselines','We selected five tokenizers: classical BPE and Unigram methods, and tokenizers used by mature large language models.'],
            ['Results','Our two tokenizers achieve leading performance on core intrinsic metrics and in the overall comparison.']],r'L{2.4cm}X'),
        note='Leading performance refers to selected core metrics and the stated aggregate comparison; the experimental tables do not show universal dominance on every individual metric.')
    frame('Baseline','Baseline Tokenizers',
        table(['Category','Tokenizer','Actual ID space','Code / tokenizer source'],[
            ['Classical tokenizers',r'Byte BPE \rcite{bpe}','1,024 / 2,048',r'\href{https://github.com/huggingface/tokenizers}{huggingface/tokenizers}'],
            ['Classical tokenizers',r'SP Unigram \rcite{sentencepiece}','1,024 / 2,048',r'\href{https://github.com/google/sentencepiece}{google/sentencepiece}'],
            ['General-Purpose Tokenizers',r'Qwen2.5 \rcite{qwen}','151,665',r'\href{https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/tokenizer.json}{Qwen2.5-0.5B/tokenizer.json}'],
            ['General-Purpose Tokenizers',r'Llama 3 \rcite{llama}','128,256',r'\href{https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/tokenizer.py}{meta-llama/llama3/tokenizer.py}'],
            ['General-Purpose Tokenizers',r'o200k \rcite{tiktoken}','200,019',r'\href{https://github.com/openai/tiktoken}{openai/tiktoken}']],r'L{2.4cm}L{1.5cm}L{1.7cm}X')+
        r'\vspace{6pt}'+bullets(['Qwen2.5: used by the Qwen2.5 model family.',
                                'Llama 3: used by the Meta Llama 3 model family.',
                                r'o200k\_base: used by GPT-4o and GPT-4.1.']),
        note='Source links point to the actual public implementation or tokenizer artifact used. Actual ID spaces are taken from the frozen active registry; SP Unigram includes byte fallback IDs.')
    frame('Baseline','SP Unigram: Training and Inference',
        table(['Training step','Procedure'],[
            ['Preparation',r'Use raw strings without language-specific word segmentation; represent spaces as a symbol, usually {\fontspec{Apple Symbols}▁}.'],
            ['1. Initialize','Create a large seed vocabulary of characters, frequent substrings and symbols; assign initial piece probabilities.'],
            ['2. EM iterations','E: sum over possible segmentations to obtain expected piece counts. M: re-estimate probabilities from those counts.'],
            ['3. Prune','Remove pieces with low contribution to corpus likelihood; repeat EM and pruning to reach 1,024 or 2,048 IDs.'],
            ['4. Final model','A piece vocabulary and probabilities; an optional byte fallback table.']],r'L{2.1cm}X')+
        r'\vspace{2pt}\textbf{Inference}\par '+
        r'Build a segmentation lattice; for \texttt{abc}: $a+b+c$, $ab+c$, $a+bc$, or $abc$.'+
        r'\[P(s)=\prod_i p(s_i),\qquad s^*=\arg\max_{s\in\mathcal S(x)}P(s).\]'+
        r'Viterbi dynamic programming returns the most probable token sequence. \rcite{unigram} \rcite{sentencepiece}',
        note='Raw training describes the SentencePiece approach. For this experiment identity normalization and whitespace-preserving settings are used. E-step uses forward/backward dynamic programming rather than explicit exponential enumeration. P(s) is one segmentation probability; the training string likelihood is P(x)=sum_s P(s). Official SentencePiece uses a Bayesian/digamma probability update and loss-based pruning, so the next example illustrates basic maximum-likelihood EM rather than byte-for-byte trainer behavior. U+2581 denotes the visible space symbol.')
    frame('Baseline','SP Unigram: A Two-Round EM Example',
        r'\textbf{Setup:} one training string \texttt{ab}; pieces $a,b,ab$; initially $p(a)=p(b)=p(ab)=1/3$.\par'+
        r'\vspace{3pt}\textbf{E step: segmentation likelihoods and posteriors}\par'+
        table(['Round','$(a,b)$ likelihood','$ab$ likelihood','$P(x)$','Posterior: $(a,b)$ / $ab$'],[
            ['1',r'$1/3\times1/3=1/9$',r'$1/3=3/9$',r'$4/9$',r'$1/4=0.25$ / $3/4=0.75$'],
            ['2',r'$0.2\times0.2=0.04$',r'$0.6$',r'$0.64$',r'$0.0625$ / $0.9375$']],r'L{1cm}L{2.5cm}L{1.5cm}L{1cm}X')+
        r'\vspace{4pt}\textbf{M step: expected counts and normalized probabilities}\par'+
        table(['Round','$\mathbb E[n_a],\mathbb E[n_b],\mathbb E[n_{ab}]$','Total','$p(a),p(b),p(ab)$'],[
            ['1',r'$0.25,\;0.25,\;0.75$',r'$1.25$',r'$0.2,\;0.2,\;0.6$'],
            ['2',r'$0.0625,\;0.0625,\;0.9375$',r'$1.0625$',r'$1/17,\;1/17,\;15/17$' + r'\newline $\approx0.0588,\;0.0588,\;0.8824$']],r'L{1cm}L{3.5cm}L{1cm}X')+
        r'\[P(x)=\sum_{s\in\mathcal S(x)}\prod_i p(s_i),\qquad p(v)=\frac{\mathbb E[n_v]}{\sum_u\mathbb E[n_u]}.\]'+
        r'For this one-string example, $p(ab)\to1$ and $p(a),p(b)\to0$: EM favors the whole piece $ab$. Low-contribution pieces become pruning candidates; required characters remain protected. \rcite{unigram}',
        note='Simplified maximum-likelihood EM example supplied by the user; exact arithmetic was checked with rational numbers. Real SentencePiece uses a Bayesian probability update, estimates pruning loss, and preserves required symbols. The toy example cannot imply universal deletion of individual characters.')
    frame('Related Work','Our Two Routes: Contribution and Prior Art',
        table(['Dimension','Math-atom BPE','Boundary AST'],[
            ['Vocabulary','Fit-derived complete-command/Unicode atoms; original BPE.','Exactly the same byte vocabulary and ID budget.'],
            ['Segmentation','Original learned BPE merge order.','Minimum additive token/tree/boundary cost on a legal source graph.'],
            ['Mathematical prior','Lexical units observed in training.','Lexical units plus a restricted source parser.'],
            ['Deployment','Native reversible codec + original Rust BPE.','Fused native trie traversal and dynamic programming.'],
            ['Novelty boundary','A reproducible domain-specific training recipe.','A tested structural trade-off; shortest paths and AST ideas are not new.']],r'L{2.2cm}X X')+
        block('Central comparison','Holding the dictionary fixed isolates the effect of the structural segmentation rule. Comparing against generic tokenizers instead measures the complete specialized system.','remarkbox'),('cmer','pathpiece','mathgpt'))
    frame('Dataset','Dataset',
        table(['Source','Accepted formulas','Source link'],[
            ['arXiv (pre-2020 papers)','51,229',r'\href{https://arxiv.org}{arxiv.org}'],
            ['OpenAI math repository','2,810',r'\href{https://github.com/openai/math}{github.com/openai/math}'],
            ['FunGrim','1,889',r'\href{https://github.com/fredrik-johansson/fungrim}{github.com/fredrik-johansson/fungrim}'],
            ['SymPy','72',r'\href{https://github.com/sympy/sympy}{github.com/sympy/sympy}']],r'L{3.2cm}L{2cm}X')+
        r'\vspace{16pt}\textbf{Construction process}\par\vspace{6pt}'+
        r'\begin{center}\includegraphics[width=\linewidth]{assets/dataset-pipeline.drawio.pdf}\end{center}')
    frame('Dataset','Dataset',
        r'\begin{center}\includegraphics[width=.96\linewidth,height=7.4cm,keepaspectratio]{assets/dataset_domains_only.png}\end{center}')
    domainrows=[]
    domains=manifest['domain']
    domain_order=['analysis','geometry_topology','combinatorics','algebra','number_theory','mathematical_physics','optimization','probability','statistics','logic','pde','ode','linear_algebra']
    for i in range(0,len(domain_order),2):
        a=domain_order[i];b=domain_order[i+1] if i+1<len(domain_order) else None
        domainrows.append([escape(a.replace('_',' / ')),f'{domains[a]:,}',escape(b.replace('_',' / ')) if b else '',f'{domains[b]:,}' if b else ''])
    left=table(['Formula type','Count','Formula type','Count'],domainrows,r'X r X r')
    right=table(['Provenance label','Formulas'],[
        ['Historical human','51,229'],['Possible mixed','2,810'],['Unknown','1,961'],['Fully AI generated','0']],r'X r')
    frame('Dataset','Dataset',
        r'\begin{columns}[T,onlytextwidth]\begin{column}{.61\linewidth}'+r'\textbf{Formula types}\par'+left+
        r'\end{column}\begin{column}{.36\linewidth}'+r'\textbf{Provenance labels}\par'+right+
        r'\vspace{4pt}\textbf{Complexity classes}\par'+table(['Simple','Complex'],[['37,400','18,600']],r'rr')+
        r'\end{column}\end{columns}'+
        r'\vspace{4pt}\textbf{Complexity score}\par'+
        r'\[s=.40\bar L+.20\bar C+.15\bar D+.15\bar S+.10\bar O,\quad \bar z=\min(z/z_0,2).\]'+
        r'{\scriptsize $L$: lexemes; $C$: command types; $D$: nesting; $S$: symbol types; $O$: operators.\\Normalizers $(L_0,C_0,D_0,S_0,O_0)=(60,6,3,18,12)$; complex iff $s\ge.65$.}\par'+
        r'\vspace{3pt}\textbf{Example:} $\frac{1}{s^a}$ has $(L,C,D,S,O)=(9,1,1,5,2)$, hence $s\approx0.202<0.65$: simple.\par'+
        r'\vspace{5pt}\textbf{Train: 36,615 formulas; validation: 7,693; test: 11,692.}',
        note='The example matches the stored complexity metadata for the accepted formula. Train=v6_train, validation=v6_dev, test=union of v6_fresh_test,v7_external_test,v8_external_test. These are disjoint and sum to56000. The scenario-specific fitting scopes remain described in the retained experimental settings.')
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
    frame('Next Work','Limitations and Next Steps',
        block('Downstream tasks','Validate the tokenizers on PDE solving, ODE solving, and mathematical reasoning beyond equation solving.')+
        table(['Step','Experimental procedure'],[
            ['1. Architecture','Select a suitable model architecture.'],
            ['2. Tokenizer integration','Connect each tokenizer to the same architecture.'],
            ['3. Training','Train each model fully from scratch.'],
            ['4. Evaluation','Select suitable benchmarks and evaluate downstream task performance.']],r'L{2.3cm}X')+
        block('Open issues',r'\textbf{Benchmarks:} identify suitable benchmarks for each mathematical task.\\\textbf{Compute:} full training from scratch requires substantial computational resources.','remarkbox'))
    used=[]
    for f in frames:
        for k in re.findall(r'\\rcite\{([^}]+)\}',f['body'])+list(f['cites']):
            if k not in used:used.append(k)
    refs=[refmap[k] for k in used];numbers={k:i+1 for i,k in enumerate(used)}
    def cite(k):return r'\hyperlink{ref-'+k+'}{('+str(numbers[k])+')}'
    for f in frames:
        body=re.sub(r'\\rcite\{([^}]+)\}',lambda m:cite(m.group(1)),f['body'])
        if f['cites']:
            marker=' '.join(cite(k) for k in f['cites'])
            ends=list(re.finditer(r'\\end\{(?:definitionbox|remarkbox|theorembox)\}',body))
            if ends:
                end=ends[-1].start();body=body[:end]+marker+'\n'+body[end:]
            else:body+='\n'+marker
        f['body']=body
    refpages=(len(refs)+6)//7
    for i in range(0,len(refs),7):
        items=[]
        for r in refs[i:i+7]:
            items.append(r'\par\noindent\hypertarget{ref-'+r['id']+r'}{}\textbf{('+str(numbers[r['id']])+')} '+escape(r['authors'])+' ('+str(r['year'])+r'). \href{'+r['url']+'}{'+escape(r['title'])+'}. '+escape(r['venue'])+r'.\par\vspace{8pt}')
        frame('References',f'References ({i//7+1}/{refpages})',r'{\scriptsize '+ '\n'.join(items)+'}')
    (OUT/'active_references.json').write_text(json.dumps(refs,indent=2)+'\n')
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
        if f['title']=='Can Mathematical Tokenization Improve Model Capability?':
            body=body.replace(r'{\footnotesize\tbl',r'{\footnotesize\renewcommand{\tbl}{\renewcommand{\arraystretch}{2.3}\setlength{\tabcolsep}{3pt}}\tbl')
        if f['title']=='Study Overview':
            body=body.replace(r'{\scriptsize\tbl',r'{\footnotesize\renewcommand{\tbl}{\renewcommand{\arraystretch}{2.8}\setlength{\tabcolsep}{5pt}}\tbl')
        if f['title']=='SP Unigram: Training and Inference':
            body=body.replace(r'{\scriptsize\tbl',r'{\scriptsize\renewcommand{\tbl}{\renewcommand{\arraystretch}{1.35}\setlength{\tabcolsep}{3pt}}\tbl')
        if f['title']=='Baseline Tokenizers':
            body=body.replace(r'{\scriptsize\tbl',r'{\scriptsize\renewcommand{\tbl}{\renewcommand{\arraystretch}{1.8}\setlength{\tabcolsep}{3pt}}\tbl')
        if f['title']=='Dataset' and 'Construction process' in body:
            body=body.replace(r'{\scriptsize\tbl',r'{\footnotesize\renewcommand{\tbl}{\renewcommand{\arraystretch}{1.8}\setlength{\tabcolsep}{3pt}}\tbl')
        if f['title']=='Limitations and Next Steps':
            body=body.replace(r'{\scriptsize\tbl',r'{\footnotesize\renewcommand{\tbl}{\renewcommand{\arraystretch}{1.25}\setlength{\tabcolsep}{3pt}}\tbl')
        if f['section']=='References':
            body=body.replace(r'\vspace{8pt}',r'\vspace{14pt}')
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
        'figures':3,'flowcharts':1,'references':len(refs),'methods':METHODS,'table_source_sha256':hashlib.sha256((ROOT/'experiments/current/rankings.json').read_bytes()).hexdigest(),
        'frames':[{'index':i+1,'title':f['title'],'section':f['section']} for i,f in enumerate(frames)]}
    (OUT/'content_manifest.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('Beamer frames',len(frames)+1,'figures 3','references',len(refs))


if __name__=='__main__':main()
