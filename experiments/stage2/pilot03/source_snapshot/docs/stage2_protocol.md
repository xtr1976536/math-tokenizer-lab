# Stage 2: learned signals and autonomous boundary policies

Status: exploratory pilot, not a replacement for the frozen stage-1 registry.
The supplied success requirements are conjunctive: substantially reduced human
intervention **and** excellent compression, structural quality, generalization,
losslessness and end-to-end speed. A pilot improvement on one proxy is not success.

## First executable experiment

1. Train original Hugging Face byte BPE on a source-disjoint fit subset. Reuse its
   exact dictionary for all signal ablations. No LaTeX command table, AST,
   lexical labels or reference parser participates in these training routes.
2. Estimate raw-byte conditional distributions using fit data. Compare entropy,
   last-byte surprise and conditional-distribution change signals; include zero
   and shuffled-signal controls. Compile a learned byte-context lookup table and
   dictionary shortest-path inference into a single native public encoder.
3. Train a **causal Bernoulli boundary policy**, with one learnable logit per
   observed two-byte context, using score-function gradients and leave-one-out
   reward-to-go advantages. Its feedback is byte prediction by a fit-only frozen
   patch-state predictor. Every boundary action precedes the predicted byte and
   sees only prior bytes. It has no command, number or tree reward.
4. Compare that learning with no updates, random search and CEM under the same
   number of raw-byte prediction evaluations, same predictor and same initial
   policy. Record uncertainty across seeds. Training uses a general patch-size
   cap and an automatically adjusted boundary-rate multiplier; these are human
   engineering choices, not mathematical boundary supervision.
5. Project learned logit preferences onto the fixed dictionary lattice. This is
   an explicit deployment approximation, not exact reproduction of the learned
   stochastic router. Report router effects and deployed-tokenizer effects
   separately. Select signal strength using training-internal prediction-code
   feedback and a 1% token/byte guard; no structural metric enters selection.

The patch predictor is a smoothed finite-state statistical model, not a neural
world model. Its causal byte loss is a training proxy. The separately evaluated
token-bigram prediction code is a code length for a chosen token sequence, not
the exact marginal likelihood of its raw string and not downstream LM quality.
Do not call this a reproduction of BLT, H-Net or the Dauncey architecture.

## Data and fairness

The classified 56,000-formula catalog is immutable. Partition only the allowed
stage-1 broad fit pool into paper-disjoint fit, policy-reward, development and
internal evaluation subsets. Hold out all number-theory source papers as an
additional domain probe. Exclude earlier public test IDs from every training
and selection scope. These are newly partitioned internal pilot scopes;
the underlying data were previously available in stage 1. They are **not** a
new independent benchmark. A later confirmatory release requires new sources.

All new ablations have the same byte-BPE dictionary, ID budget and fit data.
Retrain byte BPE and the two stage-1 recipes on the same pilot fit subset for
deployment comparisons; the AST/math-atom pair share their own dictionary and
therefore are equal-data route references, not same-dictionary causal controls.
Measure identical untruncated strings using the established intrinsic evaluator,
including independent KaTeX crossings and pylatexenc scope recall. Time complete
public encode calls, including signal lookup and native inference, in interleaved
rounds; report batch distributions, memory, and decoder speed.

## Success gates (specified before execution)

- Autonomous route: zero manually labeled boundaries, no LaTeX command inventory,
  no AST training labels, no hand-weighted math-metric reward. Disclose remaining
  architectural choices and retain byte fallback. This is low-intervention, not
  a claim of literally no human-designed assumptions.
- Fidelity: 100% raw-byte payload and string round trips on every reported suite.
- Deployment guard against the best same-fit stage-1 reference: paper-macro
  token/byte deterioration <=1%; command and number integrity, scope recall and
  lexical F1 deterioration <=1 percentage point each; crossing deterioration
  <=1 percentage point. At least one practical gain: >=5% lower token/byte,
  >=5 percentage points better structural quality, or >=20% faster encoding.
  Also require >=90% of the same-fit Boundary AST encode speed. All reported
  suites must pass guards; these are research targets, not universal guarantees.
- Learning evidence: RL must improve held-development causal byte loss by >=1%
  over both equal-budget search controls, with a paper-cluster interval excluding
  zero and consistent direction across seeds. Otherwise do not attribute gains
  to RL. A successful non-RL autonomous learner remains acceptable to the user.
- Confirmatory success cannot be declared on this pilot: multiple capacities,
  expanded seeds, independent new-source/domain tests, matched-compute neural
  prediction, and downstream mathematical operation tests are still required.

## Audit and next experiments

Run at least three concrete audit rounds: source/ID/hash isolation; algorithm,
causality, gradient, native/reference and serialization checks; independent
per-formula aggregation, equal oracle budgets and interleaved timing checks.
Keep every candidate and seed, including negative results. Do not select seeds
using evaluation scores. A later neural boundary learner and verified mathematical
operation model must be evaluated separately before any understanding claim.

Primary methodological sources:
- https://proceedings.mlr.press/v306/dauncey26a.html
- https://aclanthology.org/2025.acl-long.453/
- https://arxiv.org/abs/2507.07955
- https://aclanthology.org/2024.emnlp-main.40/

## Pilot02 follow-up (after pilot01 negative deployment result)

Pilot01 selected zero signal for every deployable signal/learned policy. Therefore
its native speed benefit is an engineering/path-search result, not learning.
Pilot02 adds exact token-transition prediction code as a path-dependent signal.
Its finite-state decoder optimizes the whole token sequence including BOS/EOS;
transition tables are fit-only. Three alternating fit/segment rounds and a fixed
rate-cost grid (0,1,2,4,8,16) select only on the internal reward scope under the
original 1% compression guard. No syntax metrics enter this selection. Additional
memory and speed cost must be counted. Pilot02 reuses the exposed internal scopes
as an adaptive development follow-up and cannot supply independent confirmation.

## Pilot03 follow-up: discovery of deployable units

The learned two-byte signal also proposes primitive strings: collect contiguous
runs below a fit-weighted signal quantile (.4,.6,.8), bounded by 32 bytes, and rank
observed runs by frequency times byte savings. Reserve at most 384 such units plus
the 256 byte fallbacks, then train original HF BPE on a generic reversible PUA
codec. This codec has no command scanner, unicode-unit scanner or AST protection.
Select among three quantiles with the same training-reward prediction-code and
compression guard, never command/tree scores. Test both byte-statistical signals
and every RL/search/no-update seed. Frequency-based vocabulary ranking and the
quantile grid are disclosed human architecture choices. Different resulting
vocabularies make this a whole-route comparison, not a fixed-dictionary ablation.
Pilot03 is adaptive exploration on the same internal scopes, not confirmation.
