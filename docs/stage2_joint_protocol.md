# Stage 2 next cycle: longer contexts, joint learning, verified operations

Exploratory scope; original success gates remain unchanged. No production
promotion from one proxy. All existing natural internal scopes are exposed.

## Pilot10: longer-context causal boundary policy and co-adaptation

Actor:16 preceding raw bytes, additive position-specific byte logits (16x257
parameters plus bias). No character classes, command dictionary, AST labels or
mathematical quality reward. This is a longer-context linear neural actor, not a
recurrent/hierarchical world model. Shared critic has16-byte current prefix plus
8-byte preceding patch suffix,16-d embedding,96 hidden units;600x512 fit targets.
All methods begin with identical actor and predictor. Compare boundary RL,
random search,CEM,no-update under frozen and alternating-fit critic variants.
32 rounds x8 reward formulas x4 rollouts; paper-balanced reward pool48. Rotate
mini-batches, use identical seed/data/evaluation budgets. Joint variants update
critic for50x512 fit targets every8 rounds (200 total steps), using only fit data
and their own causal sampled boundaries. Frozen variants are an ablation; do
not claim equal compute between frozen and jointly trained conditions.

Calibrate expected boundary rate using reward data only before evaluating;
measure proper causal byte losses on128 balanced formulas per scope (4 draws),
and report deterministic router separately. Critic/actor co-adaptation comparisons
are system comparisons, not pure actor causal effects. Also cross-evaluate every
actor under common initial/final no-update predictors to detect critic-specific
advantages. Full formula lengths are retained during scoring; training uses
complete formulas, bounded byte-feature windows and disclosed patch cap32.

## Pilot11: automatically verified mathematical-operation feedback

Create separate synthetic algebra operation pairs; keep natural benchmark
immutable. SymPy generates expressions and labels, with independent polynomial
coefficient checks for derivative, expansion and shift-substitution. Proposed
operations: derivative x, derivative y, expand, shift x->x+1. Generated forms
are bivariate factored/summed polynomials at training degree2-4; hold out generated
canonical expressions and stronger factor/degree compositions. These are
automatically labeled tasks in a narrow polynomial environment, not scientific
formula coverage and not a general mathematics world model. Explicit human
priors: generator families, operations, exact verification rules and budgets.

Train a small conditional byte seq2seq GRU to predict operation results. Input
patch embeddings are mean byte embeddings; patch GRU encodes the variable-length
segments; target GRU predicts output bytes conditioned on source+operation.
Train on randomly segmented source patches at rates.2/.4/.6. Compare shuffled
target/operation controls where feasible. Score conditional target byte NLL
with exact EOS and untruncated input/output; it is not raw-string LM likelihood.
The input is available in full for this conditional task, so a deterministic
dictionary segmentation is permitted. Teacher is fit-only; output correctness
is not inferred from low NLL. Verify exact labels separately.

Use shared frozen operation critic to evaluate actual deployed dictionary paths.
Optimize per-ID costs via parameter RL,random,CEM,no-update with same initial,
teacher and equal oracle budgets; retain initial as common selection candidate.
No command/tree reference metric enters reward or candidate selection. Auxiliary
operation data is distinct from the natural formula benchmark. Natural-dataset
metrics, including speed, must be scored separately before any success claim.

At least3 actual audit rounds: IDs/duplicates/exact verification; causality,
gradient,teacher identity and serialization; independent aggregation,budgets and
frozen hashes. Preserve every seed/control and negative result. Confirmatory
success still requires new sources,multiple capacities,more seeds and strong
quality/speed across scopes.

Primary motivation: https://proceedings.mlr.press/v306/dauncey26a.html,
https://arxiv.org/abs/2507.07955, https://arxiv.org/abs/1912.01412.
