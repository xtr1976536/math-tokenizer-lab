# Stage 2 continuation: neural feedback and deployment-consistent learning

All work remains exploratory. Keep the original seven production systems and
benchmark immutable. Reuse pilot04 scopes as exposed development evidence, not
new independent confirmation. Fix ID capacities 1024 and 2048 before scoring.
No method is promoted without the original conjunctive intervention/quality gate.

## Neural boundary experiment (pilot05)

Train one shared causal PyTorch predictor on fit-only formula bytes: fixed
8-byte early window and patch-conditioned 4-byte completed-prefix suffix plus
8-byte current-prefix window. Separate embedding/MLP heads, 16-d embeddings,
96 hidden units. Initial training sees stochastic patches at rates .2/.4/.6.
Use 600 steps x 512 byte targets; CPU, single thread, deterministic settings.
This is a small neural pilot, not the author's Gemma/U-net reproduction.

Freeze the predictor across all optimizers. Train boundary RL, random search,
CEM and no-update for 32 rounds x 4 samples x 48 reward formulas, seeds 17/29/43.
Use prior-byte context Bernoulli policies, leave-one-out reward-to-go gradients,
the early prediction head baseline and no mathematical quality reward. Disclose
discount, budget multiplier and patch cap. Report initial, trained, stochastic
and deterministic policy outcomes separately. Do not mix a stochastic router's
loss with a dictionary decoder's speed. Check exact prefix causality, policy
gradient expectation, save/load, equal teacher identity and oracle byte budget.

Deployable dictionary paths are a separate experiment. We will measure byte-BPE
and math-atom dictionary shortest paths, learned prediction paths, and learn costs
using the actual string-to-ID encoder and fit-only token prediction-code feedback.
The mathematical atom dictionary is an explicitly **minimal-human-prior** route:
it inherits stage-1 training-derived command/codepoint units, whereas the byte
dictionary route has no mathematical lexical labels. No AST reference is passed
to any new signal learner. Do not claim low annotation equals no human priors.

## Uncertainty and evidence

All candidate/seeds retained; selection on fit/reward only. Interleave end-to-end
speed passes, retain per-formula results, independent aggregate replay, source
snapshots and three actual audit rounds. Timing includes signal/mask processing.
Compare learning against same-budget controls, report fixed-rate sensitivity.
If neural byte loss worsens or deployment rejects signal, record the failure.
Neural feedback alone is not evidence of mathematical world-model understanding.

Primary source and local reference:
- https://proceedings.mlr.press/v306/dauncey26a.html
- https://github.com/SamD770/bitter-lesson-tokenization at 8992678ece92f33eb572d6e5fd213ac7510a04c0
- Existing source snapshot located outside project in outputs/rl_mathtok_a/sources;
  its CUDA-oriented full model is not silently modified or called a reproduction.

## Deployment-consistent experiment (pilot06)

Before evaluation: run 1024/2048 capacities, seeds17/29/43, actual deployed
dictionary MAP paths. Infer paired byte identities from fit-only counts: >=32
co-occurring formulas, >=90% equal-count and nonnegative-prefix balance. Select
non-conflicting pairs by support. The stack/scoping algorithm is a disclosed
generic architecture prior; pair byte identities are learned, not fixed braces.
Compare no-pair and pair-signal models; learned scalar penalty is selected on
reward prediction code with <=1% compression regression. Pair identities must
not be chosen by observed test structural scores.

Use two alphabets: raw byte BPE has no mathematical lexical prior; the math-atom
alphabet has one inherited command/codepoint lexical prior. Protect training
primitives via a generic greedy longest primitive match; never call a TeX parser
in the new encoder. Compare unprotected paths to disclose that prior's effect.

Train one cost per vocabulary ID through parameter score-function RL, random
search, CEM, and no-update. The reward is a common frozen fit-only token Markov
code plus automatically adjusted rate multiplier; final incumbent selection uses
the same reward-only compression guard. 16 rounds x 4 candidates + common initial
=65 actual encoder evaluations per algorithm/seed. Full-path code is a coding
proxy; no raw-byte LM likelihood claim. The same native MAP decoder is used in
training and deployment, and the learned cost table is retained exactly. Treat
the stochastic-neural-router experiment separately; no mixing of speed and loss.

## Pilot07: learned statistical hierarchy

Pilot06 learned scopes improve boundary recall at1024 but mathematical-reference
crossings still regress; deployment parameter RL selects the common initial.
Before new scoring, add BPE merge trees as an additional **statistical** hierarchy,
learned from the same fit corpus. A native heap constructs that hierarchy from
the actual input; a partial-crossing cost and paired-byte boundary cost guide
dictionary DP. No math AST is read. Mathematical atom alphabets still retain
their disclosed one lexical prior. All generic parser/protection/tree work is
timed inside public encode calls. Include no-tree/no-scope controls.

Select16 (tree,scope) settings in {0,.1,.3,.6}^2 using either token code or the
shared frozen neural patch predictor, subject to1% reward compression guard.
Selection uses128 paper-balanced reward formulas; keep every candidate. Neural
features are causal given a segmentation, but ordinary dictionary segmentation
inspects the full input, so this is conditional reconstruction code, NOT an
unconditional causal LM likelihood. No structural evaluation metric enters
selection. Compare both signal objectives and both alphabets at1024/2048; all
existing internal suites remain exposed exploratory evidence.

## Pilot08: fast statistical-tree distillation

Pilot07's native heap is too costly and its scope tradeoffs are inconsistent.
Distill the training-only BPE merge-tree partial-crossing labels into per-ID
occurrence-average costs. Enumerate candidate edges on768 fit formulas; no
mathematical AST labels. Grid-select scale and paired-boundary penalty from
{0,.1,.3,.6}^2 on128 reward formulas using token code and the1% compression guard.
The runtime uses existing primitive/scope DP, with no merge tree construction.
The approximation can lose occurrence context; evaluate that loss and real speed.
Include no-cost control at both capacities and both alphabets. Do not choose
student settings using structural scores or count teacher speed as student speed.
