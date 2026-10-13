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
