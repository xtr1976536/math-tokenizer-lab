# Fast reward based PDE discovery experiment

This private release records the complete compact downstream comparison for
the eight shared EqGPT benchmark targets. It compares the author's 57-ID GRE
interface with Hard AST-BPE and Soft AST-BPE at total vocabularies 280, 315,
and 350.

## Frozen protocol

- Cases: Wave, convection-diffusion, Klein-Gordon, Burgers, Eq. 6.2.12, PDE divide, KdV, and Chafee-Infante.
- One seed: 525.
- Decoder: 3 layers, 25 epochs; all three method families use the same model architecture and training sequences.
- Each case uses 200 attempts per cycle for three reward-search cycles, with two continuation epochs per cycle.
- The author's least-squares reward, candidate ranking, Top-10 retention, and continuation training remain active.
- PINN coefficient fitting is omitted. Reported outcomes classify equation structure only; no coefficient accuracy is claimed.
- One 50,000-iteration author MLP surrogate is fitted for each case and shared by all seven tokenizer arms within that case. The held-out target structures are excluded from GPT training.
- Wave is trained from scratch. Its pretrained checkpoints are reused for the other seven cases. Search models are separately trained per case.
- The initial character alphabet is 98 ASCII bytes; control tokens count toward the requested AST-BPE vocabulary size.
- Runtime: Apple MPS, PyTorch version recorded in the case runtime manifests.

## Contents

`cases/<case>/` contains each complete run directory: source GRE dataset and manifest, tokenizer assets, surrogate observations and cache, per-arm training checkpoints, reward search attempts and histories, timings, and aggregate results. `logs/` contains the exact terminal logs. `design/` contains the frozen protocol, summary report, and analysis script.

PyTorch checkpoint files use Git LFS. Other artifacts remain ordinary Git files. The source implementation and upstream EqGPT snapshot are in the parent repository. The exact local source/data hashes are retained in each run's protocol and manifests.

## Reproduction

From the repository root, install the pinned project environment and run one case with:

```bash
PYTHONPATH=src python scripts/eqgpt/fast_reward_case.py \
  --case Wave_equation --device mps \
  --output research/eqgpt_fast_reward_20261022/reproduction/Wave_equation
```

For a later case, pass `--shared-pretrain` pointing to a completed Wave case directory. Its tokenizer directory and source GRE corpus must be the ones recorded in this release. The checked-in results are the authoritative run; reproduction creates a separate output directory.

## Reading results

Each `complete.json` reports the reward-ranked Top-1 structure, whether any of the final Top-10 structures is exact, whether the Top-10 includes a redundant structure, counts among successfully evaluated generated candidates, and the first exact candidate's attempt index. Attempts that fail parsing, variable availability checks, or numerical evaluation remain visible in `search.json`. All counts are for one random seed and are screening evidence, not estimates of multi-seed success rates.
