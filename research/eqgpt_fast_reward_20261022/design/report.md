# Fast reward-based PDE discovery: all eight cases

This release contains all eight EqGPT benchmark cases. The study uses seed 525, a 3-layer decoder trained for 25 epochs, 200 attempts per cycle for three reward-search cycles, and two continuation epochs per cycle. It compares the author GRE adapter (57 IDs) with Hard and Soft AST-BPE at total vocabularies 280, 315, and 350.

The author least-squares reward, candidate ranking, Top-10 retention, and continuation training are preserved. The final PINN coefficient-fit stage is omitted; outcomes describe equation structure only. Each case has one author MLP surrogate fitted for 50,000 iterations and shared by all seven arms. All eight target structures are excluded from the shared GPT training corpus. Wave was trained from scratch; its pretrained checkpoints are reused for the other cases.

Top-1 is the reward-ranked candidate classified by the existing coefficient-free structure matcher. Attempt counts include only candidates successfully parsed and evaluated, so the correct/redundant/incorrect totals can be below 600. One seed supports screening and case-level observations, not a multi-seed success-rate claim.

| Case | Method | Total IDs | Top-1 | Top-10 exact | Top-1 extra terms | Correct / redundant / incorrect attempts | First exact attempt |
|---|---|---:|---|---|---|---:|---:|
| Wave (simple) | EqGPT GRE (57) | 57 | correct | yes | no | 4 / 60 / 536 | 130 |
| Wave (simple) | Hard AST-BPE | 280 | correct | yes | no | 1 / 48 / 242 | 113 |
| Wave (simple) | Soft AST-BPE | 280 | redundant | no | yes | 0 / 64 / 264 | - |
| Wave (simple) | Hard AST-BPE | 315 | correct | yes | no | 1 / 50 / 251 | 85 |
| Wave (simple) | Soft AST-BPE | 315 | redundant | no | yes | 0 / 52 / 234 | - |
| Wave (simple) | Hard AST-BPE | 350 | redundant | no | yes | 0 / 61 / 239 | - |
| Wave (simple) | Soft AST-BPE | 350 | redundant | no | yes | 0 / 54 / 237 | - |
| Convection-diffusion (simple) | EqGPT GRE (57) | 57 | redundant | no | yes | 0 / 9 / 591 | - |
| Convection-diffusion (simple) | Hard AST-BPE | 280 | redundant | no | yes | 0 / 10 / 293 | - |
| Convection-diffusion (simple) | Soft AST-BPE | 280 | redundant | no | yes | 0 / 10 / 312 | - |
| Convection-diffusion (simple) | Hard AST-BPE | 315 | redundant | no | yes | 0 / 15 / 288 | - |
| Convection-diffusion (simple) | Soft AST-BPE | 315 | correct | yes | no | 1 / 5 / 294 | 17 |
| Convection-diffusion (simple) | Hard AST-BPE | 350 | correct | yes | no | 4 / 13 / 297 | 108 |
| Convection-diffusion (simple) | Soft AST-BPE | 350 | redundant | no | yes | 0 / 6 / 271 | - |
| Klein-Gordon (medium) | EqGPT GRE (57) | 57 | incorrect | no | no | 0 / 10 / 590 | - |
| Klein-Gordon (medium) | Hard AST-BPE | 280 | incorrect | no | no | 0 / 7 / 280 | - |
| Klein-Gordon (medium) | Soft AST-BPE | 280 | incorrect | no | no | 0 / 5 / 319 | - |
| Klein-Gordon (medium) | Hard AST-BPE | 315 | incorrect | no | no | 0 / 14 / 286 | - |
| Klein-Gordon (medium) | Soft AST-BPE | 315 | incorrect | no | no | 0 / 14 / 268 | - |
| Klein-Gordon (medium) | Hard AST-BPE | 350 | incorrect | yes | no | 1 / 5 / 284 | 283 |
| Klein-Gordon (medium) | Soft AST-BPE | 350 | incorrect | no | no | 0 / 11 / 272 | - |
| Burgers (medium) | EqGPT GRE (57) | 57 | correct | yes | no | 1 / 9 / 590 | 497 |
| Burgers (medium) | Hard AST-BPE | 280 | redundant | no | yes | 0 / 17 / 272 | - |
| Burgers (medium) | Soft AST-BPE | 280 | redundant | no | yes | 0 / 8 / 312 | - |
| Burgers (medium) | Hard AST-BPE | 315 | redundant | no | yes | 0 / 14 / 283 | - |
| Burgers (medium) | Soft AST-BPE | 315 | redundant | no | yes | 0 / 6 / 271 | - |
| Burgers (medium) | Hard AST-BPE | 350 | correct | yes | no | 1 / 14 / 280 | 518 |
| Burgers (medium) | Soft AST-BPE | 350 | redundant | no | yes | 0 / 7 / 266 | - |
| Eq. 6.2.12 (medium) | EqGPT GRE (57) | 57 | incorrect | no | no | 0 / 0 / 600 | - |
| Eq. 6.2.12 (medium) | Hard AST-BPE | 280 | correct | yes | no | 1 / 0 / 307 | 347 |
| Eq. 6.2.12 (medium) | Soft AST-BPE | 280 | incorrect | no | no | 0 / 0 / 327 | - |
| Eq. 6.2.12 (medium) | Hard AST-BPE | 315 | incorrect | no | no | 0 / 0 / 307 | - |
| Eq. 6.2.12 (medium) | Soft AST-BPE | 315 | incorrect | no | no | 0 / 0 / 284 | - |
| Eq. 6.2.12 (medium) | Hard AST-BPE | 350 | incorrect | no | no | 0 / 0 / 310 | - |
| Eq. 6.2.12 (medium) | Soft AST-BPE | 350 | incorrect | no | no | 0 / 0 / 285 | - |
| PDE divide (medium) | EqGPT GRE (57) | 57 | incorrect | no | no | 0 / 2 / 598 | - |
| PDE divide (medium) | Hard AST-BPE | 280 | incorrect | no | no | 0 / 6 / 295 | - |
| PDE divide (medium) | Soft AST-BPE | 280 | incorrect | no | no | 0 / 7 / 310 | - |
| PDE divide (medium) | Hard AST-BPE | 315 | correct | yes | no | 1 / 5 / 293 | 501 |
| PDE divide (medium) | Soft AST-BPE | 315 | incorrect | no | no | 0 / 2 / 303 | - |
| PDE divide (medium) | Hard AST-BPE | 350 | correct | yes | no | 1 / 4 / 291 | 28 |
| PDE divide (medium) | Soft AST-BPE | 350 | incorrect | no | no | 0 / 3 / 286 | - |
| KdV (complex) | EqGPT GRE (57) | 57 | correct | yes | no | 1 / 18 / 581 | 396 |
| KdV (complex) | Hard AST-BPE | 280 | correct | yes | no | 1 / 36 / 248 | 308 |
| KdV (complex) | Soft AST-BPE | 280 | correct | yes | no | 8 / 33 / 282 | 13 |
| KdV (complex) | Hard AST-BPE | 315 | correct | yes | no | 5 / 25 / 258 | 44 |
| KdV (complex) | Soft AST-BPE | 315 | correct | yes | no | 4 / 19 / 270 | 209 |
| KdV (complex) | Hard AST-BPE | 350 | correct | yes | no | 4 / 19 / 283 | 207 |
| KdV (complex) | Soft AST-BPE | 350 | correct | yes | no | 1 / 30 / 248 | 328 |
| Chafee-Infante (complex) | EqGPT GRE (57) | 57 | redundant | no | yes | 0 / 2 / 598 | - |
| Chafee-Infante (complex) | Hard AST-BPE | 280 | incorrect | no | no | 0 / 2 / 281 | - |
| Chafee-Infante (complex) | Soft AST-BPE | 280 | incorrect | no | no | 0 / 0 / 330 | - |
| Chafee-Infante (complex) | Hard AST-BPE | 315 | incorrect | no | no | 0 / 2 / 298 | - |
| Chafee-Infante (complex) | Soft AST-BPE | 315 | redundant | no | yes | 0 / 1 / 284 | - |
| Chafee-Infante (complex) | Hard AST-BPE | 350 | incorrect | no | no | 0 / 0 / 286 | - |
| Chafee-Infante (complex) | Soft AST-BPE | 350 | incorrect | no | no | 0 / 0 / 289 | - |

## Runtime

| Case | Surrogate fit | Seven arm train + search | Surrogate validation loss |
|---|---:|---:|---:|
| Wave (simple) | 10.7 min | 32.2 min | 2.17e-06 |
| Convection-diffusion (simple) | 11.0 min | 15.5 min | 3.87e-06 |
| Klein-Gordon (medium) | 10.8 min | 18.4 min | 7.2e-07 |
| Burgers (medium) | 12.0 min | 22.9 min | 4.9e-07 |
| Eq. 6.2.12 (medium) | 12.0 min | 15.8 min | 1.79e-05 |
| PDE divide (medium) | 10.6 min | 15.7 min | 4.13e-06 |
| KdV (complex) | 6.5 min | 1.0 min | 2.57e-06 |
| Chafee-Infante (complex) | 11.1 min | 18.0 min | 1.12e-05 |

All timings use the local Apple MPS runtime. AST-BPE and EqGPT use the same 25-epoch pretraining schedule. For cases after Wave, the pretrained checkpoint was loaded from Wave; the original pretraining duration is retained in each arm summary. Search time includes reward evaluation and continuation training.

## Per-case artifacts

- `Wave_equation`: `experiments/eqgpt_fast_reward_20261022/` (complete run, logs, source sequences, tokenizer assets, surrogate data/cache, weights and search histories).
- `Convection_diffusion_equation`: `experiments/eqgpt_fast_reward_Convection_diffusion_equation_20261022/` (complete run, logs, source sequences, tokenizer assets, surrogate data/cache, weights and search histories).
- `KG_equation`: `experiments/eqgpt_fast_reward_KG_equation_20261022/` (complete run, logs, source sequences, tokenizer assets, surrogate data/cache, weights and search histories).
- `Burgers_equation`: `experiments/eqgpt_fast_reward_burgers_20261022/` (complete run, logs, source sequences, tokenizer assets, surrogate data/cache, weights and search histories).
- `Eq_6_2_12`: `experiments/eqgpt_fast_reward_Eq_6_2_12_20261022/` (complete run, logs, source sequences, tokenizer assets, surrogate data/cache, weights and search histories).
- `PDE_divide`: `experiments/eqgpt_fast_reward_PDE_divide_20261022/` (complete run, logs, source sequences, tokenizer assets, surrogate data/cache, weights and search histories).
- `KdV_equation`: `experiments/eqgpt_fast_reward_KdV_equation_20261022/` (complete run, logs, source sequences, tokenizer assets, surrogate data/cache, weights and search histories).
- `Chaffee_Infante_equation`: `experiments/eqgpt_fast_reward_chaffee_20261022/` (complete run, logs, source sequences, tokenizer assets, surrogate data/cache, weights and search histories).

The complete terminal logs are named `experiments/eqgpt_fast_reward_<case>_20261022.log`. The local original EqGPT data and source are included through the repository’s `third_party/eqgpt/` snapshot. Each case retains its own protocol, source hashes, surrogate quality, and complete JSON result. No PINN output is present.
