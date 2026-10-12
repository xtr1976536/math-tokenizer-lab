# Frozen downstream candidates and planned tasks

Two core research routes: Boundary AST and math-atom BPE. Five baselines: byte BPE, SentencePiece Unigram, Qwen2.5, Llama 3 and o200k. Standard condition: broad; trained budgets 1024/2048. Exact files and hashes are in models/registry.json. Retired models are not part of the downstream candidate pool.

Four priorities: (1) PDE solution / executable solver generation, evaluated by residual, initial/boundary error, nRMSE and execution; (2) governing-equation discovery, with equation recovery, coefficient error and held-out field prediction; (3) symbolic algebra/calculus/ODE tasks, with executable or symbolic equivalence; (4) mathematical problem solving, with verified final answers and out-of-domain/length strata. Use independent task datasets and protect task test cases from tokenizer and model adaptation training.

Start with same-capacity small-model tests using equal raw data and common training seeds. Separately study open-backbone adaptation, replacing token embeddings and the output vocabulary while retaining other weights when appropriate. Tokenizer replacement does not force complete pretraining from scratch. Initialization from token decompositions/alignment must be compared with simple initialization at equal adaptation budget.

Report non-embedding and total parameter counts, vocabulary/softmax cost, raw bytes, tokens, measured compute and training time. Mature full vocabularies are separately costed deployment references, not equal-parameter controls. A common token window covers different amounts of raw input; report truncation and raw coverage. Lossless tokenizer decoding is not a guarantee that a generated model output is valid mathematics.

No downstream task has been trained in this release. Additional mixed text/formula and numerical task data are required. The strict PDE subset is small and should not be mistaken for a numerical-solution training set.
