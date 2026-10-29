"""Content-addressed gates for the expanded experiment.

The preparation manifest records data and tokenizer payloads. Both audit
passes also record runtime source identity; changing code or data invalidates
old PASS files. This module never creates user-approved settings.
"""
import json
from pathlib import Path
from math_tokenizer.corpus import ROOT
from .expanded_corpus import sha

RUNTIME_FILES=(
 'src/math_tokenizer/eqgpt/expanded_corpus.py',
 'src/math_tokenizer/eqgpt/expanded_preflight.py',
 'src/math_tokenizer/eqgpt/existing_surrogate.py',
 'src/math_tokenizer/eqgpt/latex_experiment.py',
 'src/math_tokenizer/eqgpt/latex_codec.py',
 'src/math_tokenizer/eqgpt/corpus_conversion.py',
 'src/math_tokenizer/eqgpt/latex_tokenizers.py',
 'src/math_tokenizer/eqgpt/eqgpt_fast_config.py',
 'src/math_tokenizer/eqgpt/eqgpt_fast_tokenizers.py',
 'src/math_tokenizer/eqgpt/compact_alphabet.py',
 'src/math_tokenizer/eqgpt/author.py',
 'src/math_tokenizer/eqgpt/models.py',
 'src/math_tokenizer/eqgpt/round1_runtime.py',
 'src/math_tokenizer/eqgpt/cached_batches.py',
 'src/math_tokenizer/eqgpt/latex_generation.py',
 'src/math_tokenizer/eqgpt/tokenizer_adapters.py',
 'src/math_tokenizer/eqgpt/data.py',
 'src/math_tokenizer/eqgpt/evaluation.py',
 'src/math_tokenizer/hard_ast_2026.py',
 'native/hard_ranked_2026.c','native/frontend_v6.c',
 'scripts/prepare_expanded_downstream_v2.py',
 'scripts/prepare_expanded_downstream_80.py',
 'scripts/audit_expanded_downstream_v2.py',
 'scripts/eqgpt/expanded_fast_reward_case.py',
 'scripts/eqgpt/run_expanded_all_cases.py',
)


def runtime_hashes(root=ROOT):
 return {p:sha(Path(root)/p) for p in RUNTIME_FILES}


def verified_preflight(folder,root=ROOT):
 folder=Path(folder);root=Path(root)
 manifest=json.loads((folder/'manifest.json').read_text())
 for group,base in [('files',folder),('input_files',root)]:
  for p,expected in manifest[group].items():
   if sha(base/p)!=expected:raise ValueError('Prepared data/tokenizer hash mismatch: '+p)
 actual_runtime=runtime_hashes(root)
 for number in (1,2):
  gate=json.loads((folder/f'audit_pass{number}.json').read_text())
  if gate.get('status')!='PASS':raise ValueError('Preflight gate did not pass')
  if gate.get('runtime_files')!=actual_runtime:raise ValueError('Runtime code changed after preflight')
  if gate.get('prepared_files')!=manifest['files']:raise ValueError('Prepared files changed after preflight')
 second=json.loads((folder/'audit_pass2.json').read_text())
 if second['paired_sha256']!=sha(folder/'paired_augmented.jsonl.gz'):raise ValueError('Paired sample audit stale')
 for case,item in second['mlp_caches'].items():
  if sha(root/'data/mlp_cache'/case/'surrogate_cache.pt')!=item['cache_sha256']:
   raise ValueError('MLP cache changed: '+case)
 return manifest,actual_runtime


def verified_settings(folder,manifest):
 path=Path(folder)/'APPROVED_PROTOCOL.json'
 if not path.exists():raise ValueError('Training settings pending user answers; formal run is gated')
 cfg=json.loads(path.read_text())
 expected=dict(layers=3,augmentation='sampled_equivalent_32_preserve_original_latex',
   batch_size=128,lr=1e-4,clip=1,seed=525,attempts_per_cycle=200,cycles=3,
   continuation_epochs=2,search_max_tokens=80,gre_generation_limit=80,
   tokenizer_sizes=[280,315,350],base_bytes=98,mlp='reuse_only')
 for key,value in expected.items():
  if cfg.get(key)!=value:raise ValueError('Unregistered training/search setting: '+key)
 if cfg.get('epochs') not in (25,100):raise ValueError('Epoch setting not selected')
 if cfg.get('position_capacity')!=80 or cfg['position_capacity']<manifest['required_common_positions']:
  raise ValueError('Shared positions do not match audited 80-limit corpus')
 return cfg
