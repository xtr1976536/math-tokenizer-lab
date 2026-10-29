"""Expanded raw-LaTeX pipeline; requires two verified gates + approved settings.

Run pretraining once; all eight searches reload the same four scratch-trained
checkpoints. Existing MLP caches are mandatory and loaded with smoke=False so
no accidental refit or 32-observation cache replacement is possible.
"""
import argparse,gzip,json,hashlib
from pathlib import Path
import torch
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.expanded_corpus import sha
from math_tokenizer.eqgpt.expanded_preflight import verified_preflight,verified_settings
from math_tokenizer.eqgpt.author import verify_source,tensor_hash
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
from math_tokenizer.eqgpt.latex_tokenizers import LatexAdapter
from math_tokenizer.eqgpt.tokenizer_adapters import Adapter
from math_tokenizer.eqgpt.models import build_models
from math_tokenizer.eqgpt.round1_runtime import train_round1
from math_tokenizer.eqgpt.existing_surrogate import load_existing
from math_tokenizer.eqgpt.latex_experiment import search_latex
from math_tokenizer.eqgpt.evaluation import classify
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES
from math_tokenizer.eqgpt.experiment import dump

CORPUS=ROOT/'data/formulas/expanded_downstream_80_v1'

def adapters():
 manifest=json.loads((CORPUS/'manifest.json').read_text())
 folder=CORPUS/manifest['tokenizer_folder']
 return {'eqgpt_reference':Adapter('eqgpt'),**{
   f'ast_bpe_{s}':LatexAdapter('ast_bpe',CompactAlphabet.load(folder/f'ast_bpe_{s}.model.json')) for s in (280,315,350)}}


def protocol():
 manifest,code_hashes=verified_preflight(CORPUS)
 approved=verified_settings(CORPUS,manifest)
 return manifest,approved,code_hashes


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pretrain','search'],required=True)
 ap.add_argument('--case',choices=SHARED8_CASES,default='Wave_equation')
 ap.add_argument('--device',choices=['cpu','mps','cuda'],default='mps')
 ap.add_argument('--output',type=Path,default=ROOT/'experiments/expanded_downstream_80_v1')
 args=ap.parse_args();manifest,approved,code_hashes=protocol();verify_source();torch.set_num_threads(8)
 out=args.output;out.mkdir(parents=True,exist_ok=True)
 identity=dict(corpus=manifest['files'],approved_settings=approved,runtime_files=code_hashes,device=args.device)
 if (out/'RUN.json').exists():assert json.loads((out/'RUN.json').read_text())==identity
 else:dump(out/'RUN.json',identity)
 aa=adapters();models,metadata=build_models(aa,approved['position_capacity'],525,args.device,n_layers=3)
 if args.phase=='pretrain':
  dump(out/'shared_pretrain/initial_models.json',metadata)
  encoded={name:[] for name in aa}
  with gzip.open(CORPUS/'paired_augmented.jsonl.gz','rt') as f:
   for line in f:
    row=json.loads(line)
    for name,a in aa.items():
     seq=a.sequence(row['augmented_gre_ids'] if name=='eqgpt_reference' else row['augmented_latex'])
     assert len(seq)<=approved['position_capacity'];encoded[name].append(seq)
  for name,model in models.items():
   report=train_round1(model,encoded[name],out/'shared_pretrain'/name,args.device,525,epochs_override=approved['epochs'])
   print('PRETRAIN_COMPLETE',name,report['epochs'],report['samples'],flush=True)
  dump(out/'shared_pretrain/complete.json',dict(status='COMPLETE',epochs=approved['epochs'],samples=manifest['augmented_samples'],methods=list(aa)))
 else:
  if not (out/'shared_pretrain/complete.json').exists():raise ValueError('Shared pretraining incomplete')
  cache=ROOT/'data/mlp_cache'/args.case/'surrogate_cache.pt'
  if not cache.exists():raise FileNotFoundError('Required existing MLP cache missing; do not fit automatically')
  cache_before=sha(cache)
  env,net,database,state,shared=load_existing(cache.parent,args.case,args.device,525,10000)
  results={}
  for name,bridge in models.items():
   pre=out/'shared_pretrain'/name;training=json.loads((pre/'training.json').read_text())
   weights=torch.load(pre/'pretrained.pt',map_location=args.device,weights_only=True)
   assert tensor_hash(weights)==training['final_sha256']
   assert training['epochs']==approved['epochs'] and training['samples']==manifest['augmented_samples'] and training['batch_size']==128
   bridge.load_state_dict(weights)
   folder=out/'cases'/args.case/name;folder.mkdir(parents=True,exist_ok=True)
   search_input=dict(pretrained_sha256=training['final_sha256'],mlp_sha256=cache_before,settings=approved)
   binding=folder/'SEARCH_INPUT.json'
   if binding.exists():assert json.loads(binding.read_text())==search_input
   else:dump(binding,search_input)
   net.load_state_dict(state);net.eval();database.grad=None
   if (folder/'search.json').exists():history=json.loads((folder/'search.json').read_text())['history']
   else:history=search_latex(bridge if name=='eqgpt_reference' else bridge.core,aa[name],net,database,
        args.case,['x','t'],folder,args.device,525,attempts=200,cycles=3,finetune_epochs=2,
        max_tokens=approved['search_max_tokens'],batch_size=16,epi=.2,reference=name=='eqgpt_reference',gre_limit=approved['gre_generation_limit'],
        normalize_ast=name!='eqgpt_reference',paper_latex=name!='eqgpt_reference')
   recorded=json.loads((folder/'search.json').read_text())
   assert recorded['attempts_per_cycle']==200 and recorded['cycles']==3 and recorded['continuation_epochs']==2
   assert len(recorded['attempts'])==600 and len(history)==3
   assert recorded['max_new_tokens']==approved['search_max_tokens']
   last=history[-1];classes=[classify(seq,args.case)['status'] for seq in last['gre']]
   results[name]=dict(top1=classes[0] if classes else 'no_evaluable_candidate',top10_correct='correct' in classes,
       pretrained_sha256=training['final_sha256'],mlp_sha256=cache_before)
  assert sha(cache)==cache_before
  dump(out/'cases'/args.case/'complete.json',dict(case=args.case,results=results,mlp_reused=True))
if __name__=='__main__':main()
