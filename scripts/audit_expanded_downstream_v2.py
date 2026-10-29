"""Two independent preflight gates; formal training is a separate action.

Pass 1 proves source semantics, source/target exclusion, and tokenizer scope.
Pass 2 reopens all 72,224 samples and checks exact signed source-substring bags,
GRE factor signatures/controls, counts, hashes, byte roundtrips and lengths.
Numerical smoke uses existing fitted caches only, without calling any fitter.
"""
import argparse,gzip,json,time
from collections import Counter
from pathlib import Path
import torch
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.expanded_corpus import sha,latex_signature,gre_signature
from math_tokenizer.eqgpt.author import DICTIONARY,tensor_hash
from math_tokenizer.eqgpt.latex_codec import latex_to_structure,structure_to_latex
from math_tokenizer.eqgpt.corpus_conversion import convert,ConversionError
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
from math_tokenizer.eqgpt.latex_tokenizers import LatexAdapter
from math_tokenizer.eqgpt.tokenizer_adapters import Adapter
from math_tokenizer.eqgpt.models import build_models
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES
from math_tokenizer.eqgpt.existing_surrogate import load_existing
from math_tokenizer.eqgpt.expanded_preflight import runtime_hashes
from scripts.prepare_downstream_corpus import invariant_support,gre_expression,words_to_ids,TARGETS

OUT=ROOT/'data/formulas/expanded_downstream_v2'

def semantic_ids(text,kind):
 if kind=='eqgpt_handbook_non_target':return latex_to_structure(text)
 return convert(text)['gre_ids']


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pass-number',type=int,choices=[1,2],required=True);ap.add_argument('--corpus',default='data/formulas/expanded_downstream_v2');args=ap.parse_args()
 global OUT
 OUT=ROOT/args.corpus
 start=time.perf_counter();manifest=json.loads((OUT/'manifest.json').read_text());src=list(map(json.loads,(OUT/'sources.jsonl').read_text().splitlines()));byid={r['source_id']:r for r in src}
 targets={n:invariant_support(gre_expression(words_to_ids(w))) for n,w in TARGETS.items()}
 expected_sources=manifest['source_records'];expected_samples=manifest['augmented_samples']
 assert len(src)==expected_sources and len(byid)==expected_sources
 for p,h in manifest['input_files'].items():assert sha(ROOT/p)==h
 for p,h in manifest['files'].items():assert sha(OUT/p)==h
 tokenizer_folder=OUT/manifest.get('tokenizer_folder','tokenizers')
 models={s:CompactAlphabet.load(tokenizer_folder/f'ast_bpe_{s}.model.json') for s in (280,315,350)}
 for size,m in models.items():assert len(m.vocab)+3==size
 signatures={};g_signatures={};source_keys={};kinds=Counter();source_hashes={}
 for r in src:
  text=r['original_latex'];signatures[r['source_id']]=latex_signature(text);g_signatures[r['source_id']]=gre_signature(r['gre_ids'])
  key=invariant_support(gre_expression(r['gre_ids']));assert key not in targets.values()
  source_keys[r['source_id']]=key;kinds[r['source_kind']]+=1
  if args.pass_number==1:
   parsed=semantic_ids(text,r['source_kind']);assert invariant_support(gre_expression(parsed))==key
   for m in models.values():assert m.decode(m.encode(text))==text
  if r['source_file'] not in source_hashes:source_hashes[r['source_file']]=sha(ROOT/r['source_file'])
  assert source_hashes[r['source_file']]==r['source_sha256']
 assert dict(kinds)==manifest['source_kinds'] and kinds['eqgpt_handbook_non_target']==212
 if args.pass_number==1:
  tm=json.loads((tokenizer_folder/'manifest.json').read_text())
  import hashlib
  expected=hashlib.sha256(json.dumps([r['original_latex'] for r in src],ensure_ascii=False).encode()).hexdigest()
  assert tm['corpus_sha256']==expected
  result=dict(status='PASS',pass_number=1,source_records=len(src),source_rows_semantically_verified=len(src),target_structures_excluded=True,
     tokenizer_trained_on_original_latex_only=True,tokenizer_sizes=[280,315,350],sources_sha256=sha(OUT/'sources.jsonl'),source_hashes=source_hashes)
 else:
  first=json.loads((OUT/'audit_pass1.json').read_text());assert first['status']=='PASS' and first['sources_sha256']==sha(OUT/'sources.jsonl')
  counts=Counter();indices={k:set() for k in byid};gre_lengths=[];hard_lengths={s:[] for s in models};identical_source=0
  with gzip.open(OUT/'paired_augmented.jsonl.gz','rt') as f:
   for line in f:
    r=json.loads(line);sid=r['source_id'];source=byid[sid];ids=r['augmented_gre_ids'];text=r['augmented_latex'];i=r['augmentation_index']
    assert r['original_latex']==source['original_latex']
    assert i not in indices[sid] and 0<=i<32;indices[sid].add(i);counts[sid]+=1
    assert latex_signature(text)==signatures[sid] and gre_signature(ids)==g_signatures[sid]
    assert all(isinstance(g,int) and 0<=g<57 for g in ids)
    assert r['augmented_gre_words']==[DICTIONARY['id2word'][g] for g in ids]
    if i==0:
     assert text==source['original_latex'] and ids==source['gre_ids'];identical_source+=1
    # Reparse one nonidentity permutation per source; all other samples are
    # verified by exact signed substring/factor invariants rather than sampling.
    if i==1:
     parsed=semantic_ids(text,source['source_kind']);assert invariant_support(gre_expression(parsed))==source_keys[sid]
    gre_lengths.append(len(ids))
    for s,m in models.items():
     encoded=m.encode(text);assert m.decode(encoded)==text;hard_lengths[s].append(len(encoded)+2)
  assert sum(counts.values())==expected_samples and set(counts.values())=={32} and identical_source==expected_sources
  import numpy as np
  recomputed={'eqgpt_reference':gre_lengths,**{f'ast_bpe_{s}':v for s,v in hard_lengths.items()}}
  for n,vals in recomputed.items():
   stats=manifest['lengths'][n];assert max(vals)==stats['max'] and sum(x>50 for x in vals)==stats['over_50']
  actual_max=max(max(v) for v in recomputed.values());assert actual_max==manifest['required_common_positions']
  positions=manifest.get('training_position_capacity',actual_max)
  assert actual_max<=positions
  if positions==80:assert all(max(vals)<=80 for vals in recomputed.values())
  # Each arm receives identical initial backbone, but correct vocabulary adapter.
  adapters={'eqgpt_reference':Adapter('eqgpt'),**{f'ast_bpe_{s}':LatexAdapter('ast_bpe',m) for s,m in models.items()}}
  torch.set_num_threads(4);built,metadata=build_models(adapters,positions,525,'cpu',n_layers=3)
  smoke={}
  for n,model in built.items():
   if n=='eqgpt_reference':seq=max((x['gre_ids'] for x in src),key=len)
   else:
    a=adapters[n];seq=max((a.sequence(x['original_latex']) for x in src),key=len)
   inputs=torch.tensor([seq[:-1]],dtype=torch.long);labels=torch.tensor(seq[1:],dtype=torch.long)
   model.train();before=tensor_hash(model.state_dict());logits,_=model(inputs)
   loss=torch.nn.functional.cross_entropy(logits.reshape(-1,adapters[n].vocab_size),labels)
   assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
   # One actual optimizer step, in memory only; never writes a formal checkpoint.
   torch.optim.Adam(model.parameters(),lr=1e-4).step();assert tensor_hash(model.state_dict())!=before
   smoke[n]=dict(max_source_sequence=len(seq),loss=float(loss.detach()),input_vocab=adapters[n].vocab_size,forward_backward_step='PASS')
   model.zero_grad(set_to_none=True)
  del built
  caches={}
  for case in SHARED8_CASES:
   p=ROOT/'data/mlp_cache'/case;cache=p/'surrogate_cache.pt';before=sha(cache);quality=json.loads((p/'surrogate_quality.json').read_text());assert before==quality['cache_sha256']
   env,net,database,state,shared=load_existing(p,case,'cpu',525,10000)
   assert sha(cache)==before
   caches[case]=dict(cache_sha256=before,weights_sha256=shared['surrogate_sha256'],observations=shared['observations'],prediction_replay='PASS',fit_called=False,validation=shared['cache_validation'])
   del env,net,database,state
  result=dict(status='PASS',pass_number=2,paired_examples=expected_samples,source_records=expected_sources,each_source_exactly_32=True,
     all_samples_exact_source_latex_signed_substrings=True,all_samples_GRE_factor_signatures_and_controls=True,
     independently_reparsed_nonidentity_variants=expected_sources,all_samples_all_Hard_capacities_byte_roundtrip=True,
     no_truncation=True,required_common_positions=positions,model_initialization=metadata,model_smoke=smoke,mlp_caches=caches,
     paired_sha256=sha(OUT/'paired_augmented.jsonl.gz'),formal_training_started=False)
 result['runtime_files']=runtime_hashes()
 result['prepared_files']=manifest['files']
 result['seconds']=time.perf_counter()-start
 (OUT/f'audit_pass{args.pass_number}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
