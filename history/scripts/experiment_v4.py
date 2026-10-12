"""RL core maintained: masks+exact expected policy gradient, frozen corpus."""
import argparse,json,time
from dataclasses import asdict,replace
from collections import Counter,defaultdict
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl,file_hash,bootstrap_difference
from math_tokenizer.atoms_v4 import GrammarBPE
from math_tokenizer.rl_v4 import MaskedConfig,MaskedPolicy
from math_tokenizer.constrained_v4 import ConstrainedTokenizer
from math_tokenizer.native_v3 import NativeTokenizer
from math_tokenizer.metrics_v2 import measure
from math_tokenizer.structure_v3 import structure_fast

CONFIGS={
 'expected12':MaskedConfig(epochs=12),
 'expected24':MaskedConfig(epochs=24),
 'expected24_entropy':MaskedConfig(epochs=24,entropy=.02),
 'expected24_rare':MaskedConfig(epochs=24,rare_cost=.1),
 'expected24_prior':MaskedConfig(epochs=24,initial_length_prior=1.5),
}

def devstats(model,records):
 total=bytescount=0;groups=defaultdict(lambda:[0,0]);commands=intact=0
 for r in records:
  t=r['latex'];ids=model.encode(t);groups[r['split_group']][0]+=len(ids);groups[r['split_group']][1]+=len(t.encode());total+=len(ids)
  offset=0;boundaries={0}
  for id in ids:offset+=len(model.vocab[id]);boundaries.add(offset)
  for a,b,s in structure_fast(t).lexemes:
   if s.startswith('\\') and s[1:].isalpha():commands+=1;intact+=not any(a<x<b for x in boundaries)
 return {'macro_tokens_per_byte':float(np.mean([a/b for a,b in groups.values()])), 'tokens_per_formula':total/len(records),'command_integrity':intact/max(1,commands)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--capacity',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
 if (out/'protocol.json').exists():raise RuntimeError('immutable output')
 data=ROOT/'data/releases/v2/curated';train=read_jsonl(data/'splits/train.jsonl');dev=read_jsonl(data/'splits/dev.jsonl');test=read_jsonl(data/'splits/test.jsonl');texts=[r['latex'] for r in train]
 protocol={'version':'v4_masked_RL','capacity':a.capacity,'corpus_sha256':file_hash(data/'corpus.jsonl'),'train_ids':[r['id'] for r in train],'dev_ids':[r['id'] for r in dev],'test_ids':[r['id'] for r in test],
  'atom_budgets':[128,192,256],'configs':{k:asdict(v) for k,v in CONFIGS.items()},'dev_seeds':[17,29],'formal_seeds':[17,29,43],
  'constraints':['no_partial_tree_crossing_when_parsed','no_boundary_inside_vocab_covered_commands','no_boundary_inside_vocab_covered_utf8_codepoint'],
  'selection':'dev command integrity>=.999 first if feasible; then macro tokens/byte; choose independently of test; hard tree all candidates',
  'unknown_atoms':'explicit byte escape, no guarantee that arbitrary unseen commands/numbers have one fixed token','rl_estimator':'exact expectation of on-policy score-function on finite DAG, no teacher or shortest-path targets',
  'new_baselines_run':False,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));models=out/'models';models.mkdir(exist_ok=True);bases={};trained={};rows=[]
 for atoms in protocol['atom_budgets']:
  base=GrammarBPE(True).train(texts,a.capacity,atom_budget=atoms);bases[atoms]=base
  shortest=ConstrainedTokenizer(base.vocab,True,'commands','shortest');shortest.capacity=a.capacity;shortest.train_seconds=base.train_seconds;shortest.training_log=base.training_log;shortest.save(models/f'ast-atom{atoms}.json')
  stats=devstats(shortest,dev);rows.append({'arm':'ast','atoms':atoms,'config':'shortest','seed':0,'stats':stats});print('dev AST',atoms,stats,flush=True)
  for seed in (17,29):
   for name,config in CONFIGS.items():
    learned=MaskedPolicy(base.vocab,config).train(texts,seed);learned.capacity=a.capacity;learned.vocab_train_seconds=base.train_seconds
    for mode in ('greedy','policy_map'):
     model=ConstrainedTokenizer(learned.vocab,True,'commands',mode,learned.theta,learned.frequency);model.capacity=a.capacity;model.train_seconds=learned.train_seconds;model.training_log=base.training_log+learned.training_log
     model.save(models/f'rl-atom{atoms}-{name}-{mode}-{seed}.json');trained[(atoms,name,mode,seed)]=model;stats=devstats(model,dev)
     row={'arm':'rl','atoms':atoms,'config':name,'mode':mode,'seed':seed,'stats':stats,'policy_train_seconds':learned.train_seconds};rows.append(row);print('dev RL',atoms,name,mode,seed,stats,flush=True)
     (out/'dev_ablations.json').write_text(json.dumps(rows,indent=2))
 # Hard constraints define action legality; no hand-written merge decisions.
 ast=[r for r in rows if r['arm']=='ast'];eligible=[r for r in ast if r['stats']['command_integrity']>=.999] or ast
 chosen_ast=min(eligible,key=lambda r:(r['stats']['macro_tokens_per_byte'],r['atoms']))
 groups=defaultdict(list)
 for r in rows:
  if r['arm']=='rl':groups[(r['atoms'],r['config'],r['mode'])].append(r)
 means={key:float(np.mean([r['stats']['macro_tokens_per_byte'] for r in group])) for key,group in groups.items()}
 eligible={key:val for key,val in means.items() if np.mean([r['stats']['command_integrity'] for r in groups[key]])>=.999} or means
 chosen_rl=min(eligible,key=lambda key:(eligible[key],key))
 selected={'ast_atoms':chosen_ast['atoms'],'rl_atoms':chosen_rl[0],'rl_config':chosen_rl[1],'rl_mode':chosen_rl[2],'dev_scores':{'ast':chosen_ast['stats'],'rl_macro_tokens_per_byte':means[chosen_rl]},'test_used_for_selection':False}
 (out/'selection.json').write_text(json.dumps(selected,indent=2));print('selected',selected,flush=True)
 summaries=[];allrows=[];by={}
 def evaluate(model,name,seed=0):
  frequency=Counter()
  for text in texts:frequency.update(model.encode(text))
  summary,rr=measure(model,test,name,seed,repeats=5,train_frequency=frequency);summary['capacity_budget']=a.capacity;summary['comparison_group']='same_frozen_train_test'
  if isinstance(model,ConstrainedTokenizer):
   exceptions=Counter();formulas=0
   for r in test:
    missing=model.prepare(r['latex'])[5];formulas+=bool(missing)
    for x,y,reason in missing:exceptions[reason]+=1
   summary['atomic_escape_counts']=dict(exceptions);summary['atomic_escape_formulas']=formulas
   logs=[l for l in model.training_log if 'epochs' in l]
   if logs:summary['policy_initial_expected_tokens']=logs[-1]['epochs'][0]['train_expected_tokens'];summary['policy_final_expected_tokens']=logs[-1]['epochs'][-1]['train_expected_tokens']
  summaries.append(summary);allrows.extend(rr);by[(name,seed)]=rr
  (out/'metrics.json').write_text(json.dumps(summaries,indent=2));(out/'per_formula.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in allrows));print('test',name,seed,'tokens',round(summary['tokens_per_formula'],4),'cmd',summary['command_integrity_rate'],'tree',summary['hybrid_tree_crossing_rate'],'speed',round(summary['encode_mib_s'],3),flush=True)
 astmodel=ConstrainedTokenizer.load(models/f'ast-atom{chosen_ast["atoms"]}.json');evaluate(astmodel,'ast_v4')
 oldselection=json.loads((ROOT/f'results/v3_{a.capacity}_release/selection.json').read_text());oldast=NativeTokenizer.load(ROOT/f'results/v3_{a.capacity}_release/models/ast-{oldselection["ast"]}.json');evaluate(oldast,'ast_v3')
 atoms,name,mode=chosen_rl;base=bases[atoms]
 prior=ConstrainedTokenizer(base.vocab,True,'commands',mode,theta=np.zeros(7));prior.save(models/'no_rl_zero.json');evaluate(prior,'masked_zero_policy')
 reference=ConstrainedTokenizer(base.vocab,True,'commands','shortest');reference.save(models/'no_rl_shortest.json');evaluate(reference,'masked_shortest_control')
 # Mask-only control directly applies constraints to previous policy/checkpoint.
 for seed in (17,29,43):
  old=NativeTokenizer.load(ROOT/f'results/v3_{a.capacity}_release/models/rl_bpe_v3-{seed}.json');evaluate(old,'rl_v3',seed)
  maskonly=ConstrainedTokenizer(old.vocab,True,'commands','policy_map',old.theta,old.frequency);maskonly.save(models/f'mask_only-{seed}.json');evaluate(maskonly,'v3_mask_only_control',seed)
  model=trained.get((atoms,name,mode,seed))
  if model is None:
   learned=MaskedPolicy(base.vocab,CONFIGS[name]).train(texts,seed);model=ConstrainedTokenizer(learned.vocab,True,'commands',mode,learned.theta,learned.frequency);model.train_seconds=learned.train_seconds;model.training_log=base.training_log+learned.training_log
  model.capacity=a.capacity;model.save(models/f'rl_v4-{seed}.json');evaluate(model,'rl_v4',seed)
 paired={}
 for seed in (17,29,43):
  for control,cseed in [('rl_v3',seed),('masked_zero_policy',0),('masked_shortest_control',0),('v3_mask_only_control',seed)]:paired[f'rl_v4-vs-{control}/{seed}']=bootstrap_difference(by[('rl_v4',seed)],by[(control,cseed)])
 paired['ast_v4-vs-v3']=bootstrap_difference(by[('ast_v4',0)],by[('ast_v3',0)])
 (out/'pairwise.json').write_text(json.dumps(paired,indent=2));assert file_hash(data/'corpus.jsonl')==protocol['corpus_sha256']
 (out/'manifest.json').write_text(json.dumps({'corpus_sha256':protocol['corpus_sha256'],'protocol_sha256':file_hash(out/'protocol.json'),'selection_sha256':file_hash(out/'selection.json'),'native_source_sha256':file_hash(ROOT/'native/decoder_v4.c'),'models':{p.name:file_hash(p) for p in models.glob('*.json')},'runs':len(summaries),'records':len(allrows)},indent=2))
if __name__=='__main__':main()
