"""Train-only independent tree rewards and dev-only configuration selection."""
import argparse,json,time
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.rl_vocab_v5 import VocabRLConfig
from math_tokenizer.rl_route_v6 import AtomicVocab
from math_tokenizer.rl_structural_v6 import train_structural
from train_routes_v6 import score,fixed_dev,write

p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');p.add_argument('--capacity',type=int,default=1024);a=p.parse_args();out=ROOT/f'results/v6/{a.scenario}_{a.capacity}'
train=load_split('train' if a.scenario=='broad' else 'transfer_train');dev=fixed_dev(load_split('dev' if a.scenario=='broad' else 'transfer_dev'));conf=VocabRLConfig(fit_limit=1000000,reward_limit=512)
results=[];pool={}
write(out/'structural_search_protocol.json',{'tree_weights':[.25,.5,1.],'estimator':'exact expected contextual bandit policy gradient','test_used':False})
for weight in (.25,.5,1.):
    model=train_structural(train,a.capacity,conf,17,weight);path=out/f'models/v5_structural_{weight}.json';model.save(path);pool[weight]=model
    stats=score(model,dev,'structural_'+str(weight));results.append({'tree_weight':weight,'score':stats,'train_seconds':model.train_seconds,'policy':model.theta.tolist()});write(out/'structural_dev.json',results);print(a.scenario,a.capacity,results[-1],flush=True)
chosen=min(results,key=lambda r:r['score']['utility']);model=pool[chosen['tree_weight']]
for seed in (17,29,43):model.seed=seed;model.save(out/f'models/v5_structural_selected_{seed}.json')
for method in ('frequency','reward_greedy'):
    control=train_structural(train,a.capacity,conf,17,chosen['tree_weight'],method);control.save(out/f'models/v5_structural_{method}.json')
write(out/'structural_selection.json',{'selected':chosen,'deterministic_expected_gradient':True,'seeds_identical_by_design':True,'test_used':False,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
