"""Dev-only v5 RL barrier alternatives; freeze every choice before test scoring."""
import argparse,json,time
from dataclasses import asdict
import numpy as np
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.corpus import ROOT
from math_tokenizer.rl_mask_v6 import BarrierPolicyV5,MaskConfig,FrozenBarrierV5
from math_tokenizer.rl_vocab_v5 import VocabRLConfig
from train_routes_v6 import score,fixed_dev,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');p.add_argument('--capacity',type=int,default=1024);a=p.parse_args()
    out=ROOT/f'results/v6/{a.scenario}_{a.capacity}';models=out/'models';conf=VocabRLConfig(fit_limit=1000000,reward_limit=512)
    train=load_split('train' if a.scenario=='broad' else 'transfer_train');dev=fixed_dev(load_split('dev' if a.scenario=='broad' else 'transfer_dev'))
    options={'barrier_tree025':MaskConfig(epochs=10,samples=4,tree_weight=.25),'barrier_tree050':MaskConfig(epochs=10,samples=4,tree_weight=.5)}
    write(out/'barrier_search_protocol.json',{'options':{k:asdict(v) for k,v in options.items()},'selection_seeds':[17,29],'test_used':False})
    result=[];pool={}
    for name,mc in options.items():
        for seed in (17,29):
            path=models/f'{name}_{seed}.json'
            if path.exists():model=FrozenBarrierV5.load(path)
            else:
                trainer=BarrierPolicyV5(conf,mc);model=trainer.train(train,a.capacity,seed,progress=lambda e,s,m:print(a.scenario,a.capacity,name,seed,e,s,round(m['utility'],5),flush=True));model.save(path)
                trainer.frequency.save(models/'v5_barrier_frequency.json')
            pool[(name,seed)]=model
            result.append({'name':name,'seed':seed,'score':score(model,dev,name),'train_seconds':model.train_seconds,'mask':model.mask,'policy_change_norm':float(np.linalg.norm(np.array(model.policy)-np.array(model.initial_policy))),'initial_terminal':model.initial_terminal,'final_terminal':model.final_terminal})
            write(out/'barrier_dev.json',result);print('barrier dev',a.scenario,a.capacity,result[-1],flush=True)
    means={name:float(np.mean([r['score']['utility'] for r in result if r['name']==name])) for name in options};chosen=min(means,key=means.get)
    write(out/'barrier_selection.json',{'selected':chosen,'config':asdict(options[chosen]),'mean_dev_utility':means,'test_used':False,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    for seed in (17,29):pool[(chosen,seed)].save(models/f'v5_barrier_selected_{seed}.json')
    model=BarrierPolicyV5(conf,options[chosen]).train(train,a.capacity,43);model.save(models/'v5_barrier_selected_43.json')
    trainer=BarrierPolicyV5(conf,options[chosen]);fixed=trainer.train(train,a.capacity,17,update=False);fixed.save(models/'v5_barrier_no_update.json');trainer.best_sample.save(models/'v5_barrier_search_control.json')
    write(out/'barrier_ready.json',{'capacity':a.capacity,'scenario':a.scenario,'chosen':chosen,'test_used':False})

if __name__=='__main__':main()
