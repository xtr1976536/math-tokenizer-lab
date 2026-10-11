"""Re-fit the actual preceding v5 trajectory route, no old corpus weights."""
import argparse,json
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.rl_vocab_v5 import RLVocab,VocabRLConfig
from math_tokenizer.trajectory_v5 import TrajectoryVocab,TrajectoryConfig

p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');p.add_argument('--capacity',type=int,default=1024);a=p.parse_args()
out=ROOT/f'results/v6/{a.scenario}_{a.capacity}';parent=RLVocab.load(out/'models/v5_previous_local.json')
train=load_split('train' if a.scenario=='broad' else 'transfer_train')
conf=VocabRLConfig(fit_limit=1000000,reward_limit=512)
model=TrajectoryVocab(conf,TrajectoryConfig(epochs=3,samples=3,temperature=3.)).train(train,a.capacity,17,parent.theta,progress=lambda e,s,m:print('preceding v5 trajectory',a.scenario,e,s,round(m['utility'],6),flush=True))
model.train_seconds+=parent.train_seconds
model.save(out/'models/v5_previous_trajectory.json')
print('saved previous trajectory',a.scenario,model.train_seconds,flush=True)
