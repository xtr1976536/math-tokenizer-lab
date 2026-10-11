"""Verify fitted/reward IDs without opening frozen test text before selection."""
import json,sys
from collections import defaultdict
from math_tokenizer.corpus import ROOT
out=ROOT/sys.argv[1];rows=json.loads((out/'dev_ablations.json').read_text());p=json.loads((out/'protocol.json').read_text())
for scenario in ['broad','specialized']:
 permitted=set(json.loads((out/f'{scenario}_training_ids.json').read_text())['train_ids'])
 for row in rows:
  if row['scenario']!=scenario:continue
  assert set(row['fit_ids'])|set(row['reward_ids'])<=permitted
  assert set(row['fit_ids']).isdisjoint(row['reward_ids'])
print('Reward IDs are train-only for',len(rows),'dev runs')
