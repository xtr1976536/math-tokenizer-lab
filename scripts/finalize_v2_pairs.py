import json,sys
from collections import defaultdict
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl,bootstrap_difference
out=ROOT/sys.argv[1];rows=read_jsonl(out/'per_formula.jsonl');by=defaultdict(list)
for row in rows:by[(row['method'],row['seed'])].append(row)
paired=json.loads((out/'pairwise.json').read_text())
for seed in (17,29,43):
 for method,control in [('rl_bpe_v2','byte_policy_no_rl'),('rl_ast_bpe_v2','ast_policy_no_rl'),('rl_bpe_v2','byte_shortest_path'),('rl_ast_bpe_v2','ast_shortest_path')]:
  paired[f'{method}-vs-{control}/{seed}']=bootstrap_difference(by[(method,seed)],by[(control,0)])
(out/'pairwise.json').write_text(json.dumps(paired,indent=2))
from math_tokenizer.report_v2 import report
report(out)
print('Added explicit no-RL and shortest-path paired comparisons')
