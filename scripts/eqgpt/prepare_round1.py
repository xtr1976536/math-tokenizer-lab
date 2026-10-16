import argparse,json
from pathlib import Path
from math_tokenizer.eqgpt.frozen_math import prepare
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('experiments/eqgpt_round1/frozen_math'));a=p.parse_args()
m=prepare(a.output)
print(json.dumps(dict(corpus=m['corpus'],tokenizer_seconds=m['elapsed_tokenizer_seconds']),ensure_ascii=False,indent=2))
