import json
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl

def main():
    target=ROOT/'data/formula_only'
    for source in (ROOT/'data/curated').rglob('*.jsonl'):
        relative=source.relative_to(ROOT/'data/curated');dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        rows=read_jsonl(source)
        # JSON string per line: only LaTeX, no natural language or metadata.
        dest.write_text(''.join(json.dumps(r['latex'],ensure_ascii=False)+'\n' for r in rows))
    print('Formula-only JSON string files exported')
if __name__=='__main__':main()
