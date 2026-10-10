import json
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.corpus_v2 import build
from math_tokenizer.evaluate import file_hash

before=file_hash(ROOT/'data/releases/v2/curated/corpus.jsonl')
summary=build()
after=file_hash(ROOT/'data/releases/v2/curated/corpus.jsonl')
assert before==after,'v2 deterministic rebuild changed formula data'
(ROOT/'results/v2_rebuild_verification.json').write_text(json.dumps({'passed':True,'corpus_sha256':after,'count':summary['count'],'source_papers':summary['arxiv_papers_downloaded'],'raw_source_replay':False,'note':'Re-extracted existing pinned local source archives; no fresh network downloads.'},indent=2))
print('Deterministic rebuild verified',after)
