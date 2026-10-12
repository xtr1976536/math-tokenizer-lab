"""Freeze benchmark suites before any experiment train/dev choice/test load."""
import json,time
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.corpus_archive_v5 import ensure_master
ensure_master()
from math_tokenizer.evaluate import file_hash
folder=ROOT/'data/releases/v5/curated';summary=json.loads((folder/'summary.json').read_text());suites=['fresh_paper_test','extreme_domain_test','synthetic_stress_test']
record={'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'corpus_sha256':summary['corpus_sha256'],'suites':{name:file_hash(folder/f'splits/{name}.jsonl') for name in suites},'suite_counts':{name:sum(1 for _ in (folder/f'splits/{name}.jsonl').open()) for name in suites},
 'test_tokens_inspected_for_model_design':False,'freeze_role':'data build/quality gates independent of model outcome; model experiment reads tests only after dev selection hash','new_papers':summary['new_papers_downloaded']}
path=ROOT/'data/releases/v5/test_freeze.json'
if path.exists():assert json.loads(path.read_text())['suites']==record['suites'],'frozen test changed'
else:path.write_text(json.dumps(record,indent=2))
print(record)
