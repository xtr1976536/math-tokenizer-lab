"""Data provenance/dedup/quality gates, independent of tokenizer results."""
import json,hashlib,re
from pathlib import Path
from collections import Counter
from math_tokenizer.corpus import ROOT,canonical
from math_tokenizer.corpus_archive_v5 import ensure_master
ensure_master()
from math_tokenizer.evaluate import read_jsonl,file_hash
from math_tokenizer.validation_v2 import validate
from math_tokenizer.benchmark_v5 import template_key
folder=ROOT/'data/releases/v5/curated';new=read_jsonl(folder/'new_arxiv.jsonl');source={r['id']:r for r in json.loads((ROOT/'data/releases/v5/arxiv_sources.json').read_text())}
assert len({canonical(r['latex']) for r in new})==len(new)
for r in new:
 assert validate(r['latex']) is None
 assert not re.search(r'\\\d|[\x00-\x08\x0b\x0c\x0e-\x1f]',r['latex'])
 assert r['title'] and r['authors'] and r['location'] and r['license_url']
 assert r['version_date'][:4]<'2020'
 assert source[r['source_id'].split(':',1)[1]]['license_verified_on_version_page']
# Fixed review samples from new TRAIN only, no looking at sealed test text.
train=[r for r in read_jsonl(folder/'splits/train.jsonl') if r.get('corpus_parent')=='arxiv_v5_new'];samples=sorted(train,key=lambda r:hashlib.sha256(('quality:'+r['id']).encode()).digest())[:96]
(ROOT/'results/v5_quality_train_samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in samples))
report={'automated_gates_passed':True,'new_formulas':len(new),'new_source_papers':len({r['source_group'] for r in new}),'source_records':len(source),
 'formula_only':True,'exact_dedup_new':True,'version_license_and_attribution':True,'suspicious_controls_excluded':True,
 'origin_counts':dict(Counter(r['origin'] for r in new)),'domain_counts':dict(Counter(r['domain'] for r in new)),
 'review_sample_scope':'96 hash-fixed new-training records only, test text not sampled','expert_semantic_verification_completed':False,
 'new_jsonl_sha256':file_hash(folder/'new_arxiv.jsonl')}
(ROOT/'data/releases/v5/quality_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(report)
