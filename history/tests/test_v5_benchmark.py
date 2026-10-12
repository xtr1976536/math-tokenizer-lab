import json
from pathlib import Path
from math_tokenizer.corpus import ROOT,canonical
from math_tokenizer.evaluate import read_jsonl,file_hash
from math_tokenizer.benchmark_v5 import template_key,TARGET_CATEGORIES


def test_old_release_unchanged():
 summary=json.loads((ROOT/'data/releases/v2/curated/summary.json').read_text())
 assert file_hash(ROOT/'data/releases/v2/curated/corpus.jsonl')==summary['corpus_sha256']


def test_new_test_frozen_and_fit_disjoint_if_built():
 folder=ROOT/'data/releases/v5/curated/splits'
 if not (folder/'fresh_paper_test.jsonl').exists():return
 train=read_jsonl(folder/'train.jsonl');dev=read_jsonl(folder/'dev.jsonl');test=read_jsonl(folder/'fresh_paper_test.jsonl')
 assert {r['split_group'] for r in train+dev}.isdisjoint(r['split_group'] for r in test)
 exact={canonical(r['latex']) for r in train+dev};templates={template_key(r['latex']) for r in train+dev}
 assert all(canonical(r['latex']) not in exact and template_key(r['latex']) not in templates for r in test)
 assert all(r['source_id'].startswith('arxiv:') for r in test)


def test_specialized_source_domain_exclusion_if_built():
 folder=ROOT/'data/releases/v5/curated/splits'
 if not (folder/'transfer_train.jsonl').exists():return
 source={'arxiv:'+s['id']:set(s['categories']) for path in [ROOT/'data/arxiv_sources.json',ROOT/'data/releases/v5/arxiv_sources.json'] for s in json.loads(path.read_text())}
 for r in read_jsonl(folder/'transfer_train.jsonl')+read_jsonl(folder/'transfer_dev.jsonl'):
  assert r['domain'] in {'pde','ode','analysis'}
  assert not source.get(r['source_group'],set())&TARGET_CATEGORIES


def test_stress_test_not_natural_or_fit_if_built():
 folder=ROOT/'data/releases/v5/curated/splits'
 if not (folder/'synthetic_stress_test.jsonl').exists():return
 stress=read_jsonl(folder/'synthetic_stress_test.jsonl');fit=read_jsonl(folder/'train.jsonl')+read_jsonl(folder/'dev.jsonl')
 assert {r['id'] for r in stress}.isdisjoint(r['id'] for r in fit)
 assert all(r['origin']=='ai_generated' and r['stress'] for r in stress)
 assert any(r['stress']=='long_formula_terms_2048' for r in stress)
 assert any(r['stress']=='unknown_macro_unicode' for r in stress)
