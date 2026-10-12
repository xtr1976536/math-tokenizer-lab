import json
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.validation_v2 import validate
from math_tokenizer.structure_v2 import complexity
from math_tokenizer.evaluate import read_jsonl


def test_v2_snapshot_and_complexity():
    rows=read_jsonl(ROOT/'data/releases/v2/curated/corpus.jsonl')
    assert len(rows)>15000
    origins={r['origin'] for r in rows};assert 'human' in origins
    assert {r['complexity']['class'] for r in rows}=={'simple','complex'}
    seen={}
    for r in rows:
        assert validate(r['latex']) is None
        seen.setdefault(r['source_group'],set()).add(r['split'])
    assert all(len(v)==1 for v in seen.values())
    assert sum(r['origin']=='human' and r['domain']!='pde' for r in rows)>5000


def test_arxiv_versions_and_license():
    sources=json.loads((ROOT/'data/arxiv_sources.json').read_text())
    assert len(sources)>=60
    for s in sources:
        assert int(s['version_date'][:4])<2020
        assert s['license'].startswith(('CC-BY-','CC0-'))
        assert s['license_verified_on_version_page'] is True
        assert 'v' in s['version']


def test_old_release_untouched():
    import hashlib
    a=ROOT/'data/releases/v1/curated/corpus.jsonl';b=ROOT/'data/curated/corpus.jsonl'
    assert hashlib.sha256(a.read_bytes()).digest()==hashlib.sha256(b.read_bytes()).digest()
