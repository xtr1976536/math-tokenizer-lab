"""Hash-checked v6 master and ID-only splits, without automatic test loading."""
import gzip
import hashlib
import json
from .corpus import ROOT

RELEASE = ROOT/'data/releases/v6'


def master_path():
    path = RELEASE/'curated/corpus.jsonl'
    expected = json.loads((RELEASE/'manifest.json').read_text())['corpus_sha256']
    if not path.exists():path.write_bytes(gzip.decompress(path.with_suffix('.jsonl.gz').read_bytes()))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('v6 master hash mismatch')
    return path


def load_split(name):
    splits = json.loads((RELEASE/'splits.json').read_text())
    ids = set(splits[name])
    rows = [r for r in map(json.loads,master_path().open()) if r['id'] in ids]
    if len(rows)!=len(ids):raise ValueError('split contains missing/duplicate IDs')
    return rows
