"""Read the classified catalog while preserving frozen ID-only partitions."""
import gzip,hashlib,json
from .corpus import ROOT

FOLDER=ROOT/'data/benchmark'


def read_catalog():
    manifest=json.loads((FOLDER/'manifest.json').read_text());path=FOLDER/'metadata/formulas.jsonl.gz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['metadata_sha256']:raise ValueError('Catalog integrity failure')
    with gzip.open(path,'rt') as stream:rows=list(map(json.loads,stream))
    if len(rows)!=manifest['formulas'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Catalog ID/count failure')
    return rows


def suites():
    rows=read_catalog();split=json.loads((FOLDER/'splits.json').read_text())
    definitions={'external_test':'v8_external_test','extreme_domain_test':'v8_extreme_domain_test',
                 'exposed_v7_test':'v7_external_test','exposed_v7_extreme':'v7_extreme_domain_test',
                 'exposed_v6_test':'v6_fresh_test','exposed_v6_extreme':'v6_extreme_domain_test'}
    idsets={name:set(split[key]) for name,key in definitions.items()}
    result={name:[r for r in rows if r['id'] in ids] for name,ids in idsets.items()}
    with gzip.open(FOLDER/'stress/formulas.jsonl.gz','rt') as stream:result['stress_test']=list(map(json.loads,stream))
    return result
