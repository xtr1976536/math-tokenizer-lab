"""Uniform frozen registry loaders; only paths and disclosed backend adapters."""
import gzip,hashlib,json
from .corpus import ROOT
from .current_models import load_checkpoint
from .experiment_v6 import load_entry


def load_model(entry):
    if entry['kind']=='checkpoint':return load_checkpoint(ROOT/entry['path'])
    return load_entry(entry)


def read_external(release='v7'):
    if release not in ('v7','v8'):raise ValueError('unknown sealed release')
    folder=ROOT/f'data/releases/{release}';seal=json.loads((folder/'seal.json').read_text())
    archive=(folder/'external_test.jsonl.gz').read_bytes()
    if hashlib.sha256(archive).hexdigest()!=seal['archive_sha256']:raise ValueError('sealed external archive mismatch')
    raw=gzip.decompress(archive)
    if hashlib.sha256(raw).hexdigest()!=seal['raw_jsonl_sha256']:raise ValueError('sealed external raw records mismatch')
    if hashlib.sha256((folder/'splits.json').read_bytes()).hexdigest()!=seal['splits_sha256']:raise ValueError('sealed IDs mismatch')
    return list(map(json.loads,raw.decode().splitlines()))


def evaluation_suites(release='v7'):
    external=read_external(release)
    extreme=set(json.loads((ROOT/f'data/releases/{release}/splits.json').read_text())['extreme_domain_test'])
    result={'external_test':external,'extreme_domain_test':[r for r in external if r['id'] in extreme],
            'stress_test':list(map(json.loads,(ROOT/'results/v6/stress_test.jsonl').open()))}
    if release=='v8':
        from .benchmark_v6 import load_split
        previous=read_external('v7');previous_extreme=set(json.loads((ROOT/'data/releases/v7/splits.json').read_text())['extreme_domain_test'])
        result.update(exposed_v7_test=previous,exposed_v6_test=load_split('fresh_test'),
                      exposed_v7_extreme=[r for r in previous if r['id'] in previous_extreme],
                      exposed_v6_extreme=load_split('extreme_domain_test'))
    return result


def scope(scenario,capacity=1024):
    return json.loads((ROOT/f'results/v7/{scenario}_{capacity}/fairness_protocol.json').read_text())
