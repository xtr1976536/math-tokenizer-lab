"""Uniform frozen registry loaders; only paths and disclosed backend adapters."""
import gzip,hashlib,json
from .corpus import ROOT
from .current_models import load_checkpoint
from .experiment_v6 import load_entry


def load_model(entry):
    if entry['kind']=='checkpoint':return load_checkpoint(ROOT/entry['path'])
    return load_entry(entry)


def read_external():
    folder=ROOT/'data/releases/v7';seal=json.loads((folder/'seal.json').read_text())
    archive=(folder/'external_test.jsonl.gz').read_bytes()
    if hashlib.sha256(archive).hexdigest()!=seal['archive_sha256']:raise ValueError('sealed external archive mismatch')
    raw=gzip.decompress(archive)
    if hashlib.sha256(raw).hexdigest()!=seal['raw_jsonl_sha256']:raise ValueError('sealed external raw records mismatch')
    if hashlib.sha256((folder/'splits.json').read_bytes()).hexdigest()!=seal['splits_sha256']:raise ValueError('sealed IDs mismatch')
    return list(map(json.loads,raw.decode().splitlines()))


def scope(scenario,capacity=1024):
    return json.loads((ROOT/f'results/v7/{scenario}_{capacity}/fairness_protocol.json').read_text())
