"""Exactly two proposed routes and five selected baseline systems."""
import hashlib,json
from .corpus import ROOT


def load_final(method='ast_boundary',scenario='broad',capacity=1024):
    from .experiment_v7 import load_model
    registry=json.loads((ROOT/'models/registry.json').read_text())
    candidates=registry['models'].get(method,[])
    candidates=[e for e in candidates if e['scenario']=='pretrained' or (e['scenario']==scenario and e['capacity']==capacity)]
    if len(candidates)!=1:raise ValueError('Select one of the two active routes or five baselines; history is not a deployment registry')
    entry=candidates[0]
    if entry.get('path') and hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()!=entry['checkpoint_sha256']:
        raise ValueError('frozen downstream checkpoint mismatch')
    return load_model(entry)
