"""Frozen tokenizer selection for downstream work, no training on load."""
import hashlib,json
from .corpus import ROOT
from .experiment_v7 import load_model


def load_final(method='ast_boundary',scenario='broad',capacity=1024,include_diagnostics=False):
    registry=json.loads((ROOT/'models/final_intrinsic/registry.json').read_text())
    if method in registry['shortlist']['diagnostic_only'] and not include_diagnostics:
        raise ValueError('diagnostic checkpoint excluded from production; opt in explicitly for ablation replay')
    candidates=registry['models'].get(method,[])
    candidates=[e for e in candidates if e['scenario']=='pretrained' or (e['scenario']==scenario and e['capacity']==capacity)]
    if len(candidates)!=1:raise ValueError('method/scenario/capacity not uniquely registered for downstream')
    entry=candidates[0]
    if entry.get('path') and hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()!=entry['checkpoint_sha256']:
        raise ValueError('frozen downstream checkpoint mismatch')
    return load_model(entry)
