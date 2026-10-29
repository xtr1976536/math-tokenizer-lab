import json
from pathlib import Path
import pytest
from math_tokenizer.eqgpt import expanded_preflight as gates
from math_tokenizer.eqgpt.expanded_corpus import sha


def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(data))


@pytest.fixture
def bundle(tmp_path,monkeypatch):
 root=tmp_path/'repo';out=root/'corpus';out.mkdir(parents=True)
 for name,payload in [('sources.jsonl','source'),('paired_augmented.jsonl.gz','paired'),('tokenizers/model.backend.json','vocab')]:
  p=out/name;p.parent.mkdir(exist_ok=True);p.write_text(payload)
 f={name:sha(out/name) for name in ['sources.jsonl','paired_augmented.jsonl.gz','tokenizers/model.backend.json']}
 manifest=dict(files=f,input_files={},required_common_positions=80)
 write(out/'manifest.json',manifest)
 for i in (1,2):write(out/f'audit_pass{i}.json',dict(status='PASS',runtime_files={'runtime.py':'expected'},prepared_files=f,
   paired_sha256=sha(out/'paired_augmented.jsonl.gz'),mlp_caches={}))
 monkeypatch.setattr(gates,'runtime_hashes',lambda root:{'runtime.py':'expected'})
 return root,out,manifest


def test_missing_user_settings_blocks_run(bundle):
 root,out,m=bundle
 gates.verified_preflight(out,root)
 with pytest.raises(ValueError,match='pending user'):gates.verified_settings(out,m)


def test_model_payload_change_invalidates_all_old_passes(bundle):
 root,out,_=bundle;(out/'tokenizers/model.backend.json').write_text('changed vocab')
 with pytest.raises(ValueError,match='hash mismatch'):gates.verified_preflight(out,root)


def test_manifest_cannot_be_updated_to_bypass_audit(bundle):
 root,out,m=bundle;p=out/'tokenizers/model.backend.json';p.write_text('changed')
 m['files']['tokenizers/model.backend.json']=sha(p);write(out/'manifest.json',m)
 with pytest.raises(ValueError,match='changed after preflight'):gates.verified_preflight(out,root)


def test_runtime_change_invalidates_pass(bundle,monkeypatch):
 root,out,_=bundle;monkeypatch.setattr(gates,'runtime_hashes',lambda root:{'runtime.py':'changed'})
 with pytest.raises(ValueError,match='Runtime code changed'):gates.verified_preflight(out,root)


def test_pass_flag_is_required(bundle):
 root,out,_=bundle;d=json.loads((out/'audit_pass2.json').read_text());d['status']='FAIL';write(out/'audit_pass2.json',d)
 with pytest.raises(ValueError,match='did not pass'):gates.verified_preflight(out,root)


def test_setting_cannot_silently_change_search_budget(bundle):
 _,out,m=bundle
 cfg=dict(layers=3,augmentation='sampled_equivalent_32_preserve_original_latex',
 batch_size=128,lr=1e-4,clip=1,seed=525,attempts_per_cycle=200,cycles=3,continuation_epochs=2,
 search_max_tokens=80,gre_generation_limit=80,tokenizer_sizes=[280,315,350],base_bytes=98,
 mlp='reuse_only',epochs=25,position_capacity=80)
 write(out/'APPROVED_PROTOCOL.json',cfg)
 assert gates.verified_settings(out,m)==cfg
 cfg['attempts_per_cycle']=400;write(out/'APPROVED_PROTOCOL.json',cfg)
 with pytest.raises(ValueError,match='attempts_per_cycle'):gates.verified_settings(out,m)
