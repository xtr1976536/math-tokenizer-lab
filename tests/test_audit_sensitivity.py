"""Auditors must catch deliberately corrupted protocol/results, not just pass."""
import json,shutil
import pytest
from math_tokenizer.corpus import ROOT
from math_tokenizer.audit import audit_protocol,audit_results

@pytest.mark.parametrize('mutation',['missing_test_row','false_roundtrip','raw_bytes_changed','special_tokens','leak_train_test'])
def test_protocol_rejects_corruption(mutation,tmp_path):
    source=ROOT/'results/pilot01'
    for filename in ('manifest.json','per_formula.jsonl'):shutil.copy(source/filename,tmp_path/filename)
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    rows=[json.loads(x) for x in (tmp_path/'per_formula.jsonl').read_text().splitlines()]
    if mutation=='missing_test_row':rows.pop()
    elif mutation=='false_roundtrip':rows[0]['roundtrip']=False
    elif mutation=='raw_bytes_changed':rows[0]['bytes']+=1
    elif mutation=='special_tokens':manifest['special_tokens']=True
    else:manifest['train_ids'].append(manifest['test_ids'][0])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'per_formula.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(AssertionError):audit_protocol(tmp_path)

@pytest.mark.parametrize('mutation',['token_count','capacity','comparison_group'])
def test_metrics_rejects_corruption(mutation,tmp_path):
    source=ROOT/'results/pilot01'
    for filename in ('manifest.json','metrics.json','per_formula.jsonl'):shutil.copy(source/filename,tmp_path/filename)
    metrics=json.loads((tmp_path/'metrics.json').read_text())
    if mutation=='token_count':metrics[0]['tokens']+=1
    elif mutation=='capacity':metrics[0]['capacity_budget']+=1
    else:metrics[0]['comparison_group']='pretrained_reference'
    (tmp_path/'metrics.json').write_text(json.dumps(metrics))
    with pytest.raises(AssertionError):audit_results(tmp_path)
