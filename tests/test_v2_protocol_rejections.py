import importlib.util,json,shutil
from pathlib import Path
import pytest
from math_tokenizer.corpus import ROOT

spec=importlib.util.spec_from_file_location('v2audit',ROOT/'scripts/audit_v2.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

@pytest.mark.parametrize('mutation',['test_drop','bytes','roundtrip','vocab','entropy','selection_claim','data_hash'])
def test_v2_result_audit_rejects_corruption(mutation,tmp_path):
    source=ROOT/'results/v2_release512'
    if not (source/'manifest.json').exists():pytest.skip('formal512 artifact not present')
    for filename in ('protocol.json','manifest.json','selection.json','metrics.json','per_formula.jsonl'):
        shutil.copy(source/filename,tmp_path/filename)
    manifest=json.loads((tmp_path/'manifest.json').read_text());protocol=json.loads((tmp_path/'protocol.json').read_text());selection=json.loads((tmp_path/'selection.json').read_text());metrics=json.loads((tmp_path/'metrics.json').read_text())
    rows=[json.loads(x) for x in (tmp_path/'per_formula.jsonl').read_text().splitlines()]
    if mutation=='test_drop':rows.pop()
    elif mutation=='bytes':rows[0]['bytes']+=1
    elif mutation=='roundtrip':rows[0]['roundtrip']=False
    elif mutation=='vocab':metrics[0]['vocab_actual']+=99999
    elif mutation=='entropy':metrics[0]['shannon_entropy_bits_per_token']+=1
    elif mutation=='selection_claim':selection['test_used_for_selection']=True
    else:protocol['data_sha256']='wrong'
    (tmp_path/'protocol.json').write_text(json.dumps(protocol));(tmp_path/'selection.json').write_text(json.dumps(selection));(tmp_path/'metrics.json').write_text(json.dumps(metrics))
    (tmp_path/'per_formula.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    from math_tokenizer.evaluate import file_hash
    manifest['protocol_sha256']=file_hash(tmp_path/'protocol.json');manifest['selection_sha256']=file_hash(tmp_path/'selection.json');(tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):audit.result_audit(tmp_path)
