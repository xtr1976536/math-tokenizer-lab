import json

import pytest

from math_tokenizer.eqgpt import study3_protocol as gates
from math_tokenizer.eqgpt.study3_corpus import VERSION, SETTINGS
from math_tokenizer.eqgpt.expanded_corpus import sha


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    data, report = tmp_path/'data', tmp_path/'report'
    data.mkdir()
    report.mkdir()
    (data/'payload').write_text('paired corpus')
    meta = dict(version=VERSION, settings=SETTINGS, files={'payload':sha(data/'payload')}, inputs={})
    (data/'manifest.json').write_text(json.dumps(meta))
    audit = dict(code_sha256='current-code', corpus_manifest_sha256=sha(data/'manifest.json'),
                 rounds=[dict(status='PASS') for _ in range(4)])
    (report/'checks_4rounds.json').write_text(json.dumps(audit))
    monkeypatch.setattr(gates, 'DATA', data)
    monkeypatch.setattr(gates, 'REPORT', report)
    monkeypatch.setattr(gates, 'code_digest', lambda: 'current-code')
    return data, report, meta, audit


def test_four_passes_bind_to_exact_code_and_data(prepared):
    gates.verify_prepared(require_checks=True)


@pytest.mark.parametrize('change', ['code', 'manifest', 'missing_round', 'failed_round'])
def test_stale_or_failed_checks_block_execution(prepared, change):
    data, report, meta, audit = prepared
    if change=='code':
        audit['code_sha256']='old-code'
    elif change=='manifest':
        audit['corpus_manifest_sha256']='old-corpus'
    elif change=='missing_round':
        audit['rounds'].pop()
    else:
        audit['rounds'][2]['status']='FAIL'
    (report/'checks_4rounds.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='Four current passing'):
        gates.verify_prepared(require_checks=True)


def test_payload_tamper_is_rejected(prepared):
    data, *_ = prepared
    (data/'payload').write_text('different formulas')
    with pytest.raises(ValueError, match='hash mismatch'):
        gates.verify_prepared(require_checks=True)
