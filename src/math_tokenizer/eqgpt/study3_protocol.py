"""Read-only protocol verification shared by CLI, audits and tests."""
import gzip
import hashlib
import json
from pathlib import Path

from math_tokenizer.corpus import ROOT
from .compact_alphabet import CompactAlphabet
from .expanded_corpus import sha
from .latex_tokenizers import LatexAdapter
from .tokenizer_adapters import Adapter
from .study3_corpus import VERSION, SETTINGS, SIZES

DATA = ROOT / 'data/formulas/study3_corpus_effect_v1'
RUN = ROOT / 'experiments/study3_corpus_effect_v1'
REPORT = ROOT / 'reports/study3_corpus_effect_v1'


def code_digest():
    files = sorted(set([*ROOT.glob('src/**/*.py'), *ROOT.glob('eqgpt_original/code/*.py'),
                       *ROOT.glob('scripts/*study3*.py'), *ROOT.glob('scripts/eqgpt/*study3*.py'),
                       *ROOT.glob('tests/*study3*.py')]))
    payload = {str(p.relative_to(ROOT)): sha(p) for p in files}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def read_rows(cohort):
    with gzip.open(DATA / cohort / 'paired.jsonl.gz', 'rt') as f:
        return [json.loads(line) for line in f]


def adapters(cohort):
    folder = DATA / cohort
    meta = json.loads((folder / 'manifest.json').read_text())
    tok = folder / meta['tokenizer_folder']
    return {'eqgpt_reference': Adapter('eqgpt'), **{
        f'ast_bpe_{s}': LatexAdapter('ast_bpe', CompactAlphabet.load(tok / f'ast_bpe_{s}.model.json')) for s in SIZES}}


def verify_prepared(require_checks=False):
    meta = json.loads((DATA / 'manifest.json').read_text())
    if meta['version'] != VERSION or meta['settings'] != SETTINGS:
        raise ValueError('Study3 configuration mismatch')
    for name, expected in meta['files'].items():
        if sha(DATA / name) != expected:
            raise ValueError('Prepared file hash mismatch: ' + name)
    for name, expected in meta['inputs'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Source corpus changed: ' + name)
    if require_checks:
        audit = json.loads((REPORT / 'checks_4rounds.json').read_text())
        if (audit.get('code_sha256') != code_digest() or audit.get('corpus_manifest_sha256') != sha(DATA / 'manifest.json')
                or len(audit.get('rounds', [])) != 4 or any(r['status'] != 'PASS' for r in audit['rounds'])):
            raise ValueError('Four current passing checks required before execution')
    return meta


def safe_run_path(path):
    path = Path(path).resolve()
    if path != RUN.resolve():
        raise ValueError('Study3 writes only to its independent experiment directory')
    return path
