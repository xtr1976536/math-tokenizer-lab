"""Verify environment/documentation maintenance without changing study artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from scripts import verify_two_studies

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/repository_two_studies/maintenance'


def protected_files():
    snapshot = json.loads((OUT / 'protected_files.json').read_text())
    failures = []
    for name, expected in snapshot['files'].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
    assert not failures, failures
    return len(snapshot['files'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    count = protected_files()
    if args.round == 1:
        result = verify_two_studies.round_one()
    elif args.round == 2:
        result = verify_two_studies.round_two()
    else:
        result = verify_two_studies.round_three()
        docs = ['README.md', 'ENVIRONMENT.md', 'data/README.md', 'reports/实验方案索引.md',
                'experiments/upstream_811/README.md', 'experiments/expanded_downstream_80_v1/README.md',
                'scripts/eqgpt/README_权重.md', 'scripts/eqgpt/README_方案二模型使用.md']
        for name in docs:
            p = ROOT / name
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', p.read_text()):
                target = target.split('#', 1)[0]
                assert not target or '://' in target or (p.parent / target).exists(), (name, target)
        smoke = json.loads((OUT / 'model_smoke.json').read_text())
        assert smoke['status'] == 'PASS' and len(smoke['checks']) == 8
        result['updated_documentation_links_checked'] = len(docs)
        result['model_usage_smoke'] = 'eight Wave searched arms; CPU'
    result.update(round=args.round, unchanged_protected_files=count, interpreter=sys.version)
    (OUT / f'round_{args.round}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
