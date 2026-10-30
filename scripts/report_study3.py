"""Summarize both cohorts; missing results stay missing, never zero-filled."""
import csv
import json

from math_tokenizer.eqgpt.study3_corpus import ARMS
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES
from math_tokenizer.eqgpt.study3_protocol import RUN, REPORT
from math_tokenizer.eqgpt.expanded_corpus import sha


def aggregate(rows):
    """Equation-level rates; candidates and vocabulary arms are not replicates."""
    result = []
    for cohort in ('small', 'large'):
        for arm in ARMS:
            group = [r for r in rows if r['cohort']==cohort and r['arm']==arm]
            if not group:
                continue
            n = len(group)
            available = sum(r['ever_effective_correct_scored'] for r in group)
            result.append(dict(cohort=cohort, arm=arm, evaluated_cases=n,
                expected_cases=len(SHARED8_CASES), complete=n==len(SHARED8_CASES),
                top1_hits=sum(r['top1_correct'] for r in group),
                top10_hits=sum(r['top10_correct'] for r in group),
                top1_rate=sum(r['top1_correct'] for r in group)/n,
                top10_rate=sum(r['top10_correct'] for r in group)/n,
                complete_generation_coverage=sum(r['ever_raw_complete_correct'] for r in group)/n,
                scored_correct_coverage=available/n,
                final_selection_success_given_scored_correct=(sum(r['top10_correct'] for r in group)/available if available else None),
                generated_scored_correct_but_final_miss=sum(r['generated_scored_correct_but_final_miss'] for r in group),
                candidate_valid_rate=sum(r['evaluated'] for r in group)/sum(r['attempted'] for r in group),
                first_cycle_complete_correct_occurrences=sum(r['first_cycle_raw_complete_correct'] for r in group),
                first_cycle_complete_correct_rate=sum(r['first_cycle_raw_complete_correct'] for r in group)/(n*200),
                final_top1_redundant=sum(r['final_top1_classification']=='redundant' for r in group)))
    return result


def main():
    rows, missing = [], []
    metrics = ('top1_correct', 'top10_correct', 'attempted', 'evaluated', 'raw_correct_occurrences',
               'raw_correct_complete', 'effective_correct_scored_occurrences', 'first_cycle_raw_correct',
               'first_cycle_effective_correct_scored', 'first_raw_correct', 'ever_selected_correct',
               'ever_retained_correct', 'conditional_retention_occurrences',
               'ever_raw_complete_correct', 'ever_effective_correct_scored',
               'generated_scored_correct_but_final_miss', 'first_cycle_raw_complete_correct',
               'final_top1_classification', 'search_seconds')
    for cohort in ('small', 'large'):
        for case in SHARED8_CASES:
            for arm in ARMS:
                folder = RUN/cohort/'cases'/case/arm
                if not (folder/'complete.json').exists():
                    missing.append(dict(cohort=cohort, case=case, arm=arm))
                    continue
                complete = json.loads((folder/'complete.json').read_text())
                for filename, digest in complete['files'].items():
                    if sha(folder/filename) != digest:
                        raise ValueError('Result integrity mismatch: '+str(folder/filename))
                record = json.loads((folder/'mechanism.json').read_text())
                rows.append(dict(cohort=cohort, case=case, arm=arm, **{k: record[k] for k in metrics}))
    if not rows:
        raise SystemExit('No completed formal searches. No result report generated.')
    REPORT.mkdir(exist_ok=True)
    with (REPORT/'case_mechanism.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    differences = []
    lookup = {(r['cohort'], r['case'], r['arm']): r for r in rows}
    for case in SHARED8_CASES:
        for arm in ARMS:
            small, large = lookup.get(('small', case, arm)), lookup.get(('large', case, arm))
            if small is not None and large is not None:
                differences.append(dict(case=case, arm=arm, large_minus_small={k: int(large[k])-int(small[k]) for k in
                    ('top1_correct', 'top10_correct', 'raw_correct_occurrences', 'evaluated', 'first_cycle_raw_correct')}))
    (REPORT/'comparison.json').write_text(json.dumps(dict(completed=len(rows), expected=64,
        missing=missing, rows=rows, paired_differences=differences, overall_by_arm=aggregate(rows),
        limitation='One paired seed, 8 structures, equal epochs not equal compute; Hard vocabularies relearned per cohort; added corpus changes quantity and distribution together. No general scaling or pure quantity-only causal claim. Raw-probability probes are diagnostic not causal interventions.'),
        ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
