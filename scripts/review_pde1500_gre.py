"""Second-pass review of each original row and every layout component."""
from collections import Counter
import csv,gzip,json,signal
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.paper_normalization import review_row
from scripts.prepare_downstream_corpus import (check_author_atoms,invariant_support,gre_expression,words_to_ids,TARGETS)


def main():
    out=ROOT/'data/formulas/pde1500_gre_review';out.mkdir(exist_ok=True)
    rows=list(map(json.loads,gzip.open(ROOT/'data/formulas/pde/pde_equations.jsonl.gz','rt')))
    all_components=[];row_results=[]
    def timeout(*_):raise TimeoutError('row review >8 seconds')
    old=signal.signal(signal.SIGALRM,timeout)
    for count,row in enumerate(rows,1):
        try:
            signal.alarm(8);items=review_row(row)
        except TimeoutError as exc:
            items=[dict(component_index=None,status='unresolved',reason='review_timeout',detail=str(exc))]
        finally:signal.alarm(0)
        for item in items:
            item.update(parent_id=row['id'],source_group=row['source_group'],source_url=row['source_url'],title=row['title'],original_latex=row['latex'])
        all_components.extend(items)
        if count%100==0:print('REVIEW',count,'converted components',sum(r['status']=='converted' for r in all_components),flush=True)
    signal.signal(signal.SIGALRM,old)
    used={i for r in all_components if r['status']=='converted' for i in r['gre_ids'] if i>=6}
    atom_audit=check_author_atoms(used)
    targetkeys={name:invariant_support(gre_expression(words_to_ids(words))) for name,words in TARGETS.items()}
    for item in all_components:
        if item['status']=='converted':
            bad=[i for i in item['gre_ids'] if i>=6 and atom_audit[i]['status']!='PASS']
            if bad:item.update(status='unresolved',reason='author_atom_semantics_or_runtime',detail=str(bad));continue
            key=invariant_support(gre_expression(item['gre_ids']))
            item['matched_targets']=[n for n,k in targetkeys.items() if k==key]
            item['status']='target_excluded' if item['matched_targets'] else 'eligible_component'
    grouped={r['id']:[] for r in rows}
    for item in all_components:grouped[item['parent_id']].append(item)
    for row in rows:
        items=grouped[row['id']];counts=Counter(r['status'] for r in items)
        row_results.append(dict(id=row['id'],source_group=row['source_group'],source_url=row['source_url'],
           component_count=len(items),eligible_components=counts['eligible_component'],target_components=counts['target_excluded'],
           unresolved_components=counts['unresolved'],whole_expression_converted=len(items)==1 and counts['eligible_component']==1,
           status='has_eligible_component' if counts['eligible_component'] else 'target_only' if counts['target_excluded'] else 'unresolved'))
    for filename,records in [('components.jsonl',all_components),('row_review.jsonl',row_results),('eligible_components.jsonl',[r for r in all_components if r['status']=='eligible_component']),('excluded_components.jsonl',[r for r in all_components if r['status']=='target_excluded'])]:
        with (out/filename).open('w') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (out/'row_review.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row_results[0]));w.writeheader();w.writerows(row_results)
    summary=dict(original_rows=len(rows),row_status_counts=dict(Counter(r['status'] for r in row_results)),
          components=len(all_components),component_status_counts=dict(Counter(r['status'] for r in all_components)),
          whole_expression_eligible=sum(r['whole_expression_converted'] for r in row_results),
          unresolved_reasons=dict(Counter(r['reason'] for r in all_components if r['status']=='unresolved')),
          author_atom_audit=atom_audit,policy='Each parent retained; independent scalar PDE components reported separately; original systems never claimed fully represented',
          limitations=['Context-based dimension evidence is explicit and retained; not a proof of arbitrary semantic equivalence',
                       'Unresolved inputs are not proved mathematically impossible; all have an outcome and reason'])
    (out/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='author_atom_audit'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
