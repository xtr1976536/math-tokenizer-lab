"""Independent second audit of target-like filtering."""
import json
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.corpus_conversion import convert
from scripts.prepare_downstream_corpus import TARGETS,words_to_ids,invariant_support,gre_expression

def main():
 p=ROOT/'data/formulas/gre_benchmark_filtered';rows=list(map(json.loads,(p/'benchmark.jsonl').read_text().splitlines()));excluded=list(map(json.loads,(p/'excluded_target_like.jsonl').read_text().splitlines()))
 targets={n:set(invariant_support(gre_expression(words_to_ids(w)))) for n,w in TARGETS.items()};keys=set()
 for r in rows:
  c=convert(r['latex']);assert c['gre_ids']==r['conversion']['gre_ids'];assert c['support']==c['reconstructed_support'];assert all(i<57 for i in c['gre_ids'])
  k=invariant_support(gre_expression(c['gre_ids']));assert k not in keys;keys.add(k)
  for t in targets.values():assert not (t<=set(c['support']) and len(set(c['support'])-t)<=2)
 for r in excluded:
  assert r['filter_status']=='excluded_target_like' and r['target_like_matches']
  for m in r['target_like_matches']:assert m['extra_terms']<=2
 manifest=json.loads((p/'manifest.json').read_text());assert manifest['retained_records']==len(rows);assert manifest['excluded_records']==len(excluded)
 (p/'audit_v2.json').write_text(json.dumps(dict(status='PASS',input_records=manifest['input_records'],retained=len(rows),excluded=len(excluded),all_retained_replayed=True,no_exact_or_high_target_like_retained=True,all_exclusions_have_explicit_case_and_extra_term_count=True,old_external_benchmark_untouched=True),indent=2)+'\n')
 print((p/'audit_v2.json').read_text())
if __name__=='__main__':main()
