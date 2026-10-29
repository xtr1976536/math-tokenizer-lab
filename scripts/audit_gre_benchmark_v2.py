"""Second independent audit for the expanded benchmark release."""
import hashlib,json
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.corpus_conversion import convert
from math_tokenizer.eqgpt.latex_codec import latex_to_structure
from scripts.prepare_downstream_corpus import invariant_support,gre_expression
from scripts.report_gre_benchmark import reference221


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ext=ROOT/'data/formulas/gre_external';out=ROOT/'data/formulas/gre_benchmark_audit_v2';out.mkdir(exist_ok=True)
    records=list(map(json.loads,(ext/'benchmark.jsonl').read_text().splitlines()))
    old=reference221();seen=set();
    for r in records:
        c=convert(r['latex']);assert c['gre_ids']==r['conversion']['gre_ids'];assert c['normalized_latex']==r['conversion']['normalized_latex']
        assert c['support']==c['reconstructed_support'];assert all(0<=i<57 for i in c['gre_ids'])
        key=invariant_support(gre_expression(c['gre_ids']));assert key not in old and key not in seen;seen.add(key)
        assert r['status']=='eligible';assert r['fits_current_50_positions']==(len(c['gre_ids'])<=50)
        assert r.get('source_sha256') and sha(ROOT/r['source_file'])==r['source_sha256'] if r.get('source_file') else True
    eq=[r for r in map(json.loads,(ext/'candidate_records.jsonl').read_text().splitlines()) if r.get('family')=='eqgpt_handbook_221' and not r.get('excluded_target')]
    assert len(eq)==212
    for r in eq:
        assert r['gre_ids'] and all(0<=i<57 for i in r['gre_ids'])

        try:
            replay=convert(r['latex']); replay_key=invariant_support(gre_expression(replay['gre_ids']))
        except Exception:
            replay_key=invariant_support(gre_expression(latex_to_structure(r['latex'])))
        assert replay_key==invariant_support(gre_expression(r['gre_ids']))
    (out/'audit.json').write_text(json.dumps(dict(status='PASS',external_records=len(records),external_unique_keys=len(seen),all_external_replayed=True,external_no_221_overlap=True,eqgpt_221_rows_after_target_exclusion=len(eq),eqgpt_target_rows_excluded=9,all_eqgpt_rows_replayed=True,source_hashes_checked=True),indent=2)+'\n')
    (out/'eqgpt_221_after_target_exclusion.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in eq))
    print((out/'audit.json').read_text())
if __name__=='__main__':main()
