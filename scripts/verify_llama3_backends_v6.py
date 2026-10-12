"""Validate Meta's unchanged tiktoken implementation against official HF file.

Keep both official implementation measurements visible, one main tokenizer
entry to avoid giving identical segmentation two votes in practical rankings.
"""
from datetime import datetime, timezone
import json
from tokenizers import Tokenizer
from math_tokenizer.baselines import HFTokenizer
from math_tokenizer.baselines_v6 import Llama3Official, prepare_llama3_ranks
from math_tokenizer.benchmark_v6 import master_path
from math_tokenizer.corpus import ROOT


def main():
    folder=ROOT/'results/v6/full'
    record=next(r for r in json.loads((ROOT/'data/pretrained_registry_v6.json').read_text()) if r['repo']=='meta-llama/Meta-Llama-3-8B')
    prepare_llama3_ranks(record)
    hf=HFTokenizer(Tokenizer.from_file(str(ROOT/record['local'])));reference=Llama3Official(record)
    n=0;stress=0
    for row in map(json.loads,master_path().open()):
        text=row['latex'];ids=hf.encode(text)
        assert ids==reference.encode(text) and reference.decode(ids)==text
        n+=1
    for row in map(json.loads,(ROOT/'results/v6/stress_test.jsonl').open()):
        text=row['latex'];ids=hf.encode(text)
        assert ids==reference.encode(text) and reference.decode(ids)==text
        stress+=1
    report_path=folder/'llama3_backend_equivalence.json'
    prior=json.loads(report_path.read_text()) if report_path.exists() else {}
    metrics=json.loads((folder/'metrics.json').read_text())
    prior.setdefault('hf_implementation_readouts', [{k:m[k] for k in ('suite','tokens_per_formula','command_integrity','independent_tree_crossing','encode_mib_s','decode_mib_s','encode_seconds_repeats','decode_seconds_repeats')}
                                                   for m in metrics if m['method']==record['repo']])
    prior.update(utc=datetime.now(timezone.utc).isoformat(),formulas=n,stress_formulas=stress,
        all_ids_equal=True,all_roundtrip=True,vocab_hf=hf.vocab_size,vocab_meta=reference.vocab_size,
        main_backend='Meta original tokenizer.py + tiktoken; bos=False eos=False',
        duplicate_segmentation_not_counted_as_extra_method=True,
        full_retiming_required_after_switch=True,official_code_revision=record['official_code_revision'])
    report_path.write_text(json.dumps(prior,indent=2)+'\n')
    registry=json.loads((folder/'registry.json').read_text())
    for entry in registry:
        if entry['name']==record['repo']:entry['kind']='llama3_official'
    (folder/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    print(json.dumps(prior,indent=2),flush=True)


if __name__=='__main__':main()
