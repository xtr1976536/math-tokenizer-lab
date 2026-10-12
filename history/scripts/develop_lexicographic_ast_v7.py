"""Parameter-free primary length / secondary native structure DP control."""
import json
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.ast_balanced_v7 import BalancedAST
from math_tokenizer.measure_v6 import measure,fit_usage,selection_utility


def main():
    byid={r['id']:r for r in load_split('train')+load_split('dev')}
    for capacity in (1024,2048):
        for scenario in ('broad','specialized'):
            folder=ROOT/f'results/v7/{scenario}_{capacity}';scope=json.loads((folder/'fairness_protocol.json').read_text())
            fit=[byid[id] for id in scope['fit_ids']];dev=[byid[id] for id in scope['dev_ids']]
            base=AtomicBPE.load(folder/'models/atomic_bpe_fit_reward_union.json')
            model=BalancedAST(base.vocab,0.,lexicographic=True);model.train_seconds=base.train_seconds
            model.training_log=[{'dictionary_source':'atomic_bpe_fit_reward_union','same_dictionary':True,
                'objective':'lexicographic(token count, native crossing count), no weighting parameter'}]
            model.save(folder/'models/ast_lexicographic.json')
            m,_=measure(model,dev,'ast_lexicographic',repeats=1,training_usage=fit_usage(model,fit))
            m.update(dev_selection_utility=selection_utility(m),phase='development',lexicographic=True)
            (folder/'lexicographic_dev.json').write_text(json.dumps(m,indent=2)+'\n')
            print(capacity,scenario,m['document_macro_tokens_per_byte'],m['independent_tree_crossing'],flush=True)


if __name__=='__main__':main()
