"""Give remaining original classical algorithms the same fit+reward union."""
import json
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.baselines import SentencePieceTokenizer
from math_tokenizer.baselines_v6 import ByteWordPiece


def main():
    byid={r['id']:r for r in load_split('train')}
    for capacity in (1024,2048):
        for scenario in ('broad','specialized'):
            folder=ROOT/f'results/v7/{scenario}_{capacity}';scope=json.loads((folder/'fairness_protocol.json').read_text())
            texts=[byid[id]['latex'] for id in scope['fit_ids']+scope['reward_ids']]
            # Explicit check of the official trainer's accepted length range.
            assert max(len(t.encode()) for t in texts)<=16384
            records=[]
            for kind in ('bpe','unigram'):
                path=folder/'models'/f'sentencepiece_{kind}_union'
                if path.with_suffix('.model').exists():
                    import sentencepiece as spm
                    model=SentencePieceTokenizer(spm.SentencePieceProcessor(model_file=str(path.with_suffix('.model'))))
                else:model=SentencePieceTokenizer.train(texts,capacity,kind,path)
                records.append({'name':'sentencepiece_'+kind+'_union','kind':'sp','path':str(path.with_suffix('.model').relative_to(ROOT)),
                    'train_seconds':model.train_seconds,'fit_reward_count':len(texts),'capacity':capacity,'scenario':scenario})
            path=folder/'models'/'wordpiece_union.json'
            if path.exists():
                from tokenizers import Tokenizer
                model=ByteWordPiece(Tokenizer.from_file(str(path)))
            else:model=ByteWordPiece.train(texts,capacity,path)
            records.append({'name':'wordpiece_union','kind':'wp','path':str(path.relative_to(ROOT)),
                'train_seconds':model.train_seconds,'fit_reward_count':len(texts),'capacity':capacity,'scenario':scenario})
            (folder/'classical_union_registry.json').write_text(json.dumps(records,indent=2)+'\n')
            print(scenario,capacity,'official union baselines',len(records),flush=True)


if __name__=='__main__':main()
