"""Fit official classical and paper baselines on exact same route fit IDs."""
import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import master_path,load_split
from math_tokenizer.baselines import HFTokenizer,SentencePieceTokenizer
from math_tokenizer.baselines_v6 import PathPieceOfficial,ByteWordPiece
from math_tokenizer.rl_vocab_v5 import bytealphabet

def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');p.add_argument('--capacity',type=int,default=1024);args=p.parse_args()
    out=ROOT/f'results/v6/{args.scenario}_{args.capacity}';scope=json.loads((out/'scope.json').read_text());ids=set(scope['fit_ids'])
    rows={r['id']:r for r in map(json.loads,master_path().open()) if r['id'] in ids};texts=[rows[id]['latex'] for id in scope['fit_ids']]
    models=out/'models';logpath=out/'baselines.json';logs=json.loads(logpath.read_text()) if logpath.exists() else []
    def store(name,path,model):
        assert model.vocab_size<=args.capacity,(name,model.vocab_size,args.capacity)
        logs.append({'name':name,'path':str(path.relative_to(ROOT)),'vocab_actual':model.vocab_size,'train_seconds':model.train_seconds,'fit_id_count':len(texts),'upstream_modified':False})
        logpath.write_text(json.dumps(logs,indent=2)+'\n');print(name,model.vocab_size,model.train_seconds,flush=True)
    done={x['name'] for x in logs}
    hf_path=models/'hf_bpe.json'
    if 'hf_bpe' not in done:store('hf_bpe',hf_path,HFTokenizer.train(texts,args.capacity,hf_path))
    for kind in ('bpe','unigram'):
        name='sentencepiece_'+kind;path=models/name
        if name not in done:store(name,path.with_suffix('.model'),SentencePieceTokenizer.train(texts,args.capacity,kind,path))
    if 'wordpiece_bytes' not in done:
        path=models/'wordpiece_bytes.json';store('wordpiece_bytes',path,ByteWordPiece.train(texts,args.capacity,path))
    if 'pathpiece_bpe_vocab' not in done:
        hf=Tokenizer.from_file(str(hf_path));reverse={c:b for b,c in bytealphabet().items()}
        payloads=[bytes(reverse[c] for c in hf.id_to_token(id)) for id in range(hf.get_vocab_size())]
        path=models/'pathpiece.vocab';path.write_text(''.join(p.hex()+'\n' for p in payloads if p!=b'\x00'))
        model=PathPieceOfficial(path,next(x['train_seconds'] for x in logs if x['name']=='hf_bpe'));store('pathpiece_bpe_vocab',path,model)

if __name__=='__main__':main()
