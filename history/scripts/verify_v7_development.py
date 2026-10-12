"""Independent dev replay following extension of the native lattice API."""
from collections import Counter
import hashlib,json
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.ast_balanced_v7 import BalancedAST
from math_tokenizer.metrics_v2 import token_bytes
from math_tokenizer.measure_v5 import cross_count


def main():
    byid={r['id']:r for r in load_split('dev')};checks=[]
    sha=hashlib.sha256((ROOT/'native/balanced_v7.c').read_bytes()).hexdigest()
    for folder in sorted(list((ROOT/'results/v7').glob('*_1024'))+list((ROOT/'results/v7').glob('*_2048'))):
        scope=json.loads((folder/'fairness_protocol.json').read_text());rows=[byid[id] for id in scope['dev_ids']]
        stats={}
        for path in folder.glob('*_dev.json'):
            d=json.loads(path.read_text());rs=d if isinstance(d,list) else [d]
            stats.update({r['method']:r for r in rs if isinstance(r,dict) and 'method' in r})
        for path in folder.joinpath('models').glob('*.json'):
            d=json.loads(path.read_text())
            if d.get('format')!='ast-v4-balanced-lattice-v7':continue
            name=path.stem
            if name not in stats:continue
            previous=d['balanced_source_sha256']
            # API extension kept risk=None semantics; validate full dev output
            # BEFORE persisting metadata for updated native source integrity.
            model=BalancedAST([bytes.fromhex(p) for p in d['vocab_hex']],d['penalty'],d.get('risk_weights'),d.get('lexicographic',False))
            token_total=crossing=0
            for r in rows:
                ids=model.encode(r['latex']);payload=[token_bytes(model,id) for id in ids]
                assert b''.join(payload)==r['latex'].encode() and model.decode(ids)==r['latex']
                bounds=[0]
                for p in payload:bounds.append(bounds[-1]+len(p))
                token_total+=len(ids);crossing+=cross_count(bounds,r['reference']['spans'])
            m=stats[name]
            assert token_total==m['tokens'] and abs(crossing/token_total-m['independent_tree_crossing'])<1e-12,(name,token_total,m['tokens'])
            if previous!=sha:
                d['balanced_source_sha256']=sha;path.write_text(json.dumps(d,indent=2)+'\n')
            checks.append({'path':str(path.relative_to(ROOT)),'dev_formulas':len(rows),'tokens_unchanged':True,'crossing_unchanged':True,
                           'lossless_all':True,'previous_native_sha256':previous,'current_native_sha256':sha})
    (ROOT/'results/v7/development_replay.json').write_text(json.dumps(checks,indent=2)+'\n');print('dev model replays',len(checks),flush=True)


if __name__=='__main__':main()
