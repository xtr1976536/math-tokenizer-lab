"""Predeclared test-only transformations; source parents kept for clustering."""
import hashlib,json,subprocess
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import RELEASE,load_split
from math_tokenizer.trajectory_v5 import safe_layout_variant
from math_tokenizer.frontend_v6 import reference_lex
from math_tokenizer.quality_v6 import formula_gate
from math_tokenizer.structure_v2 import complexity


def build():
    out=ROOT/'results/v6';protocol=json.loads((out/'evaluation_protocol.json').read_text())
    # Every route/size must be frozen before generating test-dependent probes.
    for capacity in protocol['capacities']:
        for scenario in protocol['scenarios']:
            assert (out/f'{scenario}_{capacity}/final_selection.json').exists()
    parents=sorted(load_split('fresh_test'),key=lambda r:hashlib.sha256(('v6-stress:'+r['id']).encode()).digest())[:192]
    result=[]
    for parent in parents:
        text=parent['latex'];variants=[('layout',safe_layout_variant(text))]
        if not any(chr(92)+name in text for name in ('text','operatorname','mathrm','mathbb','mathcal')):
            raw=text.encode();parts=[];mapping={};names=list('zyxwvutsrqponmlkjihgfedcba')
            for a,b in reference_lex(text):
                token=raw[a:b].decode()
                if len(token)==1 and token in 'xyzuvwtijknmpqabc':mapping.setdefault(token,names[len(mapping)]);parts.append(mapping[token])
                else:parts.append(token)
            variants.append(('symbol_renaming',''.join(parts)))
        if len(text.encode())<=800:
            # Tuple of original expressions, not a fabricated equality/theorem.
            for repeats in (8,32):variants.append(('long_tuple_'+str(repeats),r'\left('+','.join('{'+text+'}' for _ in range(repeats))+r'\right)'))
        for kind,variant in variants:
            if variant==text:continue
            id=hashlib.sha256((parent['id']+kind+variant).encode()).hexdigest()[:24]
            result.append(dict(parent,id=id,latex=variant,source_parent_id=parent['id'],stress=kind,origin='synthetic_source_transformation',complexity=complexity(variant)))
    raw=ROOT/'data/raw/benchmark_v6';input=raw/'stress_input.jsonl';output=raw/'stress_reference.jsonl'
    input.write_text(''.join(json.dumps({'id':r['id'],'latex':r['latex']},ensure_ascii=False)+'\n' for r in result))
    with output.open('w') as f:subprocess.run(['node',str(ROOT/'scripts/latex_reference_v6.js'),str(input)],stdout=f,check=True)
    references={r['id']:r for r in map(json.loads,output.open())};valid=[];rejected=[]
    for r in result:
        ref=references[r['id']]
        if not ref['valid']:rejected.append({'id':r['id'],'stress':r['stress'],'reason':ref['reason']});continue
        r['reference']={k:v for k,v in ref.items() if k not in ('id','valid')};valid.append(r)
    path=out/'stress_test.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in valid))
    (out/'stress_manifest.json').write_text(json.dumps({'formulas':len(valid),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rejections':rejected,'purpose':'controlled syntax/length generalization, not mathematically certified equivalent transformations','source_parents':len(parents)},indent=2))
    return valid

if __name__=='__main__':print('stress formulas',len(build()))
