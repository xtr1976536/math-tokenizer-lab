"""Stage one: common atoms, actual vocabularies and equal information controls."""
import argparse,json,hashlib,time
from pathlib import Path
from math_tokenizer.benchmark_v6 import load_split
from math_tokenizer.corpus import ROOT
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.rl_route_v6 import derive_atoms,AtomicVocab
from math_tokenizer.ast_route_v6 import NativeASTV4
from math_tokenizer.baselines import HFTokenizer
from math_tokenizer.baselines_v6 import PathPieceOfficial
from math_tokenizer.measure_v6 import measure,fit_usage
from math_tokenizer.current_models import load_current


def main():
    p=argparse.ArgumentParser();p.add_argument('--capacity',type=int,default=1024);p.add_argument('--scenario',default='broad');a=p.parse_args()
    folder=ROOT/f'results/v7/{a.scenario}_{a.capacity}';folder.mkdir(parents=True,exist_ok=True);models=folder/'models';models.mkdir(exist_ok=True)
    old=ROOT/f'results/v6/{a.scenario}_{a.capacity}'
    scope=json.loads((old/'scope.json').read_text());byid={r['id']:r for r in load_split('train')+load_split('dev')}
    fit=[byid[id] for id in scope['fit_ids']];reward=[byid[id] for id in scope['reward_ids']];dev=[byid[id] for id in scope['dev_ids']]
    atoms=derive_atoms(fit,a.capacity,384)
    protocol={'stage':'closest_baselines_before_innovation','capacity':a.capacity,'scenario':a.scenario,
        'fit_ids':scope['fit_ids'],'reward_ids':scope['reward_ids'],'dev_ids':scope['dev_ids'],
        'shared_atom_hex':[p.hex() for p in atoms],'same_primitive_alphabet':True,'all_ids_counted':True,
        'test_loads':0,'equal_information_control':'Baseline fit+reward uses union; fit-only identifies effect of extra reward access.',
        'model_selection':'No dev variant choice in this stage; official fixed settings and both data ranges reported.',
        'corpus_sha256':scope['corpus_sha256']}
    (folder/'fairness_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    metrics=[];registry=[]
    def score(model,name,path,information):
        if not all(model.decode(model.encode(s))==s for s in ['',r'\frac{\alpha}{x_i}+12345','α+😊=1',r'\unknownCorpusMacro{x}']):raise ValueError(name+' loses raw input')
        m,rows=measure(model,dev,name,repeats=1,training_usage=fit_usage(model,fit))
        m.update(capacity=a.capacity,scenario=a.scenario,information=information,phase='development',timing='provisional_one_pass_not_for_final_ranking')
        metrics.append(m);registry.append({'name':name,'path':str(path.relative_to(ROOT)) if path else None,'information':information})
        (folder/'closest_baseline_dev.json').write_text(json.dumps(metrics,indent=2)+'\n')
        print(a.scenario,a.capacity,name,round(m['document_macro_tokens_per_byte'],6),round(m['independent_tree_crossing'],4),round(m['command_integrity'],4),flush=True)
    for union,rows in [('fit_only',fit),('fit_reward_union',fit+reward)]:
        path=models/f'atomic_bpe_{union}.json'
        if path.exists():model=AtomicBPE.load(path)
        else:model=AtomicBPE.train(rows,atoms,a.capacity);model.save(path)
        score(model,'atomic_bpe_'+union,path,union)
        # Exactly the same vocabulary for all segmentation ablations.
        for hard,label in [(False,'lexical_shortest'),(True,'native_tree_shortest')]:
            native=NativeASTV4(model.vocab,hard);native.train_seconds=model.train_seconds;native.capacity=len(model.vocab);native.training_log=[{'same_dictionary_as':'atomic_bpe_'+union,'tree_constraint':hard}]
            path=models/f'{label}_{union}.json';native.save(path);score(native,label+'_'+union,path,union)
        # Original PathPiece over precisely the same raw-byte vocabulary.
        path=models/f'pathpiece_atomic_{union}.vocab';path.write_text(''.join(p.hex()+'\n' for p in model.vocab if p!=b'\x00'))
        original=PathPieceOfficial(path,model.train_seconds)
        assert original.vocab_size==len(model.vocab)
        score(original,'pathpiece_atomic_vocab_'+union,path,union)
        path=models/f'hf_byte_{union}.json'
        from tokenizers import Tokenizer
        raw=HFTokenizer(Tokenizer.from_file(str(path))) if path.exists() else HFTokenizer.train([r['latex'] for r in rows],a.capacity,path)
        score(raw,'hf_byte_'+union,path,union)
    for name in ('ast_v4','v5'):
        model=load_current(name,scenario=a.scenario,capacity=a.capacity);score(model,name+'_v6_parent',None,'fit_with_reward_selection')
    for name,file in [('v5_frequency_v6_parent','v5_structural_frequency.json'),('v5_greedy_v6_parent','v5_structural_reward_greedy.json')]:
        path=old/'models'/file;score(AtomicVocab.load(path),name,path,'fit_with_reward_selection')
    (folder/'closest_baseline_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    (folder/'stage1_audit.json').write_text(json.dumps({'all_raw_probe_roundtrip':True,'same_vocab_segmentation_ablation':True,
        'shared_atoms_learned_from_fit_only':True,'all_ids_counted':True,'fit_reward_union_available_to_official_baselines':True,
        'no_new_or_old_test_inputs_loaded':True,'dev_formulas':len(dev),'configurations':len(metrics)},indent=2)+'\n')


if __name__=='__main__':main()
