"""Audit target-equation and coefficient-variant exclusion in fast runs."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.author import CODE, DICTIONARY
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
from math_tokenizer.eqgpt.data import canonical
from math_tokenizer.eqgpt.evaluation import TRUTH
from math_tokenizer.eqgpt.latex_codec import latex_to_structure, structure_to_latex
from math_tokenizer.eqgpt.latex_tokenizers import LatexAdapter
from math_tokenizer.eqgpt.shared8_config import SHARED8_TARGETS

RUNS={
    'Wave_equation':'eqgpt_fast_reward_20261022',
    'Convection_diffusion_equation':'eqgpt_fast_reward_Convection_diffusion_equation_20261022',
    'KG_equation':'eqgpt_fast_reward_KG_equation_20261022',
    'Burgers_equation':'eqgpt_fast_reward_burgers_20261022',
    'Eq_6_2_12':'eqgpt_fast_reward_Eq_6_2_12_20261022',
    'PDE_divide':'eqgpt_fast_reward_PDE_divide_20261022',
    'KdV_equation':'eqgpt_fast_reward_KdV_equation_20261022',
    'Chaffee_Infante_equation':'eqgpt_fast_reward_chaffee_20261022',
}
TARGET_NAMES={
    'Wave_equation':'(1.2.10)',
    'Convection_diffusion_equation':'6.2.13',
    'KG_equation':'Klein–Gordon_u',
    'Burgers_equation':'Bateman-Burgers equation',
    'Eq_6_2_12':'6.2.12',
    'PDE_divide':'8.14.1d',
    'KdV_equation':'KdV equation',
    'Chaffee_Infante_equation':'Chafee–Infante equation',
}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def gre_key(sequence):
    return canonical([DICTIONARY['id2word'][int(i)] for i in sequence])


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--output',type=Path,default=ROOT/'reports/eqgpt_fast_reward_20261022/target_exclusion_audit.json')
    args=cli.parse_args()
    source=CODE/'PDE_dataset_0725.xlsx'
    df=pd.read_excel(source)
    target_rows=df[df['Equation name'].isin(SHARED8_TARGETS)]
    assert set(SHARED8_TARGETS)<=set(target_rows['Equation name'])
    target_keys={name:canonical(str(target_rows[target_rows['Equation name']==name].iloc[0]['Terms']).split(','))
                 for name in SHARED8_TARGETS}
    target_keys_set=set(target_keys.values())
    targets=[];all_aliases=set();case_reports=[]
    for case,name in TARGET_NAMES.items():
        key=canonical(str(df[df['Equation name']==name].iloc[0]['Terms']).split(','))
        expected=tuple(sorted(canonical(term.split(','))[0] for term,coef in TRUTH[case]))
        assert key==expected,(case,name,key,expected)
        aliases=df[df['Terms'].map(lambda x:canonical(str(x).split(','))==key)]['Equation name'].astype(str).tolist()
        all_aliases.update(aliases)
        targets.append(dict(case=case,handbook_name=name,coefficient_free_term_key=repr(key),
                            same_structure_handbook_rows=aliases))
    forbidden_rows=df[df['Terms'].map(lambda x:canonical(str(x).split(',')) in target_keys_set)]
    forbidden_names=set(forbidden_rows['Equation name'].astype(str))
    assert all_aliases<=forbidden_names
    case_reports=[]
    for case,dirname in RUNS.items():
        run=ROOT/'experiments'/dirname
        sequences=json.loads((run/'source/train_gre.json').read_text())
        training_manifest=json.loads((run/'source/training_manifest.json').read_text())
        assert tuple(training_manifest['target_name'])==tuple(SHARED8_TARGETS)
        removed=set(map(str,training_manifest['removed_equations']))
        assert set(SHARED8_TARGETS)<=removed
        assert forbidden_names<=removed,(case,sorted(forbidden_names-removed))
        for seq in sequences:
            assert gre_key(seq) not in target_keys_set,(case,seq)
        texts=[structure_to_latex(seq) for seq in sequences]
        tokenizer_manifest=json.loads((run/'tokenizers/manifest.json').read_text())
        corpus_digest=sha_bytes(json.dumps(texts,ensure_ascii=False).encode())
        assert tokenizer_manifest['corpus_sha256']==corpus_digest
        per_arm=[]
        for arm in json.loads((run/'complete.json').read_text())['results']:
            if arm=='eqgpt_reference':
                encoded=[[5]+seq[1:] for seq in sequences]
                decoded=sequences
                tokenizer_corpus_digest=None
            else:
                method,size=arm.rsplit('_',1);size=int(size)
                model=CompactAlphabet.load(run/'tokenizers'/f'{method}_{size}.model.json')
                adapter=LatexAdapter(method,model)
                encoded=[adapter.sequence(text) for text in texts]
                roundtrip=[adapter.decode(ids[1:-1]) for ids in encoded]
                assert roundtrip==texts,(case,arm,'tokenizer round-trip')
                decoded=[latex_to_structure(text) for text in roundtrip]
                tokenizer_corpus_digest=corpus_digest
            for seq in decoded:
                assert gre_key(seq) not in target_keys_set,(case,arm,seq)
            training=json.loads((run/case/arm/'training.json').read_text())
            actual_sequence_sha=sha_bytes(json.dumps(encoded).encode())
            assert training['sequence_sha256']==actual_sequence_sha,(case,arm,'training hash')
            assert training['samples']==len(encoded)==6784
            per_arm.append(dict(arm=arm,training_examples=training['samples'],
                training_sequence_sha256=actual_sequence_sha,tokenizer_training_corpus_sha256=tokenizer_corpus_digest,
                encoded_sequences_checked=len(decoded),roundtrip_checked=arm!='eqgpt_reference',
                target_structures_found=0))
        case_reports.append(dict(case=case,source_gre_sequences=len(sequences),
            tokenizer_training_strings=len(texts),source_gre_sha256=training_manifest['training_sha256'],
            tokenizer_corpus_sha256=corpus_digest,removed_target_and_coefficient_variant_rows=len(forbidden_names),
            arms=per_arm))
        print('PASS',case,'GRE',len(sequences),'arms',len(per_arm),'coefficient_variant_rows',len(forbidden_names),flush=True)
    report=dict(status='PASS',source_file=str(source.relative_to(ROOT)),source_sha256=sha_bytes(source.read_bytes()),
        target_count=len(targets),target_equations=targets,
        excluded_same_structure_handbook_rows=sorted(forbidden_names),excluded_row_count=len(forbidden_names),
        coefficient_handling='Target matching uses coefficient-free canonical term multisets; coefficient-only variants have the same key and are excluded.',
        tokenizer_rule='Tokenizer corpus hash matches the exact target-excluded LaTeX strings consumed by the training adapters.',
        transformer_rule='Recomputed method-specific tokenized sequence hash equals each saved training.json sequence hash; each stored and decoded training sequence is target-structure-free.',
        case_count=len(case_reports),model_training_stream_count=sum(len(x['arms']) for x in case_reports),cases=case_reports)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('TARGET_EXCLUSION_AUDIT_PASS',len(case_reports),report['model_training_stream_count'],flush=True)


if __name__=='__main__':main()
