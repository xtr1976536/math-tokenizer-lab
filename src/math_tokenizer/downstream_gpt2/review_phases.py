"""Four local smoke-review rounds for the final two phase bundles."""
import json,hashlib,subprocess,sys
from pathlib import Path
import torch
from math_tokenizer.corpus import ROOT
from .phases import PHASES,SEEDS,PhaseAdapter,freeze_coverage,load_checkpoint,parameter_digest
from .data import generate
from .orchestrate import assemble
from .ood import generate as generate_ood
from .audit import dataset_audit
from .phase_runtime import audit_adapters
from .train import train_one
from .adapters import method_slug
from .phase_results import evaluate_method

OUT=ROOT/'tmp/two_phase_review'

def main():
    OUT.mkdir(parents=True,exist_ok=True);source=ROOT/'tmp/SymbolicMathematics';gpt2=ROOT/'tmp/gpt2-607a30d'
    report={}
    subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,check=True)
    assert SEEDS==(11,)
    report['round1_static']={'status':'passed','phases':PHASES,'training_seeds':SEEDS,'native_training_jobs':0}
    raw=OUT/'raw';data=OUT/'data';generate(source,raw,.001)
    assemble(raw,raw,raw,data,json.loads((raw/'manifest.json').read_text())['quotas'],source)
    audit=dataset_audit(data,source);coverage=freeze_coverage(data)
    for mode in ('length','composition'):generate_ood(mode,source,data,OUT/'ood_data',.002)
    adapters={phase:audit_adapters(phase,data,gpt2) for phase in PHASES}
    full_coverage=dict(coverage);full_coverage['kept']=dict(coverage['kept'])
    full_coverage['kept']['train']=coverage['kept']['train'][:8]
    full_coverage['sha256']=dict(coverage['sha256'])
    full_coverage['sha256']['train']=hashlib.sha256('\n'.join(full_coverage['kept']['train']).encode()).hexdigest()
    (data/'coverage_512.json').write_text(json.dumps(full_coverage,indent=2)+'\n')
    report['round2_data']={'status':'passed','oracle':audit,'adapters':adapters,'shared_ids_sha256':full_coverage['sha256'],'smoke_train_examples':8}
    histories={}
    for phase,spec in PHASES.items():
        for method in spec['trained_methods']:
            out=train_one(method,11,data,gpt2,OUT/phase/'runs',epochs=1,microbatch=2,effective_batch=4,
                          eval_limit=5,device_name='mps',mode=spec['mode'],prompt_style='text-v2')
            config=json.loads((out/'config.json').read_text());history=json.loads((out/'history.json').read_text())
            if phase=='phase2':assert config['frozen_before_sha256']==config['frozen_after_sha256']
            assert config['epoch_order_sha256']
            histories[phase+'|'+method]={'config':config,'loss':history[0]['train_loss_per_example']}
            reload=load_checkpoint(method,gpt2,out/'best',OUT/'reload_init',spec['mode']).eval()
            assert reload.get_input_embeddings().weight.data_ptr()==reload.get_output_embeddings().weight.data_ptr()
            del reload;torch.mps.empty_cache()
    keys=list(histories)
    assert len({histories[k]['config']['epoch_order_sha256'][0] for k in keys})==1
    for phase in PHASES:
        assert histories[phase+'|ast_boundary']['config']['embedding_sha256']==histories[phase+'|official_atomic_bpe']['config']['embedding_sha256']
    native=evaluate_method('phase1','gpt2_native',data,gpt2,None,OUT/'native_eval',device='mps',smoke_limit=1)
    assert native['seeds']['zero_shot']['optimizer_steps']==0
    report['round3_training']={'status':'passed','runs':histories,'native':native}
    replay={}
    for phase,spec in PHASES.items():
        out=train_one('ast_boundary',11,data,gpt2,OUT/phase/'replay',epochs=1,microbatch=2,effective_batch=4,
                      eval_limit=5,device_name='mps',mode=spec['mode'],prompt_style='text-v2')
        loss=json.loads((out/'history.json').read_text())[0]['train_loss_per_example']
        prior=histories[phase+'|ast_boundary']['loss'];assert abs(loss-prior)<1e-3
        replay[phase]={'first_loss':prior,'replay_loss':loss,'difference':abs(loss-prior)}
    report['round4_replay']={'status':'passed','replays':replay}
    (OUT/'review.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(OUT/'review.json',flush=True)

if __name__=='__main__':main()
