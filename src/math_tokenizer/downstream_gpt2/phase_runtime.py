"""Kaggle/manual entry points for the two self-contained phase bundles."""
from __future__ import annotations
import json,os,shutil,subprocess,sys
from pathlib import Path
from .phases import PHASES,SEEDS,PhaseAdapter,freeze_coverage
from .data import generate
from .ood import generate as generate_ood
from .orchestrate import assemble
from .audit import dataset_audit
from .train import train_one,read_split
from .phase_results import evaluate_method,aggregate
from .kaggle_runtime import find_data,find_method_run,PROJECT,WORKING

def audit_adapters(phase,data,gpt2):
    report={};rows=read_split(data,"train")[:100]
    for method in PHASES[phase]["methods"]:
        a=PhaseAdapter(method,gpt2)
        for row in rows:
            p,t=a.row(row["task"],row["input_latex"],row["target_latex"])
            from .phases import prompt_text
            if a.decode(p)!=prompt_text(row["task"],row["input_latex"]):raise RuntimeError("prompt roundtrip failed")
            if a.decode(t[:-1])!=row["target_latex"]:raise RuntimeError("target roundtrip failed")
        report[method]={"vocab_rows":a.vocab_size,"masked_ids":len(a.masked_ids())}
    return report

def prepare_data(phase,smoke=False):
    source=PROJECT/"tmp/SymbolicMathematics";gpt2=PROJECT/"tmp/gpt2-607a30d"
    folders=[WORKING/"source_main",WORKING/"source_ibp",WORKING/"source_ode"]
    if smoke:
        raw=WORKING/"source_main";generate(source,raw,.001)
        quotas=json.loads((raw/"manifest.json").read_text())["quotas"]
        final=WORKING/"final_data";assemble(raw,raw,raw,final,quotas,source)
    else:
        env=dict(os.environ,PYTHONPATH=str(PROJECT/"src"));active=[]
        for methods,folder in zip((("prim_fwd","prim_bwd"),("prim_ibp",),("ode1",)),folders):
            active.append(subprocess.Popen([sys.executable,"-m","math_tokenizer.downstream_gpt2.data",
                          "--upstream",str(source),"--output",str(folder),"--methods",*methods],env=env,cwd=PROJECT))
        for job in active:
            if job.wait():raise RuntimeError("data generator failed")
        final=WORKING/"final_data";assemble(*folders,final,upstream_source=source)
    audit={"dataset":dataset_audit(final,source),"adapters":audit_adapters(phase,final,gpt2)}
    audit["coverage"]=freeze_coverage(final)
    for mode in ("length","composition"):
        generate_ood(mode,source,final,WORKING/"ood_data",.002 if smoke else 1.0)
    (WORKING/"data_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    return final

def main(phase,job,method=None):
    spec=PHASES[phase];gpt2=PROJECT/"tmp/gpt2-607a30d"
    if job in ("smoke","train","evaluate"):
        import torch
        if not torch.cuda.is_available():raise RuntimeError("GPU required")
    if job=="data":prepare_data(phase)
    elif job=="smoke":
        data=prepare_data(phase,True)
        for candidate in spec["trained_methods"]:
            train_one(candidate,11,data,gpt2,WORKING/"runs",epochs=1,microbatch=2,effective_batch=4,
                      eval_limit=5,device_name="cuda",mode=spec["mode"],prompt_style="text-v2")
        for candidate in spec["methods"]:
            folder=None if candidate=="gpt2_native" else WORKING/"runs"/candidate.replace("/","__").replace("-","_")
            evaluate_method(phase,candidate,data,gpt2,folder,WORKING/"smoke_results"/candidate.replace("/","__"),
                            device="cuda",smoke_limit=2)
    elif job=="train":
        if method not in spec["trained_methods"]:raise ValueError("method must not be trained in this phase")
        data=find_data()
        for seed in SEEDS:
            train_one(method,seed,data,gpt2,WORKING/"runs",epochs=spec["epochs"],microbatch=2,effective_batch=32,
                      device_name="cuda",mode=spec["mode"],prompt_style="text-v2")
    elif job=="evaluate":
        if method not in spec["methods"]:raise ValueError(method)
        folder=None if method=="gpt2_native" else find_method_run(method)
        evaluate_method(phase,method,find_data(),gpt2,folder,WORKING/"eval_results")
    elif job=="aggregate":
        outputs={}
        for p in Path("/kaggle/input").rglob("summary.json"):
            if p.parent.name=="eval_results":
                s=json.loads(p.read_text());m=s.get("method")
                if m in outputs:raise RuntimeError("duplicate evaluation source")
                outputs[m]=p.parent
        aggregate(phase,outputs,WORKING/"final_results")
    else:raise ValueError(job)
    if job=="smoke":
        shutil.rmtree(WORKING/"runs",ignore_errors=True)
    for p in (PROJECT,WORKING/"init",WORKING/"source_main",WORKING/"source_ibp",WORKING/"source_ode"):
        shutil.rmtree(p,ignore_errors=True)
