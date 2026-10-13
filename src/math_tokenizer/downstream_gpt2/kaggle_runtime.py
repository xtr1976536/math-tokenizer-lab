"""Entry points for private Kaggle CPU/GPU jobs."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .adapters import METHODS, TokenizerAdapter, build_model, kernel_slug, method_slug
from .audit import dataset_audit, tokenizer_audit
from .data import generate
from .ood import generate as generate_ood
from .orchestrate import assemble
from .results import aggregate_method_outputs, evaluate_method
from .train import common_coverage, loss_for, train_one

OWNER = "xiaotongrenkk"
ASSET_SLUG = "math-tokenizer-gpt2-assets-v1-20261013"
DATA_KERNEL_SLUG = "math-tokenizer-gpt2-data-v1"
WORKING = Path("/kaggle/working")
PROJECT = WORKING / "math_tokenizer_project"


def find_data(input_root: Path = Path("/kaggle/input")):
    matches = [p for p in input_root.rglob("manifest.json") if p.parent.name=="final_data"]
    if len(matches) != 1:
        raise RuntimeError(f"expected one mounted frozen-data kernel, found {matches}")
    return matches[0].parent


def find_method_run(method: str, input_root: Path = Path("/kaggle/input")):
    slug=method_slug(method)
    matches=[p/slug for p in input_root.rglob("runs") if (p/slug).is_dir()]
    if len(matches)!=1:
        raise RuntimeError(f"expected one mounted training output for {method}: {matches}")
    return matches[0]


def find_eval_outputs(input_root: Path = Path("/kaggle/input")):
    outputs={}
    for path in input_root.rglob("summary.json"):
        if path.parent.name=="eval_results":
            summary=json.loads(path.read_text())
            method=summary.get("method")
            if method in outputs:
                raise RuntimeError(f"duplicate evaluation output for {method}")
            outputs[method]=path.parent
    if set(outputs)!=set(METHODS):
        raise RuntimeError(f"missing evaluation outputs: {set(METHODS)-set(outputs)}")
    return outputs


def run_smoke():
    source = PROJECT / "tmp/SymbolicMathematics"
    gpt2 = PROJECT / "tmp/gpt2-607a30d"
    raw = WORKING / "pilot_raw_data"
    generate(source, raw, .001)
    data = WORKING / "pilot_data"
    pilot_quotas=json.loads((raw/"manifest.json").read_text())["quotas"]
    assemble(raw,raw,raw,data,pilot_quotas,source)
    audit = {"round_1": dataset_audit(data,source),
             "round_2": tokenizer_audit(data,gpt2)}
    (data / "coverage_512.json").write_text(json.dumps(common_coverage(data,gpt2),indent=2)+"\n")
    for method in METHODS:
        train_one(method,11,data,gpt2,WORKING/"smoke_runs",epochs=1,
                  microbatch=2,effective_batch=4,eval_limit=5,device_name="cuda")
    import torch
    memory = {}
    for method in ("Qwen2.5-0.5B","meta-llama/Meta-Llama-3-8B","tiktoken_o200k_base"):
        adapter=TokenizerAdapter(method,gpt2)
        model,_=build_model(adapter,gpt2,WORKING/"init")
        model.config.use_cache=False
        model.gradient_checkpointing_enable()
        model.to("cuda").train()
        x=adapter.encode("x")[0]
        prompt=[adapter.special_ids["BOS"],adapter.special_ids["INT"]]+[x]*253+[adapter.special_ids["SEP"]]
        answer=[x]*255+[adapter.special_ids["EOS"]]
        torch.cuda.reset_peak_memory_stats()
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
        loss=loss_for(model,[(None,prompt,answer)]*2,adapter,adapter.masked_ids(),"cuda")
        loss.backward()
        optimizer.step()
        memory[method]={"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"loss":float(loss.detach())}
        del model,optimizer
        torch.cuda.empty_cache()
    (WORKING / "smoke_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    (WORKING / "gpu_memory_preflight.json").write_text(json.dumps(memory,indent=2)+"\n")
    for path in (WORKING/"smoke_runs").rglob("best"):
        shutil.rmtree(path)
    shutil.rmtree(PROJECT)
    shutil.rmtree(WORKING/"init",ignore_errors=True)
    shutil.rmtree(raw,ignore_errors=True)


def run_data():
    source = PROJECT / "tmp/SymbolicMathematics"
    gpt2 = PROJECT / "tmp/gpt2-607a30d"
    jobs = (("prim_fwd","prim_bwd"), ("prim_ibp",), ("ode1",))
    folders = [WORKING / "source_main",WORKING / "source_ibp",WORKING / "source_ode"]
    subprocess_env = dict(os.environ, PYTHONPATH=str(PROJECT/"src"))
    active = []
    for methods,folder in zip(jobs,folders):
        command = [sys.executable,"-m","math_tokenizer.downstream_gpt2.data",
                   "--upstream",str(source),"--output",str(folder),"--methods",*methods]
        active.append(subprocess.Popen(command,cwd=PROJECT,env=subprocess_env))
    for process in active:
        if process.wait() != 0:
            raise RuntimeError("source data generation failed")
    final = WORKING / "final_data"
    assemble(*folders,final)
    audit = {"round_1":dataset_audit(final,source),
             "round_2":tokenizer_audit(final,gpt2)}
    (final / "coverage_512.json").write_text(json.dumps(common_coverage(final,gpt2),indent=2)+"\n")
    ood = WORKING / "ood_data"
    active = []
    for mode in ("length","composition"):
        command = [sys.executable,"-m","math_tokenizer.downstream_gpt2.ood",
                   "--mode",mode,"--source",str(source),"--main-data",str(final),"--out",str(ood)]
        active.append(subprocess.Popen(command,cwd=PROJECT,env=subprocess_env))
    for process in active:
        if process.wait() != 0:
            raise RuntimeError("OOD data generation failed")
    (WORKING / "data_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    for folder in folders:
        shutil.rmtree(folder,ignore_errors=True)
    shutil.rmtree(PROJECT)


def run_train(method: str):
    if method not in METHODS:
        raise ValueError(method)
    data = find_data()
    gpt2 = PROJECT / "tmp/gpt2-607a30d"
    for seed in (11,23,37):
        train_one(method,seed,data,gpt2,WORKING/"runs",epochs=3,
                  microbatch=2,effective_batch=32,device_name="cuda")
    shutil.rmtree(PROJECT)
    shutil.rmtree(WORKING/"init",ignore_errors=True)


def run_evaluate(method: str):
    if method not in METHODS:
        raise ValueError(method)
    data = find_data()
    run_folder=find_method_run(method)
    evaluate_method(method,data,PROJECT/"tmp/gpt2-607a30d",run_folder,WORKING/"eval_results","cuda")
    shutil.rmtree(PROJECT)


def run_aggregate():
    outputs=find_eval_outputs()
    aggregate_method_outputs(outputs,WORKING/"final_results")
    shutil.rmtree(PROJECT)


def main(phase: str, method: str | None = None):
    if phase in ("smoke","train","evaluate"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Kaggle GPU is unavailable")
        print("GPU",torch.cuda.get_device_name(0),"VRAM",torch.cuda.get_device_properties(0).total_memory)
    if phase == "smoke":
        run_smoke()
    elif phase == "data":
        run_data()
    elif phase == "train":
        run_train(method)
    elif phase == "evaluate":
        run_evaluate(method)
    elif phase == "aggregate":
        run_aggregate()
    else:
        raise ValueError(phase)
