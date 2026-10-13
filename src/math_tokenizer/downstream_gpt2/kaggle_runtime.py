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


def _find_data():
    matches = list(Path("/kaggle/input").glob(f"{DATA_KERNEL_SLUG}/final_data/manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one mounted frozen-data kernel, found {matches}")
    return matches[0].parent


def run_smoke():
    source = PROJECT / "tmp/SymbolicMathematics"
    gpt2 = PROJECT / "tmp/gpt2-607a30d"
    data = WORKING / "pilot_data"
    generate(source, data, .001)
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
    data = _find_data()
    gpt2 = PROJECT / "tmp/gpt2-607a30d"
    for seed in (11,23,37):
        train_one(method,seed,data,gpt2,WORKING/"runs",epochs=3,
                  microbatch=2,effective_batch=32,device_name="cuda")
    shutil.rmtree(PROJECT)
    shutil.rmtree(WORKING/"init",ignore_errors=True)


def run_evaluate(method: str):
    if method not in METHODS:
        raise ValueError(method)
    data = _find_data()
    slug = method_slug(method)
    matches = list(Path("/kaggle/input").glob(f"math-tokenizer-gpt2-{kernel_slug(method)}-v1/runs/{slug}"))
    if len(matches) != 1:
        raise RuntimeError(f"missing training kernel output for {method}: {matches}")
    evaluate_method(method,data,PROJECT/"tmp/gpt2-607a30d",matches[0],WORKING/"eval_results","cuda")
    shutil.rmtree(PROJECT)


def run_aggregate():
    outputs={}
    for method in METHODS:
        slug=f"math-tokenizer-gpt2-eval-{kernel_slug(method)}-v1"
        matches=list(Path("/kaggle/input").glob(f"{slug}/eval_results/summary.json"))
        if len(matches)!=1:
            raise RuntimeError(f"missing evaluation output for {method}: {matches}")
        outputs[method]=matches[0].parent
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
