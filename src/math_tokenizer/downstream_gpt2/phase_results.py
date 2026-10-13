"""Identical test protocol for the full-update and LoRA phase packages."""
from __future__ import annotations
import csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
import torch
from .adapters import method_slug
from .phases import PHASES,SEEDS,PROTOCOL,PhaseAdapter,load_checkpoint,parameter_digest,prompt_text
from .train import read_split,examples_for,evaluate
from .results import score_table,stratified_beam_ids,paired_interval
from .verify import Verifier
from math_tokenizer.corpus import ROOT

def ood_rows(data):
    output={}
    for mode in ("length","composition"):
        rows=[]
        for kind in ("prim_fwd","prim_bwd","prim_ibp","ode1"):
            path=data.parent/"ood_data"/f"ood_{mode}_{kind}.jsonl"
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
        output[mode]=[r for r in rows if len(prompt_text(r["task"],r["input_latex"]).encode())<=512
                      and len(r["target_latex"].encode())<=256]
    return output

def evaluate_method(phase,method,data,gpt2,run_folder,output,device="cuda",smoke_limit=None):
    spec=PHASES[phase]
    if method not in spec["methods"]:raise ValueError("method outside phase")
    coverage=json.loads((data/"coverage_512.json").read_text())
    if coverage.get("protocol")!=PROTOCOL:raise RuntimeError("incorrect dataset protocol")
    allowed=set(coverage["kept"]["test"])
    rows=[r for r in read_split(data,"test") if r["id"] in allowed]
    beam_ids=stratified_beam_ids(rows)
    suites={"test":rows,"beam10":[r for r in rows if r["id"] in beam_ids],**ood_rows(data)}
    verifier=Verifier(ROOT/"tmp/SymbolicMathematics")
    output.mkdir(parents=True,exist_ok=True)
    summary={"phase":phase,"method":method,"protocol":PROTOCOL,
             "adaptation":"zero_shot" if method=="gpt2_native" else spec["mode"],
             "coverage_sha256":coverage["sha256"],"suite_id_sha256":{},"seeds":{}}
    for name,items in suites.items():
        if smoke_limit is not None:suites[name]=items[:smoke_limit]
        summary["suite_id_sha256"][name]=hashlib.sha256("\n".join(sorted(r["id"] for r in suites[name])).encode()).hexdigest()
    seeds=("zero_shot",) if method=="gpt2_native" else SEEDS
    summary["smoke_only"]=smoke_limit is not None
    for seed in seeds:
        adapter=PhaseAdapter(method,gpt2)
        folder=None if method=="gpt2_native" else run_folder/str(seed)/"best"
        model=load_checkpoint(method,gpt2,folder,output.parent/"init",spec["mode"]).to(device).eval()
        before=parameter_digest(model) if method=="gpt2_native" else None
        tables={}
        for suite,items in suites.items():
            examples=examples_for(adapter,items,{r["id"] for r in items})
            _scores,predictions=evaluate(model,adapter,examples,verifier,adapter.masked_ids(),device,
                                         beams=10 if suite=="beam10" else 1,protocol="text-v2")
            tables[suite]=score_table(predictions)
            with gzip.open(output/f"{method_slug(method)}_{seed}_{suite}.jsonl.gz","wt") as stream:
                for item in predictions:stream.write(json.dumps(item,ensure_ascii=False)+"\n")
        tables["parameters"]=sum(p.numel() for p in model.parameters())
        if method=="gpt2_native":
            if before!=parameter_digest(model):raise RuntimeError("zero-shot GPT-2 weights changed")
            tables["model_state_sha256"]=before
            tables["optimizer_steps"]=0
        else:
            config=json.loads((run_folder/str(seed)/"config.json").read_text())
            if config["mode"]!=spec["mode"] or config["prompt_style"]!="text-v2":raise RuntimeError("wrong adaptation checkpoint")
            if smoke_limit is None and config["epochs"]!=spec["epochs"]:raise RuntimeError("wrong epoch budget")
            if config["coverage_sha256"]!=coverage["sha256"]:raise RuntimeError("training/evaluation data manifests differ")
            if smoke_limit is None and config["effective_batch"]!=32:raise RuntimeError("wrong effective batch")
            tables["train_config"]=config
            tables["train_history"]=json.loads((run_folder/str(seed)/"history.json").read_text())
        summary["seeds"][str(seed)]=tables
        (output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
        del model
        if str(device).startswith("cuda"):torch.cuda.empty_cache()
        elif str(device).startswith("mps"):torch.mps.empty_cache()
    return summary

def aggregate(phase,outputs,output,replicates=2000):
    spec=PHASES[phase]
    if set(outputs)!=set(spec["methods"]):raise RuntimeError("missing or extra phase outputs")
    summaries={m:json.loads((p/"summary.json").read_text()) for m,p in outputs.items()}
    if any(s.get("smoke_only") for s in summaries.values()):raise RuntimeError("smoke scores cannot enter formal aggregation")
    shared={(s["protocol"],json.dumps(s["coverage_sha256"],sort_keys=True),json.dumps(s["suite_id_sha256"],sort_keys=True)) for s in summaries.values()}
    if len(shared)!=1:raise RuntimeError("models were evaluated on different inputs/protocols")
    predictions={};effects={};rows=[]
    for method,folder in outputs.items():
        summary=summaries[method]
        expected={"zero_shot"} if method=="gpt2_native" else set(map(str,SEEDS))
        if set(summary["seeds"])!=expected:raise RuntimeError("missing seeds")
        for seed in expected:
            with gzip.open(folder/f"{method_slug(method)}_{seed}_test.jsonl.gz","rt") as stream:
                predictions[method,seed]=[json.loads(line) for line in stream]
        for suite in ("test","length","composition","beam10"):
            tasks=next(iter(summary["seeds"].values()))[suite]
            for task in tasks:
                values=[s[suite][task]["accuracy"] for s in summary["seeds"].values()]
                rows.append({"method":method,"role":"zero_shot_reference" if method=="gpt2_native" else "trained",
                             "suite":suite,"task":task,"mean_accuracy":float(np.mean(values)),
                             "seed_std":float(np.std(values,ddof=1)) if len(values)>1 else None})
    for proposed in spec["trained_methods"][:2]:
        for other in spec["methods"]:
            if proposed==other:continue
            for seed in SEEDS:
                native_seed="zero_shot" if other=="gpt2_native" else str(seed)
                for task in ("prim_fwd","prim_bwd","prim_ibp","ode_general","ode_ivp"):
                    key=f"{proposed}|{other}|{task}|{seed}"
                    effects[key]=paired_interval(predictions[proposed,str(seed)],predictions[other,native_seed],task,replicates=replicates)
    final={"phase":phase,"protocol":PROTOCOL,"methods":summaries,"paired_effects":effects,
           "interpretation":"training+tokenizer system vs untrained reference; not an isolated tokenizer effect" if phase=="phase1" else "equal LoRA-plus-new-embedding adaptation; vocabulary cost differs"}
    output.mkdir(parents=True,exist_ok=True)
    (output/"final_results.json").write_text(json.dumps(final,indent=2,ensure_ascii=False)+"\n")
    with (output/"metrics.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return final
