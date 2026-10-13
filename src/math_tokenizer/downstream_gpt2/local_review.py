"""Four required local smoke reviews before any Kaggle run."""
from __future__ import annotations

import compileall
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from math_tokenizer.corpus import ROOT
from .adapters import METHODS, TokenizerAdapter, method_slug
from .audit import dataset_audit, tokenizer_audit
from .data import UPSTREAM_SHA, generate
from .ood import generate as generate_ood
from .orchestrate import assemble
from .train import common_coverage, examples_for, generate_one, read_split, train_one
from .verify import Verifier


def run_review(output=ROOT/"tmp/downstream_gpt2/local_review"):
    output.mkdir(parents=True,exist_ok=True)
    source = ROOT/"tmp/SymbolicMathematics"
    gpt2 = ROOT/"tmp/gpt2-607a30d"
    rounds = {}

    # Round 1: code, dependencies, pinned source and independent math unit tests.
    assert compileall.compile_dir(str(ROOT/"src/math_tokenizer/downstream_gpt2"),quiet=1)
    subprocess.run([sys.executable,"-m","pytest","-q",str(ROOT/"tests/test_downstream_gpt2.py")],
                   cwd=ROOT,check=True)
    assert subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()==UPSTREAM_SHA
    assert (gpt2/"model.safetensors").exists()
    rounds["round_1_static"]={"status":"passed","source_sha":UPSTREAM_SHA,
                              "gpt2_weight_sha256":hashlib.sha256((gpt2/"model.safetensors").read_bytes()).hexdigest()}
    (output/"round_1.json").write_text(json.dumps(rounds["round_1_static"],indent=2)+"\n")

    # Round 2: regenerate a tiny frozen dataset, verify every oracle, and audit every adapter.
    raw_data=output/"raw_data"
    generate(source,raw_data,.001)
    data=output/"data"
    pilot_quotas=json.loads((raw_data/"manifest.json").read_text())["quotas"]
    assemble(raw_data,raw_data,raw_data,data,pilot_quotas,source)
    rounds["round_2_data"]={"dataset":dataset_audit(data,source),
                            "tokenizers":tokenizer_audit(data,gpt2)}
    coverage=common_coverage(data,gpt2)
    (data/"coverage_512.json").write_text(json.dumps(coverage,indent=2)+"\n")
    assert all(coverage["kept"][split] for split in ("train","valid","test"))
    ood=output/"ood_data"
    for mode in ("length","composition"):
        generate_ood(mode,source,data,ood,.002)
    rounds["round_2_data"]["coverage"]={k:len(v) for k,v in coverage["kept"].items()}
    (output/"round_2.json").write_text(json.dumps(rounds["round_2_data"],indent=2)+"\n")

    # Round 3: all eight vocabularies undergo an actual optimizer step, save, reload, and generate.
    run_dir=output/"runs"
    histories={}
    sample=read_split(data,"valid")[:1]
    verifier=Verifier(source)
    for method in METHODS:
        result=train_one(method,11,data,gpt2,run_dir,epochs=1,microbatch=2,
                         effective_batch=4,eval_limit=5,device_name="mps")
        history=json.loads((result/"history.json").read_text())
        assert len(history)==1 and history[0]["train_loss_per_example"]>0
        config=json.loads((result/"config.json").read_text())
        model=GPT2LMHeadModel.from_pretrained(result/"best",local_files_only=True).to("mps").eval()
        adapter=TokenizerAdapter(method,gpt2)
        example=examples_for(adapter,sample,{sample[0]["id"]})[0]
        prediction=generate_one(model,adapter,example[1],adapter.masked_ids(),"mps",max_new_tokens=16)
        assert prediction and "text" in prediction[0]
        histories[method]={"loss":history[0]["train_loss_per_example"],
                           "embedding_sha256":config["embedding_sha256"],
                           "parameters":config["parameters"],
                           "generation_status":verifier.verify(example[0],prediction[0]["text"]) if prediction[0]["stopped"] else "truncated"}
        del model
        torch.mps.empty_cache()
    assert histories["ast_boundary"]["embedding_sha256"]==histories["official_atomic_bpe"]["embedding_sha256"]
    rounds["round_3_training"]={"status":"passed","methods":histories}
    (output/"round_3.json").write_text(json.dumps(rounds["round_3_training"],indent=2)+"\n")

    # Round 4: replay representative small/full vocabularies from the same frozen inputs.
    replay={}
    for method in ("ast_boundary","Qwen2.5-0.5B"):
        result=train_one(method,11,data,gpt2,output/"replay_runs",epochs=1,
                         microbatch=2,effective_batch=4,eval_limit=5,device_name="mps")
        loss=json.loads((result/"history.json").read_text())[0]["train_loss_per_example"]
        prior=histories[method]["loss"]
        difference=abs(loss-prior)
        assert difference < 1e-3,(method,loss,prior)
        replay[method]={"first_loss":prior,"replay_loss":loss,"absolute_difference":difference}
    rounds["round_4_replay"]={"status":"passed","methods":replay}
    (output/"round_4.json").write_text(json.dumps(rounds["round_4_replay"],indent=2)+"\n")
    (output/"summary.json").write_text(json.dumps(rounds,indent=2,ensure_ascii=False)+"\n")
    return rounds


if __name__=="__main__":
    run_review()
