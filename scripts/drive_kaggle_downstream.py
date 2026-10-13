"""Resumable submission order for the private GPT-2 Kaggle experiment."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
KERNELS = BASE / "tmp/kaggle_downstream_gpt2/kernels"
STATE = BASE / "tmp/kaggle_downstream_gpt2/driver_state.json"
METHODS = ("ast-boundary","atomic-bpe","byte-bpe","sp-unigram","qwen25","llama3","o200k","gpt2-native")


def command(*args):
    return subprocess.run(("kaggle",*args),text=True,capture_output=True)


def metadata(folder):
    return json.loads((folder/"kernel-metadata.json").read_text())


def status(ref):
    result=command("kernels","status",ref)
    if result.returncode:
        return "MISSING"
    line=result.stdout+result.stderr
    for state in ("COMPLETE","ERROR","CANCELLED","RUNNING","QUEUED","PENDING"):
        if f"KernelWorkerStatus.{state}" in line:
            return state
    raise RuntimeError(f"unknown status for {ref}: {line.strip()}")


def save(event):
    STATE.parent.mkdir(parents=True,exist_ok=True)
    old=json.loads(STATE.read_text()) if STATE.exists() else []
    old.append({"time":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**event})
    STATE.write_text(json.dumps(old,indent=2)+"\n")
    print(event,flush=True)


def wait_kernel(folder,allow_existing=True):
    ref=metadata(folder)["id"]
    current=status(ref)
    if current=="MISSING":
        result=command("kernels","push","-p",str(folder))
        if result.returncode:
            raise RuntimeError(f"push {ref}: {result.stdout} {result.stderr}")
        save({"event":"submitted","kernel":ref})
    elif current=="COMPLETE" and allow_existing:
        save({"event":"reused_complete","kernel":ref})
        return
    elif current in ("ERROR","CANCELLED"):
        raise RuntimeError(f"existing kernel {ref} has status {current}; inspect before a new version")
    while True:
        current=status(ref)
        if current=="COMPLETE":
            save({"event":"completed","kernel":ref})
            return
        if current in ("ERROR","CANCELLED"):
            save({"event":"failed","kernel":ref,"status":current})
            raise RuntimeError(f"{ref} ended with {current}")
        time.sleep(45)


def download(ref,folder,pattern):
    folder.mkdir(parents=True,exist_ok=True)
    result=command("kernels","output",ref,"-p",str(folder),"--file-pattern",pattern)
    if result.returncode:
        raise RuntimeError(f"output {ref}: {result.stdout} {result.stderr}")


def required_json(folder,name):
    matches=list(folder.rglob(name))
    if len(matches)!=1:
        raise RuntimeError(f"expected one {name} under {folder}, found {matches}")
    return json.loads(matches[0].read_text())


def run(through):
    wait_kernel(KERNELS/"smoke")
    smoke="xiaotongrenkk/math-tokenizer-gpt2-smoke-v1"
    download(smoke,BASE/"tmp/kaggle_downstream_gpt2/downloads/smoke",
             r"smoke_audit.json|gpu_memory_preflight.json")
    smoke_folder=BASE/"tmp/kaggle_downstream_gpt2/downloads/smoke"
    if len(required_json(smoke_folder,"gpu_memory_preflight.json"))!=3:
        raise RuntimeError("GPU smoke did not check all three large vocabularies")
    if through=="smoke":
        return
    wait_kernel(KERNELS/"data")
    data_ref="xiaotongrenkk/math-tokenizer-gpt2-data-v1"
    data_folder=BASE/"tmp/kaggle_downstream_gpt2/downloads/data"
    download(data_ref,data_folder,r"data_audit.json")
    data_audit=required_json(data_folder,"data_audit.json")
    if data_audit["round_1"]["oracle_unverified"]["test"]:
        raise RuntimeError("frozen task test has unverified reference answers")
    if through=="data":
        return
    for method in METHODS:
        wait_kernel(KERNELS/"train"/method)
    if through=="train":
        return
    for method in METHODS:
        wait_kernel(KERNELS/"eval"/method)
    if through=="eval":
        return
    wait_kernel(KERNELS/"aggregate")
    download("xiaotongrenkk/math-tokenizer-gpt2-aggregate-v1",
             BASE/"tmp/kaggle_downstream_gpt2/downloads/final",r"final_results.json")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--through",choices=("smoke","data","train","eval","aggregate"),default="aggregate")
    args=parser.parse_args()
    run(args.through)
