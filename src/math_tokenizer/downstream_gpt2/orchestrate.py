"""Assemble frozen data, audit it, and run resumable eight-way training."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from math_tokenizer.corpus import ROOT
from .adapters import METHODS, method_slug
from .data import OPERATORS, QUOTAS, UPSTREAM_SHA

SEEDS = (11, 23, 37)


def assemble(base: Path, ibp: Path, ode: Path, final: Path):
    final.mkdir(parents=True, exist_ok=True)
    origins = {"prim_fwd":base, "prim_bwd":base, "prim_ibp":ibp, "ode1":ode}
    manifest = {"upstream_repo":"facebookresearch/SymbolicMathematics",
                "upstream_sha":UPSTREAM_SHA, "upstream_license":"CC BY-NC 4.0",
                "operators":OPERATORS, "max_ops":6, "max_ops_G":3,
                "max_int":5, "ode_generator":"linear-in-constant analytical subfamily of upstream ODE1 construction",
                "quotas":QUOTAS, "files":{}}
    for split in ("train","valid","test"):
        for method in origins:
            name = f"{split}_{method}.jsonl"
            source = origins[method] / name
            if not source.exists():
                raise FileNotFoundError(source)
            expected = QUOTAS[split][method] * (2 if method == "ode1" else 1)
            count = sum(1 for _ in source.open())
            if count != expected:
                raise ValueError(f"{name}: expected {expected}, got {count}")
            destination = final / name
            if source.resolve() != destination.resolve():
                shutil.copy2(source,destination)
            manifest["files"][name] = {"rows":count,
                                       "sha256":hashlib.sha256(destination.read_bytes()).hexdigest()}
    (final / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def complete(path: Path):
    history = path / "history.json"
    return (path / "best/config.json").exists() and history.exists() and len(json.loads(history.read_text())) == 3


def train_all(data: Path, gpt2: Path, runs: Path, microbatch=2):
    for seed in SEEDS:
        for method in METHODS:
            destination = runs / method_slug(method) / str(seed)
            if complete(destination):
                print("SKIP complete",method,seed,flush=True)
                continue
            command = [sys.executable,"-m","math_tokenizer.downstream_gpt2.train",
                       "--method",method,"--seed",str(seed),"--data",str(data),
                       "--gpt2",str(gpt2),"--out",str(runs),"--epochs","3",
                       "--microbatch",str(microbatch),"--effective-batch","32"]
            print("RUN",method,seed,flush=True)
            subprocess.run(command,check=True,cwd=ROOT)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--base",type=Path,default=ROOT/"tmp/downstream_gpt2/data")
    parser.add_argument("--ibp",type=Path,default=ROOT/"tmp/downstream_gpt2/ibp_data")
    parser.add_argument("--ode",type=Path,default=ROOT/"tmp/downstream_gpt2/ode_data")
    parser.add_argument("--final",type=Path,default=ROOT/"tmp/downstream_gpt2/final_data")
    parser.add_argument("--gpt2",type=Path,default=ROOT/"tmp/gpt2-607a30d")
    parser.add_argument("--runs",type=Path,default=ROOT/"tmp/downstream_gpt2/runs")
    parser.add_argument("--microbatch",type=int,default=2)
    parser.add_argument("--assemble-only",action="store_true")
    args=parser.parse_args()
    assemble(args.base,args.ibp,args.ode,args.final)
    if args.assemble_only:
        print(args.final)
        return
    base=[sys.executable,"-m"]
    subprocess.run(base+["math_tokenizer.downstream_gpt2.audit","--data",str(args.final),
                         "--gpt2",str(args.gpt2)],check=True,cwd=ROOT)
    subprocess.run(base+["math_tokenizer.downstream_gpt2.train","--method",METHODS[0],
                         "--data",str(args.final),"--gpt2",str(args.gpt2),
                         "--freeze-coverage"],check=True,cwd=ROOT)
    train_all(args.final,args.gpt2,args.runs,args.microbatch)
    subprocess.run(base+["math_tokenizer.downstream_gpt2.results","--data",str(args.final),
                         "--gpt2",str(args.gpt2),"--runs",str(args.runs)],check=True,cwd=ROOT)


if __name__=="__main__":
    main()
