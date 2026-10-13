"""Assemble frozen data, audit it, and run resumable eight-way training."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter,defaultdict
from pathlib import Path

from math_tokenizer.corpus import ROOT
from .adapters import METHODS, method_slug
from .data import OPERATORS, QUOTAS, UPSTREAM_SHA, row_template, upstream

SEEDS = (11, 23, 37)


def assemble(base: Path, ibp: Path, ode: Path, final: Path, quotas=QUOTAS,
             upstream_source: Path = ROOT/"tmp/SymbolicMathematics"):
    final.mkdir(parents=True, exist_ok=True)
    origins = {"prim_fwd":base, "prim_bwd":base, "prim_ibp":ibp, "ode1":ode}
    env=upstream(upstream_source)
    manifest = {"upstream_repo":"facebookresearch/SymbolicMathematics",
                "upstream_sha":UPSTREAM_SHA, "upstream_license":"CC BY-NC 4.0",
                "operators":OPERATORS, "max_ops":6, "max_ops_G":3,
                "max_int":5, "ode_generator":"linear-in-constant analytical subfamily of upstream ODE1 construction",
                "quotas":quotas, "split_rule":"disjoint canonical source operator templates",
                "files":{}}
    grouped=defaultdict(list)
    for split in ("train","valid","test"):
        for method in origins:
            name = f"{split}_{method}.jsonl"
            source = origins[method] / name
            if not source.exists():
                raise FileNotFoundError(source)
            expected = quotas[split][method] * (2 if method == "ode1" else 1)
            rows=[json.loads(line) for line in source.read_text().splitlines()]
            count = len(rows)
            if count != expected:
                raise ValueError(f"{name}: expected {expected}, got {count}")
            for row in rows:
                grouped[row_template(env,row)].append(row)
    target={(split,method):count*(2 if method=="ode1" else 1)
            for split,methods in quotas.items() for method,count in methods.items()}
    assigned=Counter()
    placements={}
    for template,rows in sorted(grouped.items()):
        counts=Counter(row["method"] for row in rows)
        def score(split):
            pressure=sum((assigned[split,method]+count)/target[split,method]
                         for method,count in counts.items())/len(counts)
            overflow=sum(max(0,assigned[split,method]+count-target[split,method])/target[split,method]
                         for method,count in counts.items())
            return pressure+10*overflow
        chosen=min(("train","valid","test"),key=score)
        for method,count in counts.items():
            assigned[chosen,method]+=count
        placements[template]=chosen
    # Deterministic local balancing keeps whole templates intact while matching
    # the requested per-task split sizes as closely as the group sizes permit.
    for _ in range(10):
        moved=0
        for template,rows in sorted(grouped.items()):
            counts=Counter(row["method"] for row in rows)
            current=placements[template]
            def cell_error(split,method,count):
                return (count-target[split,method])**2/target[split,method]
            best_delta,best_split=0.0,current
            for candidate in ("train","valid","test"):
                if candidate==current:
                    continue
                delta=0.0
                for method,count in counts.items():
                    delta += cell_error(current,method,assigned[current,method]-count)-cell_error(current,method,assigned[current,method])
                    delta += cell_error(candidate,method,assigned[candidate,method]+count)-cell_error(candidate,method,assigned[candidate,method])
                if delta<best_delta-1e-9:
                    best_delta,best_split=delta,candidate
            if best_split!=current:
                for method,count in counts.items():
                    assigned[current,method]-=count
                    assigned[best_split,method]+=count
                placements[template]=best_split
                moved+=1
        if not moved:
            break
    output=defaultdict(list)
    for template,rows in sorted(grouped.items()):
        chosen=placements[template]
        for row in rows:
            item=dict(row)
            item["origin_id"]=row["id"]
            item["split"]=chosen
            item["id"]=f"{chosen}:{row['method']}:{hashlib.sha256(row['id'].encode()).hexdigest()[:20]}"
            item["template_group"]=template
            output[chosen,row["method"]].append(item)
    for split in ("train","valid","test"):
        for method in origins:
            name=f"{split}_{method}.jsonl"
            destination=final/name
            with destination.open("w") as stream:
                for row in sorted(output[split,method],key=lambda r:r["id"]):
                    stream.write(json.dumps(row,ensure_ascii=False)+"\n")
            manifest["files"][name]={"rows":len(output[split,method]),
                                      "sha256":hashlib.sha256(destination.read_bytes()).hexdigest()}
    manifest["actual_rows"]={split:{method:assigned[split,method] for method in origins}
                             for split in ("train","valid","test")}
    manifest["quota_deviation"]={split:{method:assigned[split,method]-target[split,method]
                                         for method in origins}
                                 for split in ("train","valid","test")}
    for key,planned in target.items():
        if abs(assigned[key]-planned)>int(planned*.005):
            raise RuntimeError(f"template-disjoint split misses quota by more than 0.5%: {key}, {assigned[key]} vs {planned}")
    manifest["template_groups"]=len(grouped)
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
