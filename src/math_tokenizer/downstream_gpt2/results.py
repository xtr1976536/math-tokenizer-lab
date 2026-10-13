"""Sealed-test evaluation, paired intervals, and method-level result tables."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from math_tokenizer.corpus import ROOT
from .adapters import METHODS, TokenizerAdapter, method_slug
from .train import evaluate, examples_for, read_split
from .verify import Verifier

SEEDS = (11, 23, 37)


def completed_runs(folder: Path):
    for method in METHODS:
        for seed in SEEDS:
            path = folder / method_slug(method) / str(seed)
            if not (path / "best/config.json").exists() or not (path / "history.json").exists():
                return False
            if len(json.loads((path / "history.json").read_text())) != 3:
                return False
    return True


def stratified_beam_ids(rows, count_per_group=200):
    groups = defaultdict(list)
    for row in rows:
        key = row["method"] if row["task"] == "integral" else row["task"]
        groups[key].append(row)
    chosen = []
    for key, group in sorted(groups.items()):
        chosen.extend(row["id"] for row in sorted(group, key=lambda r: r["id"])[:count_per_group])
    return set(chosen)


def ood_sets(folder: Path, gpt2: Path):
    output = {}
    for mode in ("length", "composition"):
        manifest = folder / f"manifest_ood_{mode}.json"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        rows = []
        for method in ("prim_fwd", "prim_bwd", "prim_ibp", "ode1"):
            path = folder / f"ood_{mode}_{method}.jsonl"
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
        kept = {row["id"] for row in rows}
        for name in METHODS:
            adapter = TokenizerAdapter(name, gpt2)
            for row in rows:
                prompt, answer = adapter.row(row["task"], row["input_latex"], row["target_latex"])
                if len(prompt) > 512 or len(prompt)+len(answer) > 1024:
                    kept.discard(row["id"])
        selected = [row for row in rows if row["id"] in kept]
        output[mode] = {"rows": selected, "count": len(selected),
                        "sha256": hashlib.sha256("\n".join(sorted(kept)).encode()).hexdigest()}
    return output


def score_table(predictions):
    buckets = defaultdict(list)
    for row in predictions:
        key = row["method"] if row["task"] == "integral" else row["task"]
        buckets[key].append(row)
    return {key: {"n": len(rows), "accuracy": sum(r["correct"] for r in rows)/len(rows),
                  "invalid_rate": sum(all(v in ("invalid", "timeout", "truncated") for v in r["verdicts"])
                                      for r in rows)/len(rows),
                  "mean_latency_seconds": sum(r["latency_seconds"] for r in rows)/len(rows)}
            for key, rows in buckets.items()}


def paired_interval(left, right, key, replicates=2000, seed=20261013):
    by_left = {r["id"]: r for r in left}
    by_right = {r["id"]: r for r in right}
    ids = sorted(set(by_left) & set(by_right))
    assert len(ids) == len(by_left) == len(by_right)
    grouped = defaultdict(list)
    for item in ids:
        row = by_left[item]
        label = row["method"] if row["task"] == "integral" else row["task"]
        if label == key:
            grouped[row["group"]].append(item)
    groups = list(grouped)
    if not groups:
        raise ValueError(key)
    def effect(chosen):
        a = [int(by_left[i]["correct"]) for g in chosen for i in grouped[g]]
        b = [int(by_right[i]["correct"]) for g in chosen for i in grouped[g]]
        return float(np.mean(a)-np.mean(b))
    rng = random.Random(seed)
    samples = [effect([rng.choice(groups) for _ in groups]) for _ in range(replicates)]
    return {"difference": effect(groups), "ci95": [float(np.quantile(samples, .025)),
                                                     float(np.quantile(samples, .975))],
            "groups": len(groups)}


def run(data: Path, gpt2: Path, runs: Path, output: Path, device_name="mps"):
    if not completed_runs(runs):
        raise RuntimeError("all 8 methods and 3 seeds must finish before sealed test is opened")
    coverage = json.loads((data / "coverage_512.json").read_text())
    rows = read_split(data, "test")
    allowed = set(coverage["kept"]["test"])
    rows = [row for row in rows if row["id"] in allowed]
    beam_ids = stratified_beam_ids(rows)
    ood = ood_sets(data.parent / "ood_data", gpt2)
    verifier = Verifier(ROOT / "tmp/SymbolicMathematics")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"methods": {}, "beam_ids": sorted(beam_ids),
               "coverage_sha256": coverage["sha256"]["test"],
               "ood_coverage": {mode:{"count":value["count"],"sha256":value["sha256"]}
                                for mode,value in ood.items()}}
    all_predictions = {}
    for method in METHODS:
        for seed in SEEDS:
            adapter = TokenizerAdapter(method, gpt2)
            model_path = runs / method_slug(method) / str(seed) / "best"
            model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
            model.to(device_name).eval()
            masked = adapter.masked_ids()
            examples = examples_for(adapter, rows, allowed)
            scores, predictions = evaluate(model, adapter, examples, verifier, masked, device_name)
            beam_examples = [e for e in examples if e[0]["id"] in beam_ids]
            beam_scores, beam_predictions = evaluate(model, adapter, beam_examples, verifier, masked, device_name, beams=10)
            artifact = output / f"{method_slug(method)}_{seed}.jsonl.gz"
            with gzip.open(artifact, "wt") as stream:
                for item in predictions:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            beam_artifact = output / f"{method_slug(method)}_{seed}_beam10.jsonl.gz"
            with gzip.open(beam_artifact, "wt") as stream:
                for item in beam_predictions:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            ood_tables = {}
            for mode, value in ood.items():
                ood_examples = examples_for(adapter, value["rows"], {r["id"] for r in value["rows"]})
                _ood_scores, ood_predictions = evaluate(model, adapter, ood_examples, verifier,
                                                         masked, device_name)
                ood_tables[mode] = score_table(ood_predictions)
                with gzip.open(output / f"{method_slug(method)}_{seed}_ood_{mode}.jsonl.gz", "wt") as stream:
                    for item in ood_predictions:
                        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            summary["methods"].setdefault(method, {})[str(seed)] = {
                "top1": score_table(predictions), "beam10": score_table(beam_predictions),
                "ood": ood_tables,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "validation_selection": json.loads((runs / method_slug(method) / str(seed) / "best_meta.json").read_text())["validation"]["selection"]}
            all_predictions[(method, seed)] = predictions
            (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
            print(method, seed, scores, beam_scores, flush=True)
            del model
            if device_name == "mps":
                torch.mps.empty_cache()
    comparisons = {}
    for seed in SEEDS:
        for proposed in METHODS[:2]:
            for baseline in METHODS[2:7]:
                for task in ("prim_fwd", "prim_bwd", "prim_ibp", "ode_ivp", "ode_general"):
                    name = f"{proposed}|{baseline}|{task}|{seed}"
                    comparisons[name] = paired_interval(all_predictions[(proposed,seed)],
                                                         all_predictions[(baseline,seed)],task)
    (output / "paired_effects.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    return summary


def evaluate_method(method: str, data: Path, gpt2: Path, run_folder: Path,
                    output: Path, device_name="cuda"):
    if method not in METHODS:
        raise ValueError(method)
    coverage = json.loads((data/"coverage_512.json").read_text())
    allowed = set(coverage["kept"]["test"])
    rows = [r for r in read_split(data,"test") if r["id"] in allowed]
    beam_ids = stratified_beam_ids(rows)
    ood = ood_sets(data.parent/"ood_data",gpt2)
    verifier = Verifier(ROOT/"tmp/SymbolicMathematics")
    output.mkdir(parents=True,exist_ok=True)
    summary = {"method":method,"coverage_sha256":coverage["sha256"]["test"],
               "beam_ids_sha256":hashlib.sha256("\n".join(sorted(beam_ids)).encode()).hexdigest(),
               "ood_coverage":{mode:value["sha256"] for mode,value in ood.items()},
               "seeds":{}}
    for seed in SEEDS:
        folder = run_folder/str(seed)
        if not (folder/"best/config.json").exists():
            raise FileNotFoundError(folder/"best/config.json")
        adapter = TokenizerAdapter(method,gpt2)
        model = GPT2LMHeadModel.from_pretrained(folder/"best",local_files_only=True).to(device_name).eval()
        masked = adapter.masked_ids()
        examples = examples_for(adapter,rows,allowed)
        _score, predictions = evaluate(model,adapter,examples,verifier,masked,device_name)
        beam = [e for e in examples if e[0]["id"] in beam_ids]
        _beam_score, beam_predictions = evaluate(model,adapter,beam,verifier,masked,device_name,beams=10)
        tables = {"top1":score_table(predictions),"beam10":score_table(beam_predictions),"ood":{}}
        for name,items in (("test",predictions),("beam10",beam_predictions)):
            with gzip.open(output/f"{method_slug(method)}_{seed}_{name}.jsonl.gz","wt") as stream:
                for item in items:
                    stream.write(json.dumps(item,ensure_ascii=False)+"\n")
        for mode,value in ood.items():
            ood_examples = examples_for(adapter,value["rows"],{r["id"] for r in value["rows"]})
            _s,odecisions = evaluate(model,adapter,ood_examples,verifier,masked,device_name)
            tables["ood"][mode] = score_table(odecisions)
            with gzip.open(output/f"{method_slug(method)}_{seed}_{mode}.jsonl.gz","wt") as stream:
                for item in odecisions:
                    stream.write(json.dumps(item,ensure_ascii=False)+"\n")
        tables["parameter_count"] = sum(p.numel() for p in model.parameters())
        tables["train_history"] = json.loads((folder/"history.json").read_text())
        summary["seeds"][str(seed)] = tables
        (output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
        print(method,seed,tables["top1"],flush=True)
        del model
        if device_name == "cuda":
            torch.cuda.empty_cache()
    return summary


def aggregate_method_outputs(method_outputs: dict[str,Path], output: Path):
    if set(method_outputs) != set(METHODS):
        raise ValueError("all eight method outputs are required")
    output.mkdir(parents=True,exist_ok=True)
    summaries = {name:json.loads((folder/"summary.json").read_text())
                 for name,folder in method_outputs.items()}
    coverage = {summary["coverage_sha256"] for summary in summaries.values()}
    beam = {summary["beam_ids_sha256"] for summary in summaries.values()}
    ood = {json.dumps(summary["ood_coverage"],sort_keys=True) for summary in summaries.values()}
    if len(coverage)!=1 or len(beam)!=1 or len(ood)!=1:
        raise RuntimeError("evaluation inputs differ across methods")
    predictions = {}
    for name,folder in method_outputs.items():
        for seed in SEEDS:
            path=folder/f"{method_slug(name)}_{seed}_test.jsonl.gz"
            with gzip.open(path,"rt") as stream:
                predictions[name,seed]=[json.loads(line) for line in stream]
    effects = {}
    for seed in SEEDS:
        for proposed in METHODS[:2]:
            for baseline in METHODS[2:7]:
                for task in ("prim_fwd","prim_bwd","prim_ibp","ode_general","ode_ivp"):
                    key=f"{proposed}|{baseline}|{task}|{seed}"
                    effects[key]=paired_interval(predictions[proposed,seed],predictions[baseline,seed],task)
    final={"input_coverage_sha256":next(iter(coverage)),"beam_subset_sha256":next(iter(beam)),
           "methods":summaries,"paired_effects":effects}
    (output/"final_results.json").write_text(json.dumps(final,indent=2,ensure_ascii=False)+"\n")
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "tmp/downstream_gpt2/data")
    parser.add_argument("--gpt2", type=Path, default=ROOT / "tmp/gpt2-607a30d")
    parser.add_argument("--runs", type=Path, default=ROOT / "tmp/downstream_gpt2/runs")
    parser.add_argument("--out", type=Path, default=ROOT / "tmp/downstream_gpt2/results")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    run(args.data,args.gpt2,args.runs,args.out,args.device)


if __name__ == "__main__":
    main()
