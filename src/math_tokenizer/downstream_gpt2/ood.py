"""Controlled long-form and unseen-composition tests from new generator seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import sympy as sp

from math_tokenizer.corpus import ROOT
from .data import METHODS, expressions, gen_ode1_linear, latex, make_rows, upstream
from .verify import Verifier

OOD_COUNTS = {"prim_fwd": 500, "prim_bwd": 500, "prim_ibp": 500, "ode1": 1000}


def transformed_ode(source, h, x, f):
    y = sp.Dummy("Y")
    derivative = sp.Dummy("D")
    z = sp.Dummy("Z")
    frozen = source.xreplace({sp.Derivative(f, x): derivative, f: y, x: z})
    return sp.factor(frozen.xreplace({z: h, y: f, derivative: sp.Derivative(f, x)/sp.diff(h, x)}))


def transform(env, method, pair, mode, serial, choice_rng):
    source, answer = expressions(env, method, pair)
    x = env.variables["x"]
    h = x**3 + x + 1 if mode == "length" else sp.exp(sp.sin(x))
    group_mode = "ood_" + mode
    if method == "ode1":
        f = env.functions["f"](x)
        transformed = transformed_ode(source, h, x, f)
        target = answer.subs(x, h)
        base = make_rows(env, method, pair, group_mode, serial, choice_rng)
        group = hashlib.sha256(sp.srepr(transformed).encode()).hexdigest()
        result = []
        for row in base:
            other = dict(row)
            other["group"] = group
            other["input_latex"] = latex(transformed) + "=0"
            other["source_infix"] = str(transformed)
            if row["task"] == "ode_ivp":
                target_particular = sp.simplify(target.subs(env.coefficients["a8"],row["constant"]))
                y0 = sp.simplify(target_particular.subs(x,row["x0"]))
                if y0.is_finite is not True or y0.is_real is not True or len(latex(y0)) > 48:
                    raise ValueError("irregular transformed IVP")
                other["target_latex"] = latex(target_particular)
                other["y0_latex"] = latex(y0)
                other["input_latex"] += r",\quad f(" + str(row["x0"]) + ")=" + latex(y0)
            else:
                other["target_latex"] = latex(target)
            result.append(other)
        return result
    transformed = sp.factor(source.subs(x,h)*sp.diff(h,x))
    target = answer.subs(x,h)
    group = hashlib.sha256(sp.srepr(transformed).encode()).hexdigest()
    return [{"id": f"{group_mode}:{method}:{serial}", "group": group,
             "task": "integral", "method": method, "split": group_mode,
             "input_latex": latex(transformed), "target_latex": latex(target),
             "source_infix": str(transformed), "base_source_prefix": pair[0],
             "target_prefix": pair[1]}]


def generate(mode: str, source: Path, main_data: Path, output: Path, scale=1.0):
    if mode not in ("length", "composition"):
        raise ValueError(mode)
    env = upstream(source)
    verifier = Verifier(source)
    output.mkdir(parents=True, exist_ok=True)
    existing = set()
    for path in main_data.glob("*.jsonl"):
        for line in path.open():
            existing.add(json.loads(line)["group"])
    manifest = {"mode": mode, "generator_sha": "4596d070e1a9a1c2239c923d7d68fda577c8c007",
                "transformation": "x**3+x+1" if mode == "length" else "exp(sin(x))",
                "files": {}, "train_length_p90_thresholds":{}}
    for method_index, method in enumerate(METHODS):
        seed = 6200 + method_index + (0 if mode == "length" else 100)
        rng = np.random.RandomState(seed)
        choice_rng = random.Random(seed)
        if method == "prim_ibp":
            env.PRIM_CACHE, env.PRIM_COUNT = {}, [0,0]
        fn = {"prim_fwd":env.gen_prim_fwd,
              "prim_bwd":lambda r:env.gen_prim_bwd(r,True),
              "prim_ibp":env.gen_prim_ibp,
              "ode1":lambda r:gen_ode1_linear(env,r)}[method]
        target = max(1,round(OOD_COUNTS[method]*scale))
        training = [json.loads(line) for line in (main_data/f"train_{method}.jsonl").read_text().splitlines()]
        length_threshold = max(80,int(np.quantile([len(row["input_latex"].encode()) for row in training],.9)))
        manifest["train_length_p90_thresholds"][method] = length_threshold
        accepted, attempted, rejected, last_print = 0, 0, defaultdict(int), -1
        path = output / f"ood_{mode}_{method}.jsonl"
        with path.open("w") as stream:
            while accepted < target:
                attempted += 1
                if attempted > target * 180:
                    raise RuntimeError(f"OOD {mode}/{method} stalled at {accepted}/{target}; {dict(rejected)}")
                try:
                    pair = fn(rng)
                    if pair is None:
                        rejected["generator_none"] += 1
                        continue
                    rows = transform(env, method, pair, mode, accepted, choice_rng)
                    if rows[0]["group"] in existing:
                        rejected["duplicate_group"] += 1
                        continue
                    if any(verifier.verify(row,row["target_latex"]) != "correct" for row in rows):
                        rejected["oracle_unverified"] += 1
                        continue
                    if mode == "length" and len(rows[0]["input_latex"].encode()) <= length_threshold:
                        rejected["too_short"] += 1
                        continue
                    existing.add(rows[0]["group"])
                    for row in rows:
                        stream.write(json.dumps(row,ensure_ascii=False)+"\n")
                    accepted += 1
                except Exception as exc:
                    rejected[type(exc).__name__] += 1
                except BaseException as exc:
                    if type(exc).__name__ != "TimeoutError":
                        raise
                    rejected["upstream_timeout"] += 1
                if accepted and accepted % 100 == 0 and accepted != last_print:
                    print(mode,method,accepted,target,"attempts",attempted,flush=True)
                    last_print = accepted
        manifest["files"][path.name] = {"rows": target*(2 if method=="ode1" else 1),
                                         "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                                         "attempts":attempted,"rejected":dict(rejected)}
        print(mode,method,accepted,"attempts",attempted,flush=True)
    (output / f"manifest_ood_{mode}.json").write_text(json.dumps(manifest,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("length","composition"),required=True)
    parser.add_argument("--source",type=Path,default=ROOT/"tmp/SymbolicMathematics")
    parser.add_argument("--main-data",type=Path,default=ROOT/"tmp/downstream_gpt2/data")
    parser.add_argument("--out",type=Path,default=ROOT/"tmp/downstream_gpt2/data")
    parser.add_argument("--scale",type=float,default=1.0)
    args=parser.parse_args()
    generate(args.mode,args.source,args.main_data,args.out,args.scale)


if __name__=="__main__":
    main()
