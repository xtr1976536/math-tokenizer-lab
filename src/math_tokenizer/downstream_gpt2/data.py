"""Generate LaTeX task pairs with the pinned SymbolicMathematics generator.

The upstream generator remains unmodified. Its CC BY-NC 4.0 license applies
to this derived academic dataset; every record retains generator provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from math_tokenizer.corpus import ROOT

UPSTREAM_SHA = "4596d070e1a9a1c2239c923d7d68fda577c8c007"
OPERATORS = "add:10,sub:3,mul:10,div:5,pow2:4,pow3:2,sin:4,cos:4,exp:3,ln:2"
METHODS = ("prim_fwd", "prim_bwd", "prim_ibp", "ode1")
QUOTAS = {
    "train": {"prim_fwd": 8000, "prim_bwd": 8000, "prim_ibp": 8000, "ode1": 12000},
    "valid": {"prim_fwd": 600, "prim_bwd": 600, "prim_ibp": 600, "ode1": 900},
    "test": {"prim_fwd": 2000, "prim_bwd": 2000, "prim_ibp": 2000, "ode1": 3000},
}


def latex(expr):
    # Explicit multiplication avoids the LaTeX parser reading x(x+1) as a function.
    return sp.latex(expr, mul_symbol="dot")


def upstream(source: Path):
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_SHA:
        raise RuntimeError(f"expected upstream {UPSTREAM_SHA}, got {actual}")
    sys.path.insert(0, str(source))
    module = importlib.import_module("src.envs.char_sp")
    parser = argparse.ArgumentParser(add_help=False)
    module.CharSPEnvironment.register_args(parser)
    args = parser.parse_args([])
    args.max_ops, args.max_ops_G, args.max_int = 6, 3, 5
    args.max_len, args.operators = 512, OPERATORS
    return module.CharSPEnvironment(args)


def expressions(env, method, pair):
    left, right = pair
    if method != "ode1":
        if left[:2] != ["sub", "Y'"]:
            raise ValueError("unknown integral source prefix")
        left = left[2:]
    def convert(tokens):
        infix = env.prefix_to_infix(env.unclean_prefix(tokens))
        return parse_expr(infix, local_dict=env.local_dict)
    return convert(left), convert(right)


def template_signature(expr):
    """Operator tree with numerical leaves grouped into coarse value classes."""
    if expr.is_Integer:
        return "ZERO" if expr == 0 else "UNIT" if abs(expr) == 1 else "INT"
    if expr.is_Rational:
        return "RATIONAL"
    if expr.is_Symbol:
        return "X" if expr.name == "x" else "CONSTANT"
    return expr.func.__name__ + "(" + ",".join(template_signature(arg) for arg in expr.args) + ")"


def row_template(env, row):
    tokens = row["source_prefix"]
    infix = env.prefix_to_infix(env.unclean_prefix(tokens))
    expr = parse_expr(infix, local_dict=env.local_dict)
    return hashlib.sha256(template_signature(expr).encode()).hexdigest()


def gen_ode1_linear(env, rng):
    """Lample–Charton ODE1 construction restricted to f(x,c)=c*g(x)+h(x).

    This preserves their analytical elimination of the integration constant but
    avoids SymPy's occasionally unbounded solve(c) call in the upstream code.
    Both g and h are sampled by the pinned upstream expression generator.
    """
    x = env.variables["x"]
    f = env.functions["f"](x)
    c = env.coefficients["a8"]
    try:
        g_prefix = env._generate_expr(rng.randint(1, 4), env.max_int, rng, require_x=True)
        h_prefix = env._generate_expr(rng.randint(1, 4), env.max_int, rng)
        g = env.infix_to_sympy(env.prefix_to_infix(g_prefix))
        h = env.infix_to_sympy(env.prefix_to_infix(h_prefix))
        if g == 0 or x not in (g.free_symbols | h.free_symbols):
            return None
        solution = sp.expand(c*g+h)
        equation = sp.expand(g*sp.diff(f,x)-sp.diff(g,x)*f-g*sp.diff(h,x)+h*sp.diff(g,x))
        if sp.simplify(equation.subs(f,solution).doit()) != 0:
            return None
        eq_prefix = env.clean_prefix(env.sympy_to_prefix(equation))
        sol_prefix = env.clean_prefix(env.sympy_to_prefix(solution))
        if max(len(eq_prefix),len(sol_prefix)) > env.max_len-2:
            return None
        return eq_prefix, sol_prefix
    except Exception:
        return None


def make_rows(env, method: str, pair, split: str, serial: int, rng: random.Random):
    source, answer = expressions(env, method, pair)
    x = env.variables["x"]
    if method == "ode1":
        f = env.functions["f"](x)
        if sp.simplify(source.subs(f, answer).doit()) != 0:
            raise ValueError("source ODE solution failed substitution")
        base = latex(source) + "=0"
        key = hashlib.sha256(sp.srepr(source).encode()).hexdigest()
        general = {"id": f"{split}:ode1:{serial}:general", "group": key,
                   "task": "ode_general", "method": method, "split": split,
                   "input_latex": base, "target_latex": latex(answer),
                   "source_prefix": pair[0], "target_prefix": pair[1]}
        a8 = env.coefficients["a8"]
        if a8 not in answer.free_symbols:
            raise ValueError("ODE solution lacks integration constant")
        for _ in range(8):
            x0 = rng.choice((-1, 0, 1))
            c = rng.choice((-2, -1, 1, 2))
            particular = sp.simplify(answer.subs(a8, c))
            y0 = sp.simplify(particular.subs(x, x0))
            if y0.is_finite is True and y0.is_real is True and len(latex(y0)) <= 48:
                break
        else:
            raise ValueError("could not choose regular initial condition")
        if sp.simplify(source.subs(f, particular).doit()) != 0 or sp.simplify(particular.subs(x, x0)-y0) != 0:
            raise ValueError("initial value solution failed")
        ivp = {"id": f"{split}:ode1:{serial}:ivp", "group": key,
               "task": "ode_ivp", "method": method, "split": split,
               "input_latex": base + r",\quad f(" + str(x0) + ")=" + latex(y0),
               "target_latex": latex(particular), "source_prefix": pair[0],
               "x0": x0, "y0_latex": latex(y0), "constant": c}
        return [general, ivp]
    if sp.simplify(sp.diff(answer, x) - source) != 0:
        raise ValueError("source integral failed differentiation")
    key = hashlib.sha256(sp.srepr(sp.factor(source)).encode()).hexdigest()
    return [{"id": f"{split}:{method}:{serial}", "group": key,
             "task": "integral", "method": method, "split": split,
             "input_latex": latex(source), "target_latex": latex(answer),
             "source_prefix": pair[0][2:], "target_prefix": pair[1]}]


def generate(source: Path, output: Path, scale: float = 1.0, max_attempts_per_row: int = 120,
             methods=METHODS):
    if not (0 < scale <= 1):
        raise ValueError("scale must be in (0,1]")
    env = upstream(source)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"upstream_repo": "facebookresearch/SymbolicMathematics", "upstream_sha": UPSTREAM_SHA,
                "upstream_license": "CC BY-NC 4.0", "operators": OPERATORS,
                "max_ops": 6, "max_ops_G": 3, "max_int": 5, "scale": scale,
                "ode_generator": "linear-in-constant analytical subfamily of upstream ODE1 construction",
                "quotas": {s: {m: max(1, round(q * scale)) for m, q in ms.items()} for s, ms in QUOTAS.items()}}
    seen = set()
    for method in methods:
        if method not in METHODS:
            raise ValueError(method)
        method_idx = METHODS.index(method)
        # Reset the upstream IBP cache so the sample stream is independent of other runs.
        if method == "prim_ibp":
            env.PRIM_CACHE = {}
            env.PRIM_COUNT = [0, 0]
        seed = 4100 + method_idx
        rng = np.random.RandomState(seed)
        choice_rng = random.Random(seed)
        fn = {"prim_fwd": env.gen_prim_fwd,
              "prim_bwd": lambda r: env.gen_prim_bwd(r, True),
              "prim_ibp": env.gen_prim_ibp, "ode1": lambda r: gen_ode1_linear(env,r)}[method]
        for split in ("train", "valid", "test"):
            target = manifest["quotas"][split][method]
            path = output / f"{split}_{method}.jsonl"
            accepted, attempts, rejected, last_print = 0, 0, defaultdict(int), -1
            with path.open("w") as stream:
                while accepted < target:
                    attempts += 1
                    if attempts > target * max_attempts_per_row:
                        raise RuntimeError(f"generation stalled at {split}/{method}: {accepted}/{target}; {dict(rejected)}")
                    try:
                        pair = fn(rng)
                        if pair is None:
                            rejected["generator_none"] += 1
                            continue
                        rows = make_rows(env, method, pair, split, accepted, choice_rng)
                        if rows[0]["group"] in seen:
                            rejected["duplicate_group"] += 1
                            continue
                        seen.add(rows[0]["group"])
                        for row in rows:
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        accepted += 1
                    except Exception as exc:
                        rejected[type(exc).__name__] += 1
                    except BaseException as exc:
                        # The upstream timeout inherits BaseException, not Exception.
                        if type(exc).__name__ != "TimeoutError":
                            raise
                        rejected["upstream_timeout"] += 1
                    if accepted and accepted % 1000 == 0 and accepted != last_print:
                        print(method, split, accepted, target, "attempts", attempts, flush=True)
                        last_print = accepted
            manifest.setdefault("files", {})[path.name] = {
                "rows": accepted * (2 if method == "ode1" else 1), "groups": accepted,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "attempts": attempts, "rejected": dict(rejected)}
            print(method, split, accepted, "attempts", attempts, flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=ROOT / "tmp/SymbolicMathematics")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/downstream_gpt2/data")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()
    generate(args.upstream, args.output, args.scale, methods=args.methods)


if __name__ == "__main__":
    main()
