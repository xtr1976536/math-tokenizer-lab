"""Executable, tokenizer-independent symbolic correctness checks."""
from __future__ import annotations

import re
import signal
from contextlib import contextmanager
from pathlib import Path

import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

from .data import upstream

ALLOWED_COMMANDS = {"frac", "sqrt", "sin", "cos", "exp", "log", "ln", "left", "right",
                    "cdot", "times", "quad", "pi", "mathrm"}


@contextmanager
def time_limit(seconds=3):
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    def expired(_signum, _frame):
        raise TimeoutError("symbolic verification timed out")
    old = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old)


def parse_answer(text: str, x: sp.Symbol):
    if not isinstance(text, str) or not text.strip() or len(text) > 1024:
        raise ValueError("empty or overlong output")
    if not re.fullmatch(r"[0-9A-Za-z_{}()\[\]\\^+*/.,=\-\s|]+", text):
        raise ValueError("output contains characters outside math grammar")
    if set(re.findall(r"\\([A-Za-z]+)", text)) - ALLOWED_COMMANDS:
        raise ValueError("output contains unsupported LaTeX command")
    result = parse_latex(text)
    if isinstance(result, sp.Equality):
        raise ValueError("answer must be an expression")
    result = result.xreplace({sp.Symbol("x"): x, sp.Symbol("e"): sp.E,
                              sp.Symbol("E"): sp.E, sp.Symbol("pi"): sp.pi})
    result = result.xreplace({symbol: sp.Symbol(symbol.name, real=True)
                              for symbol in result.free_symbols if symbol != x})
    if result.has(sp.Integral, sp.Derivative, sp.Piecewise, sp.Limit):
        raise ValueError("unresolved operation in answer")
    if sp.count_ops(result) > 100:
        raise ValueError("answer complexity limit")
    return result


class Verifier:
    def __init__(self, source: Path):
        self.env = upstream(source)
        self.x = self.env.variables["x"]
        self.f = self.env.functions["f"](self.x)

    def source(self, row):
        infix = row.get("source_infix")
        if infix is None:
            infix = self.env.prefix_to_infix(self.env.unclean_prefix(row["source_prefix"]))
        return parse_expr(infix, local_dict=self.env.local_dict)

    def _check(self, row, answer):
        source = self.source(row)
        if row["task"] == "integral":
            extra = answer.free_symbols - {self.x}
            if any(s.name in ("x", "e", "pi") for s in extra) or len(extra) > 1:
                return "invalid"
            residual = sp.diff(answer, self.x) - source
        else:
            extra = answer.free_symbols - {self.x}
            if row["task"] == "ode_ivp" and extra:
                return "invalid"
            residual = source.subs(self.f, answer).doit()
            if row["task"] == "ode_ivp":
                y0 = parse_answer(row["y0_latex"], self.x)
                if sp.simplify(answer.subs(self.x, row["x0"]) - y0) != 0:
                    return "wrong"
        if sp.simplify(sp.together(residual)) == 0:
            return "correct"
        # A numerical witness is only used to reject; it cannot certify correctness.
        for value in (sp.Rational(1, 3), sp.Rational(7, 5), sp.Rational(-2, 3)):
            try:
                result = complex(residual.subs(self.x, value).evalf(16))
                if all(abs(v) < 1e9 for v in (result.real, result.imag)) and abs(result) > 1e-7:
                    return "wrong"
            except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                continue
        return "unverified"

    def verify(self, row, text: str, timeout_seconds=3):
        try:
            with time_limit(timeout_seconds):
                answer = parse_answer(text, self.x)
                return self._check(row, answer)
        except TimeoutError:
            return "timeout"
        except (ValueError, TypeError, SyntaxError, AttributeError, AssertionError, KeyError, IndexError):
            return "invalid"
        except Exception:
            return "unverified"
