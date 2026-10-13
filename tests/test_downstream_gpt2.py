import gzip
import json
import sympy as sp

from math_tokenizer.downstream_gpt2.adapters import ByteDecomposer, kernel_slug
from math_tokenizer.downstream_gpt2.data import catalog_hashes, template_signature
from math_tokenizer.downstream_gpt2.results import paired_interval, stratified_beam_ids
from math_tokenizer.downstream_gpt2.verify import Verifier, parse_answer


def make_verifier(source):
    verifier = Verifier.__new__(Verifier)
    verifier.x = sp.Symbol("x", real=True, nonzero=True)
    verifier.f = sp.Function("f")(verifier.x)
    verifier.source = lambda _row: source
    return verifier


def test_latex_multiplication_and_euler_constant():
    x = sp.Symbol("x", real=True, nonzero=True)
    assert sp.simplify(parse_answer(r"x \cdot \left(x+1\right)", x)-x*(x+1)) == 0
    assert sp.simplify(parse_answer(r"e^{x}", x)-sp.exp(x)) == 0


def test_integral_verified_by_derivative_and_wrong_rejected():
    x = sp.Symbol("x", real=True, nonzero=True)
    verifier = make_verifier(x)
    row = {"task": "integral"}
    assert verifier.verify(row, r"\frac{x^{2}}{2}") == "correct"
    assert verifier.verify(row, r"x^{2}") == "wrong"
    assert verifier.verify(row, r"\bad{x}") == "invalid"


def test_ode_initial_condition_prevents_easy_special_solution():
    x = sp.Symbol("x", real=True, nonzero=True)
    f = sp.Function("f")(x)
    verifier = make_verifier(sp.diff(f, x)-f)
    general = {"task": "ode_general"}
    ivp = {"task": "ode_ivp", "x0": 0, "y0_latex": "2"}
    assert verifier.verify(general, r"2 \cdot e^{x}") == "correct"
    assert verifier.verify(ivp, r"2 \cdot e^{x}") == "correct"
    assert verifier.verify(ivp, "0") == "wrong"


def test_byte_decomposition_covers_partial_token_bytes():
    class Native:
        name = "gpt2_native"
        base_size = 3
        def payload(self, index):
            return (b"a", b"b", b"ab")[index]
    decomposer = ByteDecomposer(Native())
    assert decomposer.decompose(b"ab") == [(2, 1.0)]
    assert decomposer.decompose(b"ba") == [(1, .5), (0, .5)]


def test_kaggle_slugs_and_paired_group_analysis():
    assert kernel_slug("meta-llama/Meta-Llama-3-8B") == "llama3"
    assert kernel_slug("Qwen2.5-0.5B") == "qwen25"
    left = [{"id":"a","group":"A","task":"integral","method":"prim_fwd","correct":True},
            {"id":"b","group":"B","task":"integral","method":"prim_fwd","correct":False}]
    right = [{**left[0],"correct":False},{**left[1],"correct":False}]
    interval = paired_interval(left,right,"prim_fwd",replicates=100,seed=2)
    assert interval["difference"] == .5
    assert interval["groups"] == 2
    assert stratified_beam_ids(left,count_per_group=1) == {"a"}


def test_structural_split_groups_numeric_variants():
    x=sp.Symbol("x")
    assert template_signature(x+2)==template_signature(x+5)
    assert template_signature(x+2)!=template_signature(sp.sin(x)+2)


def test_kaggle_extracted_formula_hashes(tmp_path):
    folder=tmp_path/"data/benchmark"
    folder.mkdir(parents=True)
    archive=folder/"formula_hashes.json.gz"
    with gzip.open(archive,"wt") as stream:
        json.dump(["one","two"],stream)
    assert catalog_hashes(tmp_path)=={"one","two"}
    archive.unlink()
    (folder/"formula_hashes.json").write_text(json.dumps(["one","two"]))
    assert catalog_hashes(tmp_path)=={"one","two"}
