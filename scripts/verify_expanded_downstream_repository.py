"""Three independent repository checks for the expanded 80-token run."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/expanded_downstream_80_v1"
CORPUS = ROOT / "data/formulas/expanded_downstream_80_v1"
REPORT = ROOT / "reports/expanded_downstream_80_v1"
CASES = (
    "Wave_equation", "Convection_diffusion_equation", "KG_equation",
    "Burgers_equation", "Eq_6_2_12", "PDE_divide", "KdV_equation",
    "Chaffee_Infante_equation",
)
ARMS = ("eqgpt_reference", "ast_bpe_280", "ast_bpe_315", "ast_bpe_350")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def round_one() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_expanded_preflight.py",
         "tests/test_expanded_corpus.py", "tests/test_pde_preparation.py", "-q"],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)
    return {"status": "PASS", "command": "pytest expanded preflight/corpus/PDE preparation",
            "output": p.stdout.strip()}


def round_two() -> dict:
    checks = {}
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    checks["corpus_counts"] = manifest["source_records"] == 1855 and manifest["augmented_samples"] == 59360
    sources = [json.loads(x) for x in (CORPUS / "sources.jsonl").read_text().splitlines()]
    kinds = collections.Counter(x["source_kind"] for x in sources)
    checks["source_partition"] = kinds == {"external_benchmark": 1643, "eqgpt_handbook_non_target": 212}
    for name in ("audit_pass1.json", "audit_pass2.json"):
        checks[name] = json.loads((CORPUS / name).read_text()).get("status") == "PASS"
    audit = json.loads((ROOT / "data/formulas/gre_benchmark_audit_v2/audit.json").read_text())
    ext = json.loads((ROOT / "data/formulas/gre_benchmark_filtered/manifest.json").read_text())
    checks["independent_benchmark_audit"] = audit["external_records"] == 2049 and audit["external_no_221_overlap"]
    checks["external_benchmark_size"] = ext["input_records"] == 2049 and ext["excluded_records"] == 4 and ext["retained_records"] == 2045
    status = json.loads((EXP / "status.json").read_text())
    checks["formal_run_complete"] = status["status"] == "COMPLETE" and len(status["completed_stages"]) == 9
    result_files = sorted(EXP.glob("cases/*/*/search.json"))
    checks["32_search_results"] = len(result_files) == 32
    for p in result_files:
        d = json.loads(p.read_text())
        checks["all_search_budgets"] = checks.get("all_search_budgets", True) and len(d["attempts"]) == 600 and len(d["history"]) == 3 and all(h["attempts"] == 200 and h["continuation_epochs"] == 2 for h in d["history"])
        checks["per_arm_fast_summaries"] = checks.get("per_arm_fast_summaries", True) and (p.parent / "fast_summary.json").exists()
    checks["complete_structured_result"] = (REPORT / "完整结果.json").exists()
    weights = json.loads((EXP / "weight_manifest.json").read_text())
    ok = len(weights["files"]) == 36
    for item in weights["files"]:
        p = EXP / item["path"]
        if not p.is_file() or p.stat().st_size != item["size_bytes"] or sha(p) != item["sha256"]:
            ok = False
    checks["36_local_checkpoint_hashes"] = ok
    for arm in ARMS:
        d = json.loads((EXP / "shared_pretrain" / arm / "training.json").read_text())
        checks["all_pretrain_records"] = checks.get("all_pretrain_records", True) and d["epochs"] == 25 and d["samples"] == 59360
    for case in CASES:
        d = json.loads((EXP / "cases" / case / "complete.json").read_text())
        checks["all_case_completions"] = checks.get("all_case_completions", True) and len(d["results"]) == 4
    old_unchanged = subprocess.run(
        ["git", "diff", "--quiet", "--", "experiments/downstream",
         "data/formulas/eqgpt/PDE_dataset_0725.xlsx", "data/formulas/eqgpt/augmented"],
        cwd=ROOT,
    ).returncode == 0
    checks["old_benchmark_and_experiment_unchanged"] = old_unchanged
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks, ensure_ascii=False, indent=2))
    return {"status": "PASS", "checks": checks}


def round_three() -> dict:
    docs = [
        ROOT / "README.md", ROOT / "data/README.md",
        ROOT / "reports/downstream/README.md",
        ROOT / "experiments/expanded_downstream_80_v1/README.md",
        ROOT / "data/formulas/gre_benchmark_filtered/README.md",
        ROOT / "data/formulas/gre_benchmark_merged/README.md",
        REPORT / "report_zh.md", REPORT / "comparison_previous_zh.md",
    ]
    missing = []
    pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for doc in docs:
        for target in pattern.findall(doc.read_text()):
            target = target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (doc.parent / target).exists():
                missing.append(f"{doc.relative_to(ROOT)} -> {target}")
    ignore = (ROOT / ".gitignore").read_text()
    ignore_rule = "/experiments/expanded_downstream_80_v1/**/*.pt" in ignore
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "experiments/expanded_downstream_80_v1/shared_pretrain/eqgpt_reference/pretrained.pt"],
        cwd=ROOT,
    ).returncode == 0
    checks = {
        "documentation_links": not missing,
        "checkpoint_policy_documented": ignore_rule and ignored,
        "old_and_new_benchmark_explicitly_distinguished": all(
            term in (ROOT / "data/README.md").read_text()
            for term in ("221 条", "gre_benchmark_filtered", "gre_benchmark_merged")
        ),
        "separate_experiment_directory_documented": "expanded_downstream_80_v1" in (ROOT / "README.md").read_text(),
    }
    if not all(checks.values()):
        raise AssertionError(json.dumps({"checks": checks, "missing_links": missing}, ensure_ascii=False, indent=2))
    return {"status": "PASS", "checks": checks, "checked_documents": [str(p.relative_to(ROOT)) for p in docs]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    result = (round_one, round_two, round_three)[args.round - 1]()
    result["round"] = args.round
    path = REPORT / f"repository_check_round_{args.round}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    all_results = {}
    for n in range(1, 4):
        p = REPORT / f"repository_check_round_{n}.json"
        if p.exists():
            all_results[str(n)] = json.loads(p.read_text())
    if len(all_results) == 3:
        (REPORT / "repository_checks_3rounds.json").write_text(
            json.dumps({"status": "PASS" if all(x["status"] == "PASS" for x in all_results.values()) else "FAIL",
                        "rounds": all_results}, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
