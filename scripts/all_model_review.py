"""Audit and classify saved experiments without encoding, training or data edits.

Only the review JSON and Markdown appendix are written. Historical experiment
artifacts remain untouched. Run with the project's Python from the repo root.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import statistics
import subprocess
from math_tokenizer.checkpoints import verify_saved_hash

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    "pilot01", "sensitivity512", "v2_release512", "v2_release1024",
    "v3_512_release", "v3_1024_release", "v4_512", "v4_1024", "v5_1024",
]
FIELDS = [
    "tokens_per_formula", "bytes_per_token", "roundtrip_rate",
    "command_integrity_rate", "number_integrity_rate", "lexeme_preservation_rate",
    "hybrid_tree_crossing_rate", "encode_mib_s", "vocab_actual",
    "vocab_utilization", "utf8_complete_token_rate",
    "empirical_unigram_bits_per_byte", "renyi_efficiency_2",
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_digest(path):
    """Hash the tracked lossless archive without unpacking/mutating the corpus."""
    if path.exists():
        return digest(path)
    h = hashlib.sha256()
    with gzip.open(path.with_suffix(path.suffix + ".gz"), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(method):
    if method.startswith(("tiktoken_", "qwen")):
        return "pretrained_reference", "mature_fixed_vocabulary"
    if method.startswith("legacy_"):
        return "prior_data_reference", "RL_AST_path_legacy" if "v4" in method else "RL_path_legacy"
    if method == "utf8_bytes":
        return "fallback_reference", "bytes"
    if method in ("byte_bpe", "hf_byte_bpe", "hf_bpe_fulltrain", "hf_bpe_matched_fit"):
        return "baseline", "frequency_BPE"
    if method.startswith(("sentencepiece_", "sp_")):
        return "baseline", "BPE" if method.endswith("bpe") else "Unigram"
    if method in ("frequency", "reward_greedy", "random", "rl_length_only"):
        return "control", "RL_merge_ablation" if method == "rl_length_only" else "merge_selection"
    if "native_exact" in method:
        return "engineering_control", "RL_path"
    if "control" in method or "no_rl" in method or "shortest" in method or method in (
        "masked_zero_policy", "masked_matched_initial_policy",
    ):
        return "control", "structure_or_initialization"
    if method == "rl_composite":
        return "proposed", "offline_RL_merge"
    if method.startswith("rl_ast"):
        return "proposed", "RL_AST_merge" if method in ("rl_ast_bpe", "rl_ast_bpe_v1") else "RL_AST_path"
    if method == "rl_v4":
        return "proposed", "RL_AST_path"
    if method.startswith("rl"):
        return "proposed", "RL_merge" if method in ("rl_bpe", "rl_bpe_v1") else "RL_path"
    if method.startswith("ast"):
        return "proposed", "structure_BPE"
    raise ValueError(f"Unclassified model: {method}")


def key(row):
    return row.get("scenario", "historical"), row.get("suite", "test"), row["method"], row["seed"]


def audit_run(name):
    directory = ROOT / "results" / name
    metrics = json.loads((directory / "metrics.json").read_text())
    grouped = defaultdict(list)
    byte_lengths = {}
    with (directory / "per_formula.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            grouped[key(row)].append(row)
            identity = row.get("suite", "test"), row["id"]
            if identity in byte_lengths:
                assert byte_lengths[identity] == row["bytes"], (name, identity)
            byte_lengths[identity] = row["bytes"]
    assert len(metrics) == len(grouped)
    assert {key(m) for m in metrics} == set(grouped)
    assert len({key(m) for m in metrics}) == len(metrics)
    scopes = defaultdict(list)
    for metric in metrics:
        rows = grouped[key(metric)]
        n = metric.get("n_formulas", metric.get("formulas"))
        ids = sorted(r["id"] for r in rows)
        assert len(rows) == len(set(ids)) == n
        assert sum(r["tokens"] for r in rows) == metric["tokens"]
        assert sum(r["bytes"] for r in rows) == metric["bytes"]
        assert abs(metric["tokens_per_formula"] - metric["tokens"] / n) < 1e-12
        assert abs(metric["roundtrip_rate"] - sum(r["roundtrip"] for r in rows) / n) < 1e-12
        for numerator, denominator, field in [
            ("commands_intact", "commands", "command_integrity_rate"),
            ("numbers_intact", "numbers", "number_integrity_rate"),
            ("lexemes_intact", "lexemes", "lexeme_preservation_rate"),
        ]:
            if field in metric:
                total = sum(r[denominator] for r in rows)
                if total:
                    value = sum(r[numerator] for r in rows) / total
                    assert abs(value - metric[field]) < 1e-12, (name, key(metric), field)
        if "hybrid_tree_crossing_rate" in metric:
            measurable = [r for r in rows if r["crossing_tokens"] is not None]
            value = sum(r["crossing_tokens"] for r in measurable) / max(1, sum(r["tokens"] for r in measurable))
            assert abs(value - metric["hybrid_tree_crossing_rate"]) < 1e-12
        id_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if "test_id_sha256" in metric:
            assert metric["test_id_sha256"] == id_hash
        scopes[key(metric)[:2]].append(id_hash)
    for hashes in scopes.values():
        assert len(set(hashes)) == 1, (name, "unequal inputs")
    # v5 references and train scenarios must receive identical suite inputs too.
    by_suite = defaultdict(set)
    for (_, suite), hashes in scopes.items():
        by_suite[suite].update(hashes)
    assert all(len(hashes) == 1 for hashes in by_suite.values())

    summaries = defaultdict(list)
    for m in metrics:
        summaries[key(m)[:3]].append(m)
    models = []
    for (scenario, suite, method), mm in sorted(summaries.items()):
        role, family = classify(method)
        rr = grouped[key(mm[0])]
        summary = {
            "scenario": scenario, "suite": suite, "method": method,
            "role": role, "family": family, "seeds": [m["seed"] for m in mm],
            "formulas": len(rr), "groups": len({r["group"] for r in rr}),
        }
        for field in FIELDS:
            if all(m.get(field) is not None for m in mm):
                summary[field] = statistics.mean(m[field] for m in mm)
        # Saved per-formula statistics support independent subgroup review.
        strata = {}
        for dimension in ("domain", "complexity", "origin", "stress"):
            if dimension not in rr[0]:
                continue
            groups = defaultdict(list)
            for r in rr:
                groups[str(r[dimension])].append(r)
            strata[dimension] = {}
            for label, subset in sorted(groups.items()):
                same_ids = {r["id"] for r in subset}
                seed_rows = [r for m in mm for r in grouped[key(m)] if r["id"] in same_ids]
                record = {
                    "formulas": len(subset), "groups": len({r["group"] for r in subset}),
                    "tokens_per_formula": statistics.mean(r["tokens"] for r in seed_rows),
                    "tokens_per_byte": sum(r["tokens"] for r in seed_rows) / sum(r["bytes"] for r in seed_rows),
                    "roundtrip_rate": statistics.mean(r["roundtrip"] for r in seed_rows),
                }
                if "commands" in subset[0]:
                    total = sum(r["commands"] for r in seed_rows)
                    record["command_integrity_rate"] = sum(r["commands_intact"] for r in seed_rows) / total if total else None
                strata[dimension][label] = record
        summary["strata"] = strata
        models.append(summary)
    manifest = json.loads((directory / "manifest.json").read_text())
    model_hashes = manifest.get("models", manifest.get("model_hashes", {}))
    checkpoint_states = defaultdict(int)
    for filename, expected in model_hashes.items():
        checkpoint_states[verify_saved_hash(directory / "models" / filename, expected)] += 1
    for filename in ("protocol", "selection", "metrics"):
        if f"{filename}_sha256" in manifest:
            assert digest(directory / f"{filename}.json") == manifest[f"{filename}_sha256"]
    return {
        "run": name, "passed": True, "method_seed_suite_records": len(metrics),
        "per_formula_records": sum(len(r) for r in grouped.values()),
        "manifest_model_files_verified": checkpoint_states['actual'] + checkpoint_states['relocated'],
        "checkpoint_evidence_states": dict(checkpoint_states),
        "retired_checkpoint_hashes_matched_to_cleanup_manifest": checkpoint_states['retired'],
        "test_hashes": {suite: next(iter(hashes)) for suite, hashes in by_suite.items()},
        "artifact_sha256": {filename: digest(directory / filename) for filename in ("metrics.json", "per_formula.jsonl")},
        "models": models,
    }


def fmt(value, percentage=False):
    if value is None:
        return "—"
    return f"{value * 100:.2f}%" if percentage else f"{value:.4f}"


def main():
    runs = [audit_run(name) for name in RUNS]
    hashes = {r["test_hashes"]["test"] for r in runs[2:8]}
    assert len(hashes) == 1, "v2-v4 test inputs changed"
    corpus_hashes = {}
    for release, names in [("v2", RUNS[2:8]), ("v5", ["v5_1024"])]:
        corpus_hashes[release] = corpus_digest(ROOT / "data/releases" / release / "curated/corpus.jsonl")
        for name in names:
            protocol = json.loads((ROOT / "results" / name / "protocol.json").read_text())
            assert protocol.get("data_sha256", protocol.get("corpus_sha256")) == corpus_hashes[release]
    pairwise = json.loads((ROOT / "results/v5_1024/pairwise.json").read_text())
    dev_inventory = {}
    for path in sorted((ROOT / "results").glob("*/dev_ablations.json")):
        rows = json.loads(path.read_text())
        compact = []
        for row in rows:
            record = {k: v for k, v in row.items() if k not in ("training_log", "fit_ids", "reward_ids")}
            for field in ("training_log", "fit_ids", "reward_ids"):
                if field in row:
                    record[f"{field}_count"] = len(row[field])
            compact.append(record)
        item = {"formal_run": path.parent.name in RUNS, "rows": len(rows), "sha256": digest(path), "records": compact}
        if path.parent.name.startswith("v3_") and path.parent.name.endswith("_release"):
            item["unique_policy_training_runs"] = len({(r["name"].rsplit("-cross", 1)[0], r["seed"]) for r in rows if r["arm"] == "rl_byte"})
        elif path.parent.name.startswith("v4_"):
            item["unique_policy_training_runs"] = len({(r["atoms"], r["config"], r["seed"]) for r in rows if r["arm"] == "rl"})
        dev_inventory[str(path.relative_to(ROOT))] = item
    result = {
        "scope": "saved-result retrospective only; no model/corpus edits, no encoding/training rerun",
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_reviewed": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "formal_directories": len(runs),
        "method_seed_suite_records": sum(r["method_seed_suite_records"] for r in runs),
        "per_formula_records": sum(r["per_formula_records"] for r in runs),
        "manifest_model_files_verified": sum(r["manifest_model_files_verified"] for r in runs),
        "corpus_sha256_verified": corpus_hashes,
        "dev_sweep_inventory": dev_inventory,
        "formal_dev_evaluation_rows": sum(v["rows"] for v in dev_inventory.values() if v["formal_run"]),
        "runs": runs,
        "excluded_from_formal_rankings": {
            "smoke_only": ["smoke01", "v2_smoke512"],
            "incomplete_or_invalidated": ["v2_512", "v2_512_final", "v2_1024_final", "v3_512"],
            "dev_only_or_implementation_probes": ["v2_variance_probe.json", "v3_exploratory_probes.json", "v3_native_probe.json"],
        },
        "v5_pairwise_seed17": {k: v for k, v in pairwise.items() if k.endswith("/17")},
        "ranking_limits": [
            "Ranks are descriptive point estimates within listed inputs and budgets; adjacent ranks are not significance claims.",
            "Original v1, v2-v4 and v5 use different corpora. Frozen legacy and pretrained references cannot enter matched-training ranks.",
            "v5 matched-fit BPE matches fit formulas, but RL and greedy also use training-internal reward formulas.",
            "Deterministic repeated seeds are not independent replications. Timing is stored mean of seed-level median runs, not rerun here.",
            "Zero crossing is relative to the project's parser; no independent semantic AST truth or downstream model evaluation.",
        ],
    }
    dest = ROOT / "results/retrospective/all_model_inventory.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 全部正式实验的数值附录", "",
        "由 scripts/all_model_review.py 从保存的逐公式记录独立核对；没有重训或重新编码。",
        "token/公式、跨树比例越低越好；命令完整率、编码速度越高越好。速度是原实验记录，未在本次重测。",
        "训练秒的统计口径跨版本不同，因此不做跨版本训练时间排名。参考模型、控制组与正式方法用 role 区分。",
        "重复 seeds 不等于独立发现；本表是描述性排序，不自动表示相邻名次具有统计显著差异。", "",
    ]
    for run in runs:
        lines += [f"## {run['run']}", "", f"原始逐公式记录 {run['per_formula_records']:,}；方法/种子/场景记录 {run['method_seed_suite_records']}。", ""]
        scope_rows = defaultdict(list)
        for model in run["models"]:
            scope_rows[(model["scenario"], model["suite"])].append(model)
        for (scenario, suite), mm in sorted(scope_rows.items()):
            lines += [f"场景：{scenario}/{suite}。", "", "| 方法 | role | seeds | token/公式 | 命令完整 | 跨树 | MiB/s |", "|---|---|---|---:|---:|---:|---:|"]
            for m in sorted(mm, key=lambda x: x["tokens_per_formula"]):
                lines.append(f"| {m['method']} | {m['role']} | {','.join(map(str,m['seeds']))} | {fmt(m['tokens_per_formula'])} | {fmt(m.get('command_integrity_rate'),True)} | {fmt(m.get('hybrid_tree_crossing_rate'),True)} | {fmt(m.get('encode_mib_s'))} |")
            lines.append("")
    (ROOT / "docs/all_model_review_numeric_appendix.md").write_text("\n".join(lines).rstrip() + "\n")
    print(json.dumps({k: result[k] for k in ("formal_directories", "method_seed_suite_records", "per_formula_records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
