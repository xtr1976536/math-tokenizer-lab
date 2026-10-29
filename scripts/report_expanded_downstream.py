"""Regenerate per-arm summaries and the complete expanded-run result record."""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.evaluation import classify
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES

ARMS = ("eqgpt_reference", "ast_bpe_280", "ast_bpe_315", "ast_bpe_350")
LABELS = {"correct": "correct", "redundant": "redundant", "incorrect": "incorrect"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/expanded_downstream_80_v1")
    args = parser.parse_args()
    out = args.output
    summaries = {}
    for case in SHARED8_CASES:
        case_results = {}
        for arm in ARMS:
            folder = out / "cases" / case / arm
            search = json.loads((folder / "search.json").read_text())
            training = json.loads((out / "shared_pretrain" / arm / "training.json").read_text())
            attempts = search["attempts"]
            statuses = collections.Counter(row["status"] for row in attempts)
            final = search["history"][-1]
            ranked = [classify(seq, case)["status"] for seq in final["gre"]]
            first_correct = next((i + 1 for i, row in enumerate(attempts)
                                  if row.get("status") == "evaluated" and
                                  classify(row["gre"], case)["status"] == "correct"), None)
            fast = {
                "training_seconds": sum(row["seconds"] for row in training["history"]),
                "search_seconds": search["elapsed_seconds"],
                "top1": ranked[0] if ranked else "no_evaluable_candidate",
                "top10_correct": "correct" in ranked,
                "top10_has_redundant": "redundant" in ranked,
                "ranked_classes": ranked,
                "attempted_statuses": dict(statuses),
                "first_correct": first_correct,
                "reward_calls": search["performance"]["author_reward_calls"],
                "cache_hits": search["performance"]["exact_reward_cache_hits"],
            }
            (folder / "fast_summary.json").write_text(json.dumps(fast, ensure_ascii=False, indent=2) + "\n")
            case_results[arm] = {
                **fast,
                "top1_equation": final["equations"][0] if final["equations"] else None,
                "valid_candidates": statuses["evaluated"],
                "attempts": len(attempts),
                "valid_rate": statuses["evaluated"] / len(attempts),
                "pretrained_sha256": json.loads((folder / "SEARCH_INPUT.json").read_text())["pretrained_sha256"],
                "searched_tensor_sha256": final["after_sha256"],
            }
        summaries[case] = case_results
    report = {
        "experiment": "expanded_downstream_80_v1",
        "status": "COMPLETE",
        "corpus": {"source_records": 1855, "external_benchmark": 1643,
                   "eqgpt_non_target": 212, "augmented_samples": 59360},
        "protocol": {"seed": 525, "layers": 3, "epochs": 25, "capacity": 80,
                     "attempts_per_cycle": 200, "cycles": 3, "continuation_epochs": 2},
        "cases": summaries,
    }
    dest = ROOT / "reports/expanded_downstream_80_v1/完整结果.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(SHARED8_CASES) * len(ARMS)} per-arm summaries and {dest}")


if __name__ == "__main__":
    main()
