"""Pre-result audits for dataset, tokenizer transfer, and verifier integrity."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from math_tokenizer.corpus import ROOT
from .adapters import METHODS, TokenizerAdapter
from .train import read_split
from .verify import Verifier


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_audit(data: Path, source: Path):
    manifest = json.loads((data / "manifest.json").read_text())
    for filename, entry in manifest["files"].items():
        path = data / filename
        assert digest(path) == entry["sha256"], filename
        assert sum(1 for _ in path.open()) == entry["rows"], filename
    groups = {}
    summaries = {}
    oracle = {}
    failures = {}
    verifier = Verifier(source)
    for split in ("train", "valid", "test"):
        rows = read_split(data, split)
        tally = Counter()
        good, bad = [], []
        for row in rows:
            previous = groups.setdefault(row["group"], split)
            assert previous == split, (row["id"], previous, split)
            verdict = verifier.verify(row, row["target_latex"])
            (good if verdict == "correct" else bad).append(row["id"])
            tally[row["task"] + ":" + row["method"]] += 1
        assert len({r["id"] for r in rows}) == len(rows)
        if len(bad)/len(rows) > .005:
            raise RuntimeError(f"reference verification failed on {split}: {len(bad)}/{len(rows)}")
        oracle[split], failures[split] = sorted(good), sorted(bad)
        summaries[split] = dict(tally)
    # Literal overlap is one auditable layer; it does not prove semantic non-overlap.
    hash_file = ROOT / "data/benchmark/formula_hashes.json.gz"
    normalize = lambda s: "".join(s.split())
    if hash_file.exists():
        with gzip.open(hash_file,"rt") as stream:
            catalog = set(json.load(stream))
    else:
        catalog = set()
        for path in (ROOT / "data/benchmark/formula_only").rglob("*.jsonl"):
            catalog.update(hashlib.sha256(normalize(json.loads(line)).encode()).hexdigest()
                           for line in path.read_text().splitlines())
    overlap = {split: sum(hashlib.sha256(normalize(row["input_latex"]).encode()).hexdigest() in catalog or
                          hashlib.sha256(normalize(row["target_latex"]).encode()).hexdigest() in catalog
                          for row in read_split(data, split)) for split in summaries}
    (data / "oracle_valid_ids.json").write_text(json.dumps(oracle,indent=2)+"\n")
    return {"dataset_counts": summaries, "distinct_groups": len(groups),
            "oracle_unverified": failures, "oracle_valid_counts":{s:len(ids) for s,ids in oracle.items()},
            "catalog_literal_overlap": overlap, "manifest_sha256": digest(data / "manifest.json")}


def tokenizer_audit(data: Path, gpt2: Path):
    rows = read_split(data, "train")[:1000]
    report = {}
    for name in METHODS:
        adapter = TokenizerAdapter(name, gpt2)
        for row in rows:
            for text in (row["input_latex"], row["target_latex"]):
                assert adapter.decode(adapter.encode(text)) == text, (name, row["id"])
        assert all(adapter.payload(i) is not None for row in rows for text in
                   (row["input_latex"], row["target_latex"]) for i in adapter.encode(text))
        report[name] = {"vocab_rows": adapter.vocab_size, "masked_output_ids": len(adapter.masked_ids())}
    left = TokenizerAdapter("ast_boundary", gpt2)
    right = TokenizerAdapter("official_atomic_bpe", gpt2)
    assert left.base_size == right.base_size
    assert all(left.payload(i) == right.payload(i) for i in range(left.base_size))
    return {"methods": report, "shared_proposed_dictionary": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "tmp/downstream_gpt2/data")
    parser.add_argument("--source", type=Path, default=ROOT / "tmp/SymbolicMathematics")
    parser.add_argument("--gpt2", type=Path, default=ROOT / "tmp/gpt2-607a30d")
    parser.add_argument("--out", type=Path, default=ROOT / "tmp/downstream_gpt2/audit.json")
    args = parser.parse_args()
    result = {"round_1_dataset": dataset_audit(args.data, args.source),
              "round_2_tokenizers": tokenizer_audit(args.data, args.gpt2)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
