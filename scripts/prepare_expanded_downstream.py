"""Build the paired expanded corpus used by the next downstream run.

The benchmark and the non-target EqGPT handbook rows are kept as separate
provenance classes. Each source structure receives the same deterministic
32-way additive-term augmentation in GRE and LaTeX form. This script prepares
and audits inputs only; it never trains a Transformer or runs a numerical
search.
"""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random

from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.corpus_conversion import gre_expression
from math_tokenizer.eqgpt.latex_codec import structure_to_latex
from math_tokenizer.eqgpt.author import DICTIONARY
from scripts.prepare_downstream_corpus import TARGETS, invariant_support, gre_expression as gre_expr, words_to_ids

SEED = 525
AUGMENTATIONS = 32
TARGET_KEYS = {name: invariant_support(gre_expr(words_to_ids(words)))
               for name, words in TARGETS.items()}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def term_groups(sequence: list[int]) -> list[list[int]]:
    if not sequence or sequence[0] != 5 or sequence[-1] != 1:
        raise ValueError("GRE sequence must have S/E controls")
    groups, current = [], []
    for token in sequence[1:-1]:
        if token == 2:
            if not current:
                raise ValueError("Empty GRE additive term")
            groups.append(current)
            current = []
        else:
            current.append(int(token))
    if not current:
        raise ValueError("Empty final GRE additive term")
    groups.append(current)
    return groups


def augmented_sequences(sequence: list[int], seed: int) -> list[list[int]]:
    groups = term_groups(sequence)
    rng = random.Random(seed)
    result: list[list[int]] = []
    # The first item is the source ordering. The remaining items use the same
    # deterministic term permutations for both GRE and rendered LaTeX.
    for index in range(AUGMENTATIONS):
        order = list(range(len(groups)))
        if index:
            rng.shuffle(order)
        candidate = [5]
        for position, group_index in enumerate(order):
            if position:
                candidate.append(2)
            candidate.extend(groups[group_index])
        candidate.append(1)
        result.append(candidate)
    return result


def read_sources():
    records = []
    benchmark = ROOT / "data/formulas/gre_benchmark_filtered/benchmark.jsonl"
    for row in map(json.loads, benchmark.read_text().splitlines()):
        records.append(dict(
            source_id=row["id"], source_kind="external_benchmark",
            source_group=row["source_group"], source_url=row["source_url"],
            source_latex=row["latex"], gre_ids=row["conversion"]["gre_ids"],
            source_parameters=row.get("parameters", {}),
            source_family=row.get("family"), source_record=row["id"]))
    handbook = ROOT / "data/formulas/gre_benchmark_audit_v2/eqgpt_221_after_target_exclusion.jsonl"
    for row in map(json.loads, handbook.read_text().splitlines()):
        records.append(dict(
            source_id="eqgpt221:" + row["id"], source_kind="eqgpt_handbook_non_target",
            source_group="eqgpt_original_221", source_url=row.get("source_url"),
            source_latex=row["latex"], gre_ids=row["gre_ids"],
            source_parameters={}, source_family="eqgpt_handbook_221",
            source_record=row["id"], equation_name=row.get("equation_name")))
    return records


def main():
    output = ROOT / "data/formulas/expanded_downstream_v1"
    output.mkdir(parents=True, exist_ok=True)
    records = read_sources()
    if len(records) != 2045 + 212:
        raise ValueError(f"Expected 2257 source structures, found {len(records)}")
    seen = set()
    external_seen = set()
    for row in records:
        key = invariant_support(gre_expr(row["gre_ids"]))
        if row["source_kind"] == "external_benchmark":
            if key in external_seen:
                raise ValueError("Duplicate external benchmark structure: " + row["source_id"])
            external_seen.add(key)
        seen.add(key)
        if key in TARGET_KEYS.values():
            raise ValueError("Target structure entered expanded corpus: " + row["source_id"])
        if not all(0 <= int(token) < 57 for token in row["gre_ids"]):
            raise ValueError("GRE OOV in " + row["source_id"])
    augmented = []
    for index, source in enumerate(records):
        sequences = augmented_sequences(source["gre_ids"], SEED + index)
        for augmentation_index, gre_ids in enumerate(sequences):
            augmented.append(dict(
                source_id=source["source_id"], source_kind=source["source_kind"],
                source_group=source["source_group"], source_record=source["source_record"],
                source_latex=source["source_latex"],
                augmented_gre_ids=gre_ids,
                augmented_gre_words=[DICTIONARY["id2word"][int(token)] for token in gre_ids],
                augmented_latex=structure_to_latex(gre_ids),
                augmentation_index=augmentation_index,
                seed=SEED + index))
    # The actual words are reconstructed by the downstream reader from IDs;
    # retaining IDs here prevents a second, potentially inconsistent parser.
    with (output / "paired_augmented.jsonl").open("w") as stream:
        for row in augmented:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    source_payload = json.dumps(records, ensure_ascii=False, sort_keys=True).encode()
    augmented_payload = (output / "paired_augmented.jsonl").read_bytes()
    manifest = dict(
        version="expanded-downstream-v1",
        source_structures=len(records),
        source_kinds=dict(Counter(row["source_kind"] for row in records)),
        augmentation_policy="32 deterministic additive-term permutations; source ordering first; same GRE ordering rendered to LaTeX",
        augmentations_per_source=AUGMENTATIONS,
        augmented_examples=len(augmented),
        expected_augmented_examples=len(records) * AUGMENTATIONS,
        seed=SEED,
        source_sha256=sha_bytes(source_payload),
        augmented_sha256=sha_bytes(augmented_payload),
        target_keys_excluded=list(TARGET_KEYS),
        tokenizer_input_policy="Hard tokenizer and Hard Transformer use augmented_latex; EqGPT uses augmented_gre_ids",
        mlp_policy="Reuse data/mlp_cache/<case>/; no surrogate fitting in preparation",
        training_policy=dict(epochs=25, layers=3, attempts_per_cycle=200,
                             cycles=3, continuation_epochs=2, sizes=[280, 315, 350],
                             shared_seed=SEED),
        downstream_training_started=False,
        numerical_search_started=False,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
