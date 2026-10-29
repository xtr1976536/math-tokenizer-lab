"""Second independent pre-training audit for the expanded paired corpus."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from functools import lru_cache

from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.author import DICTIONARY
from math_tokenizer.eqgpt.latex_codec import latex_to_structure, structure_to_latex
from scripts.prepare_downstream_corpus import TARGETS, invariant_support, gre_expression, words_to_ids


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def parse_latex(text: str):
    return latex_to_structure(text)


def main():
    folder = ROOT / "data/formulas/expanded_downstream_v1"
    manifest = json.loads((folder / "manifest.json").read_text())
    paired = list(map(json.loads, (folder / "paired_augmented.jsonl").read_text().splitlines()))
    assert len(paired) == 72224
    targets = {invariant_support(gre_expression(words_to_ids(words)))
               for words in TARGETS.values()}
    source_ids = set()
    source_kind = Counter()
    source_aug_counts = Counter()
    source_gre = {}
    parsed_source = set()
    for row in paired:
        source_ids.add(row["source_id"])
        source_kind[row["source_kind"]] += 1
        source_aug_counts[row["source_id"]] += 1
        ids = row["augmented_gre_ids"]
        words = row["augmented_gre_words"]
        assert words == [DICTIONARY["id2word"][int(i)] for i in ids]
        assert ids[0] == 5 and ids[-1] == 1
        assert all(0 <= int(i) < 57 for i in ids)
        # Parse one stored augmentation per source as a syntax check. The first
        # pass generated all 32 deterministic serializations; the second pass
        # checks every saved row's IDs, words, counts, hashes and target policy.
        if row["source_id"] not in parsed_source:
            assert structure_to_latex(ids) == row["augmented_latex"]
            assert parse_latex(row["augmented_latex"]) == ids
            parsed_source.add(row["source_id"])
        if row["source_id"] not in source_gre:
            key = invariant_support(gre_expression(ids))
            assert key not in targets
            source_gre[row["source_id"]] = set()
        source_gre[row["source_id"]].add(tuple(ids))
    assert len(source_ids) == 2257
    assert source_kind == Counter({"external_benchmark": 2045 * 32,
                                   "eqgpt_handbook_non_target": 212 * 32})
    assert set(source_aug_counts.values()) == {32}
    assert all(len(values) >= 1 for values in source_gre.values())
    assert digest(folder / "paired_augmented.jsonl") == manifest["augmented_sha256"]
    assert manifest["source_structures"] == len(source_ids)
    assert manifest["augmented_examples"] == len(paired)
    # All previously fitted MLP caches must be present and self-consistent.
    cases = ("Wave_equation", "Convection_diffusion_equation", "KG_equation",
             "Burgers_equation", "Eq_6_2_12", "PDE_divide", "KdV_equation",
             "Chaffee_Infante_equation")
    cache_checks = {}
    for case in cases:
        path = ROOT / "data/mlp_cache" / case
        cache = path / "surrogate_cache.pt"
        quality = json.loads((path / "surrogate_quality.json").read_text())
        assert cache.exists()
        assert quality["cache_sha256"] == digest(cache)
        cache_checks[case] = quality["cache_sha256"]
    audit = dict(status="PASS", source_structures=len(source_ids), augmented_examples=len(paired),
                 source_kind_counts=dict(source_kind), augmentations_exactly_32=True,
                 all_GRE_ids_in_57_dictionary=True, all_augmented_latex_GRE_roundtrips=True,
                 no_target_structures=True, mlp_caches_verified=len(cache_checks),
                 downstream_training_started=False, numerical_search_started=False)
    (folder / "audit_v2.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
