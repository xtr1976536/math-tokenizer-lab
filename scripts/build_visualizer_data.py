"""导出与原生 hard AST-BPE 一致的浏览器词表和基准公式。"""
import gzip
import json
from pathlib import Path

from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
from math_tokenizer.frontend_v6 import structure_native


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/data"
SOURCE = ROOT / "data/benchmark_pde1500_20261022/pde_equations.jsonl.gz"
MODELS = ROOT / "experiments/downstream/Wave_equation/tokenizers"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for size in (280, 315, 350):
        model = CompactAlphabet.load(MODELS / f"ast_bpe_{size}.model.json")
        data = {
            "size": size,
            "alphabet": sorted(model.alphabet),
            "vocab": [part.hex() for part in model.internal.vocab],
            "merges": model.internal.merges,
            "compact_ids": model.reverse,
        }
        (OUT / f"hard-{size}.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    formulas = []
    with gzip.open(SOURCE, "rt") as stream:
        for row in map(json.loads, stream):
            latex = row["latex"]
            formulas.append({
                "id": row["id"], "latex": latex,
                "level": row["complexity_bin"],
                "title": row.get("title"),
                "source_url": row.get("abstract_url"),
                "license": row.get("license"),
                "spans": list(structure_native(latex).spans),
            })
    (OUT / "pde1500.json").write_text(json.dumps(formulas, ensure_ascii=False, separators=(",", ":")) + "\n")
    bundled = {
        "models": {size: json.loads((OUT / f"hard-{size}.json").read_text()) for size in (280, 315, 350)},
        "formulas": formulas,
    }
    (OUT / "离线数据.js").write_text(
        "window.AST_BPE_DATA = " + json.dumps(bundled, ensure_ascii=False, separators=(",", ":")) + ";\n"
    )
    print(f"导出 {len(formulas)} 条公式和 3 个 hard 词表")


if __name__ == "__main__":
    main()
