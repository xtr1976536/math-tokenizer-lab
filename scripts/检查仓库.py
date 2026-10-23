"""发布前三轮可重复检查：内容范围、数据一致性、网页资产。"""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "Wave_equation", "Convection_diffusion_equation", "KG_equation",
    "Burgers_equation", "Eq_6_2_12", "PDE_divide", "KdV_equation",
    "Chaffee_Infante_equation",
)
FOUR = {"ast_bpe", "Qwen2.5-0.5B", "meta-llama/Meta-Llama-3-8B", "tiktoken_o200k_base"}


def audit():
    main = json.loads((ROOT / "experiments/pde1500_main_final_20261022/results.json").read_text())
    sweep = json.loads((ROOT / "experiments/pde1500_sweep_200_450_20261022/results.json").read_text())
    assert len(main) == 16 and {row["method"] for row in main} == FOUR
    assert sweep and {row["method"] for row in sweep} == {"ast_bpe"}
    with (ROOT / "reports/downstream/对比结果.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 32 and Counter(row["method"] for row in rows) == {"ast_bpe": 24, "eqgpt_reference": 8}
    assert {row["case"] for row in rows} == set(CASES)
    for case in CASES:
        folder = ROOT / "experiments/downstream" / case
        assert (ROOT / "third_party/eqgpt/data" / case).is_dir()
        assert (folder / "surrogate/surrogate_cache.pt").is_file()
        assert (folder / "eqgpt_reference/search.json").is_file()
        for size in (280, 315, 350):
            assert (folder / f"ast_bpe_{size}/search.json").is_file()
            assert (folder / "tokenizers" / f"ast_bpe_{size}.model.backend.json").is_file()
        log = (folder / "完整运行日志.log").read_text()
        assert "soft_ast_bpe" not in log
        assert "eqgpt_reference" in log and "ast_bpe_350" in log
    for log in (ROOT / "experiments/upstream").glob("*.log"):
        assert "soft_ast_bpe" not in log.read_text()

    formulas = [json.loads(line) for line in gzip.open(ROOT / "data/benchmark_pde1500_20261022/pde_equations.jsonl.gz", "rt")]
    assert len(formulas) == 1500 and len({row["source_group"] for row in formulas}) == 490
    assert Counter(row["complexity_bin"] for row in formulas) == {"simple": 450, "moderate": 600, "complex": 450}
    web = json.loads((ROOT / "docs/data/pde1500.json").read_text())
    assert [(row["id"], row["latex"]) for row in web] == [(row["id"], row["latex"]) for row in formulas]
    for size in (280, 315, 350):
        model = json.loads((ROOT / f"docs/data/hard-{size}.json").read_text())
        assert len(model["compact_ids"]) + 3 == size
    assert (ROOT / "docs/data/离线数据.js").is_file()
    assert all(row.get("source_url", "").startswith("https://arxiv.org/abs/") for row in web)
    assert all(row.get("title") and row.get("license") for row in web)
    assert not any(host in (ROOT / "docs/index.html").read_text()
                   for host in ("fonts.googleapis.com", "fonts.gstatic.com"))
    for name in ("train_gpt.py", "continue_train_GPT.py", "PINN_optimization.py", "surrogate_model.py"):
        assert (ROOT / "third_party/eqgpt/code" / name).is_file()
    weights = json.loads((ROOT / "benchmark/weights_manifest.json").read_text())
    records = [*weights["pretrained"].values(),
               *(record for case in weights["searched"].values() for record in case.values())]
    assert len(weights["pretrained"]) == 4 and len(weights["searched"]) == 8 and len(records) == 36
    assert len({record["asset"] for record in records}) == 36
    assert len(list((ROOT / "参考资料/PDF").glob("*.pdf"))) == 4
    assert all(path.stat().st_size < 95_000_000 for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts)
    digest = hashlib.sha256((ROOT / "data/benchmark_pde1500_20261022/pde_equations.jsonl.gz").read_bytes()).hexdigest()
    return {"检查": "通过", "上游方法": sorted(FOUR), "下游案例": len(CASES),
            "下游实验": len(rows), "PDE公式": len(formulas), "公式文件SHA256": digest,
            "参考PDF": 4, "已训练权重": len(records), "网页": "离线数据与来源字段齐全"}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
