"""整理并验证八案例的共享预训练权重和搜索后权重。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import torch

from math_tokenizer.eqgpt.author import tensor_hash


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT.parent / "math_tokenizer_project"
DEST = ROOT / "data/weights"
RUNS = {
    "Wave_equation": "eqgpt_fast_reward_20261022",
    "Convection_diffusion_equation": "eqgpt_fast_reward_Convection_diffusion_equation_20261022",
    "KG_equation": "eqgpt_fast_reward_KG_equation_20261022",
    "Burgers_equation": "eqgpt_fast_reward_burgers_20261022",
    "Eq_6_2_12": "eqgpt_fast_reward_Eq_6_2_12_20261022",
    "PDE_divide": "eqgpt_fast_reward_PDE_divide_20261022",
    "KdV_equation": "eqgpt_fast_reward_KdV_equation_20261022",
    "Chaffee_Infante_equation": "eqgpt_fast_reward_chaffee_20261022",
}
ARMS = ("eqgpt_reference", "ast_bpe_280", "ast_bpe_315", "ast_bpe_350")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(source, expected):
    actual = tensor_hash(torch.load(source, map_location="cpu", weights_only=True))
    if actual != expected:
        raise ValueError(f"张量哈希不匹配：{source}: {actual} != {expected}")
    return dict(bytes=source.stat().st_size, sha256=digest(source), tensor_sha256=actual)


def main():
    if not ORIGINAL.exists():
        raise FileNotFoundError("缺少昨天的本地完整实验目录")
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = dict(version="eqgpt-fast-20261022-weights-v1", pretrained={}, searched={})
    for arm in ARMS:
        source = ORIGINAL / "experiments" / RUNS["Wave_equation"] / "Wave_equation" / arm / "pretrained.pt"
        report = json.loads((ROOT / "experiments/downstream/Wave_equation" / arm / "training.json").read_text())
        record = check(source, report["final_sha256"])
        destination = DEST / "pretrained" / f"{arm}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest["pretrained"][arm] = dict(asset=f"{arm}.pt", path=destination.relative_to(ROOT).as_posix(), **record)
    for case, run in RUNS.items():
        manifest["searched"][case] = {}
        for arm in ARMS:
            source = ORIGINAL / "experiments" / run / case / arm / "searched.pt"
            report = json.loads((ROOT / "experiments/downstream" / case / arm / "search.json").read_text())
            record = check(source, report["history"][-1]["after_sha256"])
            destination = DEST / "searched" / case / f"{arm}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest["searched"][case][arm] = dict(asset=f"{case}_{arm}.pt", path=destination.relative_to(ROOT).as_posix(), **record)
            alias = DEST / f"{case}_{arm}.pt"
            if not alias.exists():
                alias.symlink_to(destination.relative_to(DEST))
        print("WEIGHTS_VERIFIED", case, flush=True)
    target = ROOT / "data/weights/manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("权重矩阵校验完成：", len(manifest["pretrained"]), "组预训练，", sum(map(len, manifest["searched"].values())), "组搜索后权重")


if __name__ == "__main__":
    main()
