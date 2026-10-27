"""下载并加载已训练的八案例模型；可生成候选或复用缓存继续搜索。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from urllib.parse import quote

import torch

from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.author import DICTIONARY, model_namespace, tensor_hash
from math_tokenizer.eqgpt.data import case_environment, build_surrogate
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
from math_tokenizer.eqgpt.latex_generation import generate
from math_tokenizer.eqgpt.latex_tokenizers import LatexAdapter
from math_tokenizer.eqgpt.latex_experiment import RewardEvaluator, search_latex
from math_tokenizer.eqgpt.tokenizer_adapters import Adapter


MANIFEST = ROOT / "data/weights/manifest.json"
VERSION = "eqgpt-fast-20261022-weights-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter_for(case, arm):
    if arm == "eqgpt_reference":
        return Adapter("eqgpt")
    size = int(arm.removeprefix("ast_bpe_"))
    if arm != f"ast_bpe_{size}" or size not in (280, 315, 350):
        raise ValueError("未知模型 arm")
    source_case = "Wave_equation" if case == "new" else case
    source = ROOT / "experiments/downstream" / source_case / "tokenizers" / f"ast_bpe_{size}.model.json"
    return LatexAdapter("ast_bpe", CompactAlphabet.load(source))


def load_checkpoint(case, arm, stage, device):
    manifest = json.loads(MANIFEST.read_text())
    if manifest["version"] != VERSION or (case not in manifest["searched"] and case != "new"):
        raise ValueError("权重清单版本或案例不匹配")
    if case == "new" and stage != "pretrained":
        raise ValueError("新案例尚无搜索后权重，请使用 --stage pretrained")
    record = manifest["pretrained"][arm] if stage == "pretrained" else manifest["searched"][case][arm]
    path = ROOT / record["path"]
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，请先运行本脚本的 --download 命令")
    if sha(path) != record["sha256"]:
        raise ValueError(f"文件哈希不匹配：{path}")
    adapter = adapter_for(case, arm)
    protocol_case = "Wave_equation" if case == "new" else case
    protocol = json.loads((ROOT / "experiments/downstream" / protocol_case / "protocol.json").read_text())
    core = model_namespace(adapter.vocab_size, protocol["positions"], device,
                           n_layers=protocol["layers"])["GPT"]().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    if tensor_hash(state) != record["tensor_sha256"]:
        raise ValueError("张量哈希不匹配")
    core.load_state_dict(state, strict=True)
    core.eval()
    return core, adapter


def download():
    manifest = json.loads(MANIFEST.read_text())
    records = [*manifest["pretrained"].values(),
               *(record for case in manifest["searched"].values() for record in case.values())]
    directory = ROOT / "data/weights/.downloads"
    directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        target = ROOT / record["path"]
        if target.exists() and sha(target) == record["sha256"]:
            print("已有且校验通过：", record["path"])
            continue
        url = ("https://github.com/xtr1976536/math-tokenizer-lab/releases/download/"
               + VERSION + "/" + quote(record["asset"]))
        temporary = directory / (record["asset"] + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "math-tokenizer-lab"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if temporary.stat().st_size != record["bytes"] or sha(temporary) != record["sha256"]:
            temporary.unlink()
            raise ValueError(f"Release 文件大小或哈希不匹配：{record['asset']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(target)
        print("已下载并校验：", record["path"])
    print(f"{len(records)} 个模型权重已就绪：data/weights/")


def load_surrogate(case, device):
    folder = ROOT / "data/mlp_cache" / case
    cache = folder / "surrogate_cache.pt"
    quality = json.loads((folder / "surrogate_quality.json").read_text())
    if sha(cache) != quality["cache_sha256"]:
        raise ValueError("MLP 缓存文件哈希不匹配")
    payload = torch.load(cache, map_location="cpu", weights_only=True)
    report = payload["report"]
    if tensor_hash(payload["state"]) != report["surrogate_sha256"]:
        raise ValueError("MLP 权重张量哈希不匹配")
    coordinates = payload["database"]
    if hashlib.sha256(coordinates.numpy().tobytes()).hexdigest() != report["metadata_sha256"]:
        raise ValueError("MLP 坐标哈希不匹配")
    env = case_environment(case, 10000, 0, device, 525)
    net = build_surrogate(env, device, 525)
    net.load_state_dict(payload["state"])
    net.eval()
    database = coordinates.to(device).requires_grad_(True)
    # 原缓存来自 MPS；跨设备浮点舍入可能使预测字节不同，不把该差异当作权重损坏。
    with torch.no_grad():
        deviation = (net(database).detach().cpu() - payload["predictions"]).abs().max().item()
    if deviation > 1e-4:
        raise ValueError(f"跨设备 MLP 预测误差异常：{deviation}")
    return net, database, deviation


def run(args):
    core, adapter = load_checkpoint(args.case, args.arm, args.stage, args.device)
    if args.case == "new" and args.action != "generate":
        raise ValueError("新数据须先定义观测变量并拟合 surrogate；当前 new 模式只生成候选结构")
    if args.action == "generate":
        if args.arm == "eqgpt_reference":
            env = model_namespace(57, 50, args.device, n_layers=3)
            mask = torch.ones(57, device=args.device)
            for attempt in range(args.attempts):
                sentence = [5]
                while len(sentence) < 49:
                    value, _ = core.step(sentence, mask)
                    sentence.append(int(value))
                    if int(value) == 1:
                        break
                print(attempt + 1, "".join(DICTIONARY["id2word"][i] for i in sentence[1:-1]))
        else:
            rows = generate(core, adapter, attempts=args.attempts, max_tokens=50, seed=args.seed)
            for index, row in enumerate(rows, 1):
                print(index, row.get("text"), row["stop_reason"])
        return
    net, database, deviation = load_surrogate(args.case, args.device)
    print(f"已加载 MLP 缓存；跨设备最大预测差 {deviation:.3g}")
    if args.action == "score":
        if args.arm == "eqgpt_reference":
            raise ValueError("score 接口当前接受 LaTeX 公式，请选择 Hard AST-BPE arm")
        evaluator = RewardEvaluator(net, database, args.case, ["x", "t"], args.device,
                                    0.2, normalize_ast=True)
        result = evaluator.evaluate(dict(stopped=True, text=args.formula))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.stage != "pretrained":
        raise ValueError("继续搜索须从共享预训练权重开始；请指定 --stage pretrained")
    output = ROOT / "experiments/new_search" / args.case / args.arm
    output.mkdir(parents=True, exist_ok=True)
    history = search_latex(core, adapter, net, database, args.case, ["x", "t"], output,
                           args.device, args.seed, attempts=args.attempts, cycles=3,
                           finetune_epochs=2, max_tokens=50, batch_size=16, epi=0.2,
                           reference=args.arm == "eqgpt_reference", gre_limit=50,
                           normalize_ast=args.arm != "eqgpt_reference")
    print("搜索完成；Top-1：", history[-1]["equations"][0] if history[-1]["equations"] else "无可评价候选")


def main():
    parser = argparse.ArgumentParser(description="使用已训练的 Hard AST-BPE 或缩小版 EqGPT 模型")
    parser.add_argument("--download", action="store_true", help="从公开 GitHub Release 下载并校验权重")
    parser.add_argument("--case", default="Wave_equation")
    parser.add_argument("--arm", choices=("eqgpt_reference", "ast_bpe_280", "ast_bpe_315", "ast_bpe_350"), default="ast_bpe_350")
    parser.add_argument("--stage", choices=("pretrained", "searched"), default="searched")
    parser.add_argument("--action", choices=("generate", "score", "search"), default="generate")
    parser.add_argument("--formula", help="score 时要评价的 LaTeX 候选公式")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=525)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.download:
        download(); return
    if args.action == "score" and not args.formula:
        parser.error("score 必须提供 --formula")
    run(args)


if __name__ == "__main__":
    main()
