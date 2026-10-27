"""加载上游最新实验的四个分词器。"""
from pathlib import Path
import argparse

from tokenizers import Tokenizer

from math_tokenizer.baselines import HFTokenizer, TikToken
from math_tokenizer.baselines_v6 import Llama3Official
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
from math_tokenizer.corpus import ROOT
import json


def load(method, fold="pde_fold_1", size=256):
    if method == "hard_ast_bpe":
        path = ROOT / f"experiments/upstream_main/{fold}/ast_bpe_{size}.model.json"
        if not path.exists():
            path = ROOT / f"experiments/upstream_vocab_size/{fold}/ast_bpe_{size}.model.json"
        return CompactAlphabet.load(path)
    if method == "qwen2.5":
        return HFTokenizer(Tokenizer.from_file(str(ROOT / "data/tokenizers/Qwen2.5-0.5B/qwen-tokenizer.json")))
    if method == "llama3":
        record = json.loads((ROOT / "data/tokenizers/llama3_provenance.json").read_text())
        return Llama3Official(record)
    if method == "o200k_base":
        return TikToken("o200k_base")
    raise ValueError(f"未知分词器：{method}")


def main():
    parser = argparse.ArgumentParser(description="查看当前四个分词器的分词结果")
    parser.add_argument("method", choices=("hard_ast_bpe", "qwen2.5", "llama3", "o200k_base"))
    parser.add_argument("formula")
    parser.add_argument("--fold", default="pde_fold_1")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    model = load(args.method, args.fold, args.size)
    ids = model.encode(args.formula)
    print(f"方法：{args.method}；token 数：{len(ids)}")
    print(ids)
    assert model.decode(ids) == args.formula


if __name__ == "__main__":
    main()
