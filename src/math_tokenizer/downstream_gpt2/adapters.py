"""Frozen tokenizer IDs and exact-byte GPT-2 embedding transfer."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from math_tokenizer.final_models import load_final
from math_tokenizer.rl_vocab_v5 import bytealphabet

GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
METHODS = ("ast_boundary", "official_atomic_bpe", "hf_byte_bpe",
           "sentencepiece_unigram", "Qwen2.5-0.5B",
           "meta-llama/Meta-Llama-3-8B", "tiktoken_o200k_base", "gpt2_native")
SPECIALS = ("BOS", "EOS", "PAD", "INT", "ODE", "IVP", "SEP")


class TokenizerAdapter:
    def __init__(self, name: str, gpt2_path: Path):
        if name not in METHODS:
            raise ValueError(name)
        self.name = name
        self.model = GPT2TokenizerFast.from_pretrained(str(gpt2_path)) if name == "gpt2_native" else load_final(name, "broad", 2048)
        self.base_size = len(self.model) if name == "gpt2_native" else getattr(self.model, "vocab_size", None)
        if self.base_size is None:
            self.base_size = len(self.model.vocab)
        self.special_ids = {token: self.base_size + idx for idx, token in enumerate(SPECIALS)}
        self.vocab_size = self.base_size + len(SPECIALS)
        self._reverse_byte_alphabet = {symbol: byte for byte, symbol in bytealphabet().items()}
        self._payload_cache = {}

    def encode(self, text):
        return self.model.encode(text, add_special_tokens=False) if self.name == "gpt2_native" else self.model.encode(text)

    def decode(self, ids):
        return self.model.decode(ids)

    def payload(self, index):
        if index in self._payload_cache:
            return self._payload_cache[index]
        if index < 0 or index >= self.base_size:
            return None
        obj = self.model
        if self.name == "gpt2_native":
            piece = obj.convert_ids_to_tokens(index)
            value = None if piece == obj.eos_token else self._byte_piece(piece)
        elif hasattr(obj, "vocab"):
            value = obj.vocab[index]
        elif self.name == "sentencepiece_unigram":
            piece = obj.tokenizer.id_to_piece(index)
            if piece in ("<unk>", "<s>", "</s>"):
                value = None
            elif piece.startswith("<0x") and piece.endswith(">"):
                value = bytes([int(piece[3:-1], 16)])
            else:
                value = piece.replace("▁", " ").encode()
        elif self.name in ("hf_byte_bpe", "Qwen2.5-0.5B"):
            piece = obj.tokenizer.id_to_token(index)
            value = None if piece is None or (piece.startswith("<|") and piece.endswith("|>")) else self._byte_piece(piece)
        elif self.name == "meta-llama/Meta-Llama-3-8B":
            if index >= 128000:
                value = None
            else:
                try:
                    value = obj.tokenizer.decode_single_token_bytes(index)
                except (KeyError, ValueError):
                    value = None
        elif self.name == "tiktoken_o200k_base":
            try:
                value = obj.tokenizer.decode_single_token_bytes(index)
            except (KeyError, ValueError):
                value = None
        else:
            raise TypeError(self.name)
        self._payload_cache[index] = value
        return value

    def _byte_piece(self, piece):
        try:
            return bytes(self._reverse_byte_alphabet[c] for c in piece)
        except KeyError:
            return None

    def masked_ids(self):
        return [i for i in range(self.base_size) if self.payload(i) is None] + [
            self.special_ids[k] for k in SPECIALS if k != "EOS"]

    def row(self, task, source, target):
        task_id = {"integral": "INT", "ode_general": "ODE", "ode_ivp": "IVP"}[task]
        prompt = [self.special_ids["BOS"], self.special_ids[task_id]] + self.encode(source) + [self.special_ids["SEP"]]
        output = self.encode(target) + [self.special_ids["EOS"]]
        return prompt, output


class ByteDecomposer:
    def __init__(self, native: TokenizerAdapter):
        if native.name != "gpt2_native":
            raise ValueError("expected native GPT-2 tokenizer")
        self.children = [{}]
        self.terminal = [-1]
        for index in range(native.base_size):
            piece = native.payload(index)
            if not piece:
                continue
            node = 0
            for byte in piece:
                node = self.children[node].setdefault(byte, len(self.children))
                if node == len(self.children):
                    self.children.append({})
                    self.terminal.append(-1)
            self.terminal[node] = index

    def decompose(self, payload: bytes):
        if not payload:
            raise ValueError("empty vocabulary payload")
        n = len(payload)
        count = [10**9] * (n + 1)
        first = [None] * (n + 1)
        count[n] = 0
        for pos in range(n - 1, -1, -1):
            node = 0
            for end in range(pos, n):
                node = self.children[node].get(payload[end])
                if node is None:
                    break
                index = self.terminal[node]
                if index >= 0 and (1 + count[end+1] < count[pos] or
                                   (1 + count[end+1] == count[pos] and (first[pos] is None or index < first[pos][0]))):
                    count[pos], first[pos] = 1 + count[end+1], (index, end+1)
        if first[0] is None:
            raise ValueError(f"GPT-2 cannot decompose {payload!r}")
        result, pos = [], 0
        while pos < n:
            index, end = first[pos]
            result.append((index, (end-pos)/n))
            pos = end
        return result


def initial_embeddings(adapter: TokenizerAdapter, native: TokenizerAdapter,
                       old_weight: torch.Tensor, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    digest.update(GPT2_REVISION.encode())
    digest.update(str(adapter.vocab_size).encode())
    for index in range(adapter.base_size):
        payload = adapter.payload(index)
        digest.update(index.to_bytes(4, "little"))
        digest.update(b"\xff" if payload is None else len(payload).to_bytes(4, "little") + payload)
    key = digest.hexdigest()
    destination = cache_dir / f"{key}.npy"
    if destination.exists():
        result = np.load(destination)
        if result.shape != (adapter.vocab_size, old_weight.size(1)):
            raise ValueError("embedding cache shape mismatch")
        return torch.from_numpy(result.copy()), key
    decomposer = ByteDecomposer(native)
    original = old_weight.detach().cpu().numpy()
    rows = np.empty((adapter.vocab_size, old_weight.size(1)), dtype=np.float32)
    for index in range(adapter.base_size):
        payload = adapter.payload(index)
        if payload is None:
            rows[index] = original.mean(axis=0)
        else:
            fragments = decomposer.decompose(payload)
            rows[index] = sum(original[old_id] * weight for old_id, weight in fragments)
    for name, index in adapter.special_ids.items():
        fragments = decomposer.decompose(("<" + name + ">").encode())
        rows[index] = sum(original[old_id] * weight for old_id, weight in fragments)
    np.save(destination, rows)
    return torch.from_numpy(rows), key


def build_model(adapter: TokenizerAdapter, gpt2_path: Path, cache_dir: Path):
    native = TokenizerAdapter("gpt2_native", gpt2_path)
    model = GPT2LMHeadModel.from_pretrained(str(gpt2_path), local_files_only=True)
    old = model.transformer.wte.weight.detach().clone()
    rows, key = initial_embeddings(adapter, native, old, cache_dir)
    model.resize_token_embeddings(adapter.vocab_size, mean_resizing=False)
    with torch.no_grad():
        model.transformer.wte.weight.copy_(rows)
    model.tie_weights()
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()
    model.config.vocab_size = adapter.vocab_size
    model.config.bos_token_id = adapter.special_ids["BOS"]
    model.config.eos_token_id = adapter.special_ids["EOS"]
    model.config.pad_token_id = adapter.special_ids["PAD"]
    model.generation_config.bos_token_id = adapter.special_ids["BOS"]
    model.generation_config.eos_token_id = adapter.special_ids["EOS"]
    model.generation_config.pad_token_id = adapter.special_ids["PAD"]
    return model, key


def method_slug(name):
    return name.replace("/", "__").replace("-", "_")


def kernel_slug(name):
    short = {"Qwen2.5-0.5B":"qwen25", "meta-llama/Meta-Llama-3-8B":"llama3",
             "tiktoken_o200k_base":"o200k", "sentencepiece_unigram":"sp-unigram",
             "official_atomic_bpe":"atomic-bpe", "hf_byte_bpe":"byte-bpe"}
    return short.get(name,re.sub(r"[^a-z0-9]+","-",name.lower()).strip("-"))
