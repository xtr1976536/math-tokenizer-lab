"""Compact output IDs with unchanged AST-BPE training/segmentation algorithms.

Unused single-byte entries are internal scaffolding only, never model IDs.
Inputs outside the declared byte alphabet are explicitly rejected.
"""
import json
from pathlib import Path

from math_tokenizer.ast_boundary_v8 import BoundaryAST
from math_tokenizer.hard_ast_2026 import HardASTBPE
from .latex_tokenizers import train_one


NUMERIC_BYTES = b"0123456789.+-eE"


def derive_alphabet(texts):
    return sorted(set(NUMERIC_BYTES).union(*(set(text.encode()) for text in texts)))


class CompactAlphabet:
    def __init__(self, method, alphabet, internal):
        self.method = method
        self.alphabet = frozenset(alphabet)
        self.internal = internal
        kept = [i for i, payload in enumerate(internal.vocab)
                if len(payload) != 1 or payload[0] in self.alphabet]
        self.vocab = [internal.vocab[i] for i in kept]
        self.remap = {old: new for new, old in enumerate(kept)}
        self.reverse = kept
        self.capacity = len(self.vocab)

    def encode(self, text):
        missing = set(text.encode()) - self.alphabet
        if missing:
            raise ValueError("Input outside compact alphabet: " + ",".join(f"0x{x:02x}" for x in sorted(missing)))
        return [self.remap[i] for i in self.internal.encode(text)]

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode()

    def save(self, path):
        path = Path(path)
        backend = path.with_name(path.stem + ".backend.json")
        self.internal.save(backend)
        path.write_text(json.dumps(dict(format="compact-alphabet-ast-bpe-v1", method=self.method,
            alphabet=sorted(self.alphabet), content_ids=len(self.vocab), total_ids=len(self.vocab)+3,
            backend=backend.name, unsupported_input="explicit ValueError; no silent UNK"), indent=2)+"\n")

    @classmethod
    def load(cls, path):
        path = Path(path)
        config = json.loads(path.read_text())
        if config["format"] != "compact-alphabet-ast-bpe-v1":
            raise ValueError("Wrong compact alphabet format")
        loader = HardASTBPE if config["method"] == "ast_bpe" else BoundaryAST
        model = cls(config["method"], config["alphabet"], loader.load(path.parent / config["backend"]))
        if len(model.vocab) != config["content_ids"]:
            raise ValueError("Compact vocabulary size mismatch")
        return model


def train_compact(method, texts, total_ids, alphabet):
    if method not in ("ast_bpe", "soft_ast_bpe"):
        raise ValueError(method)
    observed = set().union(*(set(text.encode()) for text in texts))
    if not observed <= set(alphabet):
        raise ValueError("Training data outside alphabet")
    # Original trainers reserve all 256 bytes. Reclaim only absent byte slots;
    # their empty occurrence counts cannot affect any merge or AST boundary.
    internal = train_one(method, texts, total_ids + 256 - len(alphabet))
    result = CompactAlphabet(method, alphabet, internal)
    if len(result.vocab) + 3 != total_ids:
        raise ValueError("Requested vocabulary could not be filled")
    return result
