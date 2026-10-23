"""Reversible sequence adapters for the author's fixed scientific action space."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .author import DICTIONARY

NAMES = ("eqgpt", "ast_bpe")

class Adapter:
    """A GRE action remains one scientific term/operator in every arm.

    Subword codewords include a comma terminator. They are prefix-free, so a
    product of next-token probabilities defines an unambiguous action score.
    The tokenizer is invoked separately at each GRE action boundary; this is
    necessary to keep the author's alternating scientific mask unchanged.
    """
    def __init__(self, name, model=None):
        if name not in NAMES: raise ValueError(name)
        self.name, self.model = name, model
        self.pad, self.eos, self.bos = (0, 1, 5) if name == 'eqgpt' else (0, 1, 2)
        self.offset = 0 if name == 'eqgpt' else 3
        self.vocab_size = 57 if name == 'eqgpt' else 1024
        if model is not None and len(model.vocab)+self.offset > self.vocab_size:
            raise ValueError('Vocabulary exceeds the fixed 1024-ID capacity')
        self.codewords = {}
        for i, word in enumerate(DICTIONARY['id2word']):
            if i in (0, 5): continue
            self.codewords[i] = [self.eos] if i == 1 else ([i] if name == 'eqgpt' else
                              [int(t) + self.offset for t in model.encode(word + ',')])
        values = list(self.codewords.values())
        for i, a in enumerate(values):
            for j, b in enumerate(values):
                if i != j and b[:len(a)] == a:
                    raise ValueError('GRE codewords must be prefix-free')
        self.reverse = {tuple(v): k for k,v in self.codewords.items()}

    def sequence(self, gre_ids):
        if not gre_ids or gre_ids[0] != 5: raise ValueError('GRE BOS required')
        return [self.bos] + [t for g in gre_ids[1:] for t in self.codewords[int(g)]]

    def prefix(self, gre_ids): return self.sequence(gre_ids)

    def decode_sequence(self, ids):
        if not ids or int(ids[0]) != self.bos: raise ValueError('BOS required')
        result, part = [5], []
        for t in ids[1:]:
            if int(t) == self.pad: break
            part.append(int(t))
            if tuple(part) in self.reverse:
                result.append(self.reverse[tuple(part)]); part = []
        if part: raise ValueError('Incomplete codeword')
        return result

    def manifest(self):
        d = dict(name=self.name,vocab_size=self.vocab_size,pad=self.pad,eos=self.eos,bos=self.bos,
                 serialization='GRE actions; each non-EOS action encoded as exact word plus comma',
                 interface='tokenizer called at GRE action boundaries; author action space retained',
                 codewords={str(k):v for k,v in self.codewords.items()})
        if self.model is not None:
            d['vocab_sha256'] = hashlib.sha256(json.dumps([v.hex() for v in self.model.vocab]).encode()).hexdigest()
        return d


def train_adapters(sequences, folder, capacity=1024):
    from math_tokenizer.hard_ast_2026 import HardASTBPE
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=True)
    texts = [','.join(DICTIONARY['id2word'][int(i)] for i in seq[1:] if i!=1) for seq in sequences]
    hard = HardASTBPE.train(texts,capacity-3)
    hard.save(folder/'ast_bpe.json')
    adapters = {'eqgpt':Adapter('eqgpt'),'ast_bpe':Adapter('ast_bpe',hard)}
    for name, a in adapters.items():
        (folder/f'{name}_adapter.json').write_text(json.dumps(a.manifest(),indent=2)+'\n')
    return adapters


def load_adapters(folder):
    from math_tokenizer.hard_ast_2026 import HardASTBPE
    folder=Path(folder)
    return {'eqgpt':Adapter('eqgpt'),'ast_bpe':Adapter('ast_bpe',HardASTBPE.load(folder/'ast_bpe.json'))}


def common_positions(adapters):
    longest=max(len(c) for a in adapters.values() for c in a.codewords.values())
    # Every author run stops when the GRE sentence reaches length 49. Include
    # an entire candidate codeword for the final autoregressive score.
    return max(50,1+49*longest)
