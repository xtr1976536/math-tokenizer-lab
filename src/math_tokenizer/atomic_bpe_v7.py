"""Unmodified HF BPE with the same training-derived lexical alphabet.

The external reversible codec is identical to v5. Every primitive and BPE ID
counts toward capacity. No custom merge algorithm is used by this baseline.
"""
from dataclasses import asdict
import json,time
from tokenizers import Tokenizer,models,trainers
from .rl_route_v6 import AtomCodec


class AtomicBPE:
    def __init__(self, atoms, tokenizer, train_seconds=0):
        self.atoms=list(atoms);self.codec=AtomCodec(atoms);self.tokenizer=tokenizer
        self.vocab=[]
        for id in range(tokenizer.get_vocab_size()):
            spelling=tokenizer.id_to_token(id)
            self.vocab.append(b''.join(atoms[self.codec.symbol_reverse[c]] for c in spelling))
        self.capacity=len(self.vocab);self.train_seconds=train_seconds

    @classmethod
    def train(cls, rows, atoms, capacity):
        started=time.perf_counter();codec=AtomCodec(atoms)
        tokenizer=Tokenizer(models.BPE())
        tokenizer.train_from_iterator((codec.symbols(r['latex']) for r in rows),
            trainers.BpeTrainer(vocab_size=capacity,min_frequency=2,show_progress=False,
                initial_alphabet=list(codec.symbol_map.values()),special_tokens=[]))
        obj=cls(atoms,tokenizer,time.perf_counter()-started)
        assert len(obj.vocab)<=capacity and all(bytes([b]) in obj.vocab for b in range(256))
        return obj

    def encode(self, text):return self.tokenizer.encode(self.codec.symbols(text),add_special_tokens=False).ids
    def encode_batch(self, texts):return [e.ids for e in self.tokenizer.encode_batch([self.codec.symbols(t) for t in texts],add_special_tokens=False)]
    def decode_bytes(self, ids):return b''.join(self.vocab[i] for i in ids)
    def decode(self, ids):return self.decode_bytes(ids).decode()
    def clear_cache(self):
        if hasattr(self.tokenizer.model,'clear_cache'):self.tokenizer.model.clear_cache()
    def save(self, path):
        path.write_text(json.dumps({'format':'official-atomic-bpe-v7','atom_hex':[p.hex() for p in self.atoms],
            'tokenizer':json.loads(self.tokenizer.to_str()),'train_seconds':self.train_seconds,'capacity':self.capacity},ensure_ascii=False,indent=2)+'\n')
    @classmethod
    def load(cls, path):
        d=json.loads(path.read_text())
        return cls([bytes.fromhex(p) for p in d['atom_hex']],Tokenizer.from_str(json.dumps(d['tokenizer'])),d['train_seconds'])
