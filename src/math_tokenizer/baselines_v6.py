"""Official backends with disclosed adapters and no upstream modifications."""
import json
import re
import time
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from .baselines import HFTokenizer, SentencePieceTokenizer, TikToken, ByteTokenizer
from .rl_vocab_v5 import bytealphabet


class PathPieceOfficial:
    def __init__(self,path,train_seconds=0):
        import pathpiece
        # Use byte zero as the library-required special so its slot is one of
        # the 256 byte fallback slots; no hidden 1025th token.
        self.tokenizer=pathpiece.Tokenizer(str(path),special='\x00',greedy=False,random_tiebreaker=False)
        self.payloads={int(k):bytes(v) for k,v in self.tokenizer.get_ids().items()};self.vocab_size=len(self.payloads);self.train_seconds=train_seconds
    def encode(self,text):return self.tokenizer(text)['input_ids'] if text else []
    def decode(self,ids):return self.tokenizer.decode(ids)
    def decode_bytes(self,ids):return b''.join(self.payloads[id] for id in ids)


class ByteWordPiece:
    """Official WordPiece over reversible bytes; explicit whole-input byte escape.

    Official UNK is kept, reported as a byte-escape event. All 256 escape IDs
    are counted. No normalization, trimming, or per-model input exclusions.
    """
    def __init__(self,tokenizer,train_seconds=0):
        self.tokenizer=tokenizer;self.vocab_size=tokenizer.get_vocab_size()+256;self.train_seconds=train_seconds
        self.reverse={c:b for b,c in bytealphabet().items()};self.base=tokenizer.get_vocab_size();self.unk=tokenizer.token_to_id('[UNK]')
        self.input_limit=json.loads(tokenizer.to_str())['model']['max_input_chars_per_word']
        from .frontend_v6 import python_bridge
        self.bridge=python_bridge();first=[];continuations=[]
        for piece,id in tokenizer.get_vocab().items():
            if id==self.unk:continue
            continued=piece.startswith('##');spelling=piece[2:] if continued else piece
            payload=bytes(self.reverse[c] for c in spelling)
            (continuations if continued else first).append((payload,id))
        def trie(items):
            children=[{}];terminals=[-1]
            for payload,id in items:
                node=0
                for b in payload:
                    if b not in children[node]:children[node][b]=len(children);children.append({});terminals.append(-1)
                    node=children[node][b]
                terminals[node]=id
            array=np.full((len(children),256),-1,dtype=np.int32)
            for i,row in enumerate(children):
                for b,child in row.items():array[i,b]=child
            return array,np.array(terminals,dtype=np.int32)
        self.first_children,self.first_terminal=trie(first);self.cont_children,self.cont_terminal=trie(continuations)
    @classmethod
    def train(cls,texts,capacity,path):
        started=time.perf_counter();t=Tokenizer(models.WordPiece(unk_token='[UNK]',max_input_chars_per_word=1000000))
        t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
        t.train_from_iterator(texts,trainers.WordPieceTrainer(vocab_size=capacity-256,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=['[UNK]'],show_progress=False))
        t.save(str(path));return cls(t,time.perf_counter()-started)
    def encode(self,text):
        raw=text.encode()
        if len(raw)>self.input_limit:return [self.base+b for b in raw]
        if len(raw)>256:
            if '[UNK]' in text:return [self.base+b for b in raw]
            ids=self.bridge.wordpiece(raw,self.first_children,self.first_terminal,self.cont_children,self.cont_terminal)
            return ids if ids is not None else [self.base+b for b in raw]
        return self.encode_official(text)
    def encode_official(self,text):
        ids=self.tokenizer.encode(text,add_special_tokens=False).ids
        return [self.base+b for b in text.encode()] if self.unk in ids else ids
    def decode_bytes(self,ids):
        return b''.join(bytes([id-self.base]) if id>=self.base else bytes(self.reverse[c] for c in self.tokenizer.id_to_token(id).removeprefix('##')) for id in ids)
    def decode(self,ids):return self.decode_bytes(ids).decode()


def prepare_llama3_ranks(record):
    """Lossless official HF byte vocabulary -> Meta tiktoken rank file."""
    import base64
    import hashlib
    import json
    from .corpus import ROOT
    from .rl_vocab_v5 import bytealphabet
    destination=ROOT/record['official_ranks_path']
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()==record['official_ranks_sha256']:
        return destination
    payload=(ROOT/record['local']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=record['sha256']:
        raise ValueError('Llama 3 official HF configuration hash mismatch')
    reverse={c:b for b,c in bytealphabet().items()}
    ranks={bytes(reverse[c] for c in token):rank for token,rank in json.loads(payload)['model']['vocab'].items()}
    if len(ranks)!=128000 or set(ranks.values())!=set(range(128000)):
        raise ValueError('Llama 3 base rank IDs are not the official contiguous range')
    data=b''.join(base64.b64encode(token)+b' '+str(rank).encode()+b'\n' for token,rank in sorted(ranks.items(),key=lambda x:x[1]))
    if hashlib.sha256(data).hexdigest()!=record['official_ranks_sha256']:
        raise ValueError('Llama 3 converted rank file does not match pinned hash')
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
    return destination


class Llama3Official:
    """Thin adapter around Meta's pinned, unmodified public tokenizer code."""
    def __init__(self, record):
        import hashlib
        import importlib.util
        from .corpus import ROOT
        source=record['official_code_files']['llama/tokenizer.py']
        path=ROOT/source['local']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Llama 3 official source hash mismatch')
        ranks=ROOT/record['official_ranks_path']
        if hashlib.sha256(ranks.read_bytes()).hexdigest()!=record['official_ranks_sha256']:
            raise ValueError('Llama 3 rank conversion hash mismatch')
        spec=importlib.util.spec_from_file_location('_llama3_official',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.reference=module.Tokenizer(str(ranks));self.tokenizer=self.reference.model
        self.vocab_size=self.reference.n_words;self.train_seconds=0

    def encode(self, text):
        return self.reference.encode(text,bos=False,eos=False,allowed_special=set(),disallowed_special=())

    def decode(self, ids):
        return self.reference.decode(ids)
