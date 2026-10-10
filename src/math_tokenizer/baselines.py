"""Thin adapters; official implementations are used without source modifications."""
from pathlib import Path
import time,json,hashlib
import tokenizers
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders
import sentencepiece as spm
import tiktoken

class ByteTokenizer:
    vocab_size=256;train_seconds=0
    def encode(self,s):return list(s.encode())
    def decode_bytes(self,ids):return bytes(ids)
    def decode(self,ids):return self.decode_bytes(ids).decode()

class HFTokenizer:
    def __init__(self,tokenizer,train_seconds=0):
        self.tokenizer=tokenizer;self.vocab_size=tokenizer.get_vocab_size();self.train_seconds=train_seconds
    @classmethod
    def train(cls,texts,size,path):
        t=Tokenizer(models.BPE())
        t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
        t.decoder=decoders.ByteLevel()
        trainer=trainers.BpeTrainer(vocab_size=size,min_frequency=1,show_progress=False,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=[])
        started=time.perf_counter();t.train_from_iterator(texts,trainer);duration=time.perf_counter()-started
        t.save(str(path));return cls(t,duration)
    def encode(self,s):return self.tokenizer.encode(s,add_special_tokens=False).ids
    def decode(self,ids):return self.tokenizer.decode(ids,skip_special_tokens=False)

class SentencePieceTokenizer:
    def __init__(self,processor,train_seconds=0):
        self.tokenizer=processor;self.vocab_size=processor.vocab_size();self.train_seconds=train_seconds
    @classmethod
    def train(cls,texts,size,kind,path):
        # All normalization, whitespace alteration and unknown loss disabled.
        started=time.perf_counter()
        spm.SentencePieceTrainer.train(sentence_iterator=iter(texts),model_prefix=str(path),vocab_size=size,
            model_type=kind,byte_fallback=True,character_coverage=1.0,
            normalization_rule_name='identity',add_dummy_prefix=False,
            remove_extra_whitespaces=False,escape_whitespaces=True,
            bos_id=-1,eos_id=-1,pad_id=-1,unk_id=0,hard_vocab_limit=False,
            shuffle_input_sentence=False,num_threads=1,max_sentence_length=16384,
            split_by_whitespace=False,split_by_number=False,split_digits=False,
            minloglevel=2)
        return cls(spm.SentencePieceProcessor(model_file=str(path)+'.model'),time.perf_counter()-started)
    def encode(self,s):return self.tokenizer.encode(s,out_type=int)
    def decode(self,ids):return self.tokenizer.decode(ids)

class TikToken:
    def __init__(self,name):
        self.tokenizer=tiktoken.get_encoding(name);self.vocab_size=self.tokenizer.n_vocab;self.train_seconds=0
    def encode(self,s):return self.tokenizer.encode(s,allowed_special=set(),disallowed_special=())
    def decode(self,ids):return self.tokenizer.decode(ids)
