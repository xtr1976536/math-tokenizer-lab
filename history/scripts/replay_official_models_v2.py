"""Compare saved official adapters to recorded per-formula counts/lossless fields."""
import json,sys
from pathlib import Path
from tokenizers import Tokenizer
import sentencepiece as spm
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl
from math_tokenizer.baselines import HFTokenizer,SentencePieceTokenizer,TikToken,ByteTokenizer

out=ROOT/sys.argv[1];protocol=json.loads((out/'protocol.json').read_text());ids=set(protocol['test_ids']);records={r['id']:r for r in read_jsonl(ROOT/'data/releases/v2/curated/corpus.jsonl') if r['id'] in ids}
rows=read_jsonl(out/'per_formula.jsonl');models=out/'models';qwen=json.loads((ROOT/'data/baseline_registry.json').read_text())['qwen']
adapters={'hf_byte_bpe':HFTokenizer(Tokenizer.from_file(str(models/'hf-bpe.json'))),
 'sentencepiece_bpe':SentencePieceTokenizer(spm.SentencePieceProcessor(model_file=str(models/'sentencepiece-bpe.model'))),
 'sentencepiece_unigram':SentencePieceTokenizer(spm.SentencePieceProcessor(model_file=str(models/'sentencepiece-unigram.model'))),
 'utf8_bytes':ByteTokenizer(),'qwen2.5_official':HFTokenizer(Tokenizer.from_file(str(ROOT/qwen['local'])))}
for name in ['gpt2','cl100k_base','o200k_base']:adapters['tiktoken_'+name]=TikToken(name)
summary=[]
for name,model in adapters.items():
 rr=[r for r in rows if r['method']==name]
 for row in rr:
  text=records[row['id']]['latex'];encoded=model.encode(text);assert len(encoded)==row['tokens'];assert model.decode(encoded)==text
 summary.append({'method':name,'records_replayed':len(rr),'token_counts_and_losslessness_equal':True})
(out/'official_replay.json').write_text(json.dumps(summary,indent=2));print('Official baseline replay passed',len(summary))
