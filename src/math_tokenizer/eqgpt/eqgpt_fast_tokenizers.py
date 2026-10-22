"""Persist fixed-budget compact AST-BPE models for quick downstream runs."""
import hashlib
import json

from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet,train_compact


def train_or_load(texts,folder,sizes,alphabet):
    folder.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
    manifest=folder/'manifest.json'
    expected=dict(corpus_sha256=fingerprint,sizes=list(sizes),alphabet=list(alphabet))
    if manifest.exists():
        assert json.loads(manifest.read_text())==expected
    else:manifest.write_text(json.dumps(expected,indent=2)+'\n')
    models={}
    for size in sizes:
        for method in ('ast_bpe','soft_ast_bpe'):
            path=folder/f'{method}_{size}.model.json'
            if path.exists():model=CompactAlphabet.load(path)
            else:
                model=train_compact(method,texts,size,alphabet)
                model.save(path)
            assert len(model.vocab)+3==size
            assert all(model.decode(model.encode(text))==text for text in texts[:20])
            models[(method,size)]=model
            print('TOKENIZER_READY',method,size,flush=True)
    return models
