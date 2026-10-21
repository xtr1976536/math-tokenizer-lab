"""Independent broad-math tokenizer preparation and strict frozen loading.

Only the benchmark's existing train partition is eligible. To avoid opaque PDE
normalizations leaking targets, derivative-bearing relational or multi-term
expressions are conservatively quarantined rather than guessed equivalent.
The public numerical GRE dictionary is not used to train these vocabularies.
"""
from __future__ import annotations
from collections import Counter
import gzip,hashlib,json,re,time
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from .author import CODE,DICTIONARY
from .latex_codec import structure_to_latex,LatexParser,key,LatexError
from .latex_tokenizers import train_tokenizers,load_tokenizers,METHODS

DERIVATIVE=re.compile(r'\\(?:partial|nabla|Delta|dot|ddot)\b|[A-Za-z]\s*_\s*(?:\{[xyzt]+\}|[xyzt])|[A-Za-z][\u2032\u2033\u2034\']')
RELATION_OR_SUM=re.compile(r'=|\+|-|\\(?:leq?|geq?|approx|sim)\b')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def text_digest(texts):return hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()


def handbook_keys():
    import pandas as pd
    words=DICTIONARY['word2id'];result=set();normalized=set()
    for text in pd.read_excel(CODE/'PDE_dataset_0725.xlsx')['Terms']:
        seq=[5]+[words[t] for t in text.split(',')]+[1]
        latex=structure_to_latex(seq)
        result.add(repr(key(LatexParser(latex).parse())))
        normalized.add(re.sub(r'\s+','',latex))
    return result,normalized


def exclusion_reason(row,semantic_keys,normalized):
    if row.get('split')!='train':return 'not_fixed_train_partition'
    if 'eqgpt' in str(row.get('source_id','')).lower():return 'handbook_source'
    text=row['latex']
    if re.sub(r'\s+','',text) in normalized:return 'handbook_exact_or_layout_equivalent'
    # Parse only a supported safe subset. Unknown syntax is never executed.
    if len(text)<4096:
        try:
            if repr(key(LatexParser(text).parse())) in semantic_keys:return 'handbook_semantic_equivalent'
        except (LatexError,ValueError,RecursionError):pass
    if DERIVATIVE.search(text) and RELATION_OR_SUM.search(text):
        return 'quarantine_derivative_equation_or_multiterm_expression'
    return None


def prepare(folder):
    folder=Path(folder).resolve()
    if (folder/'FROZEN.json').exists():raise FileExistsError('Frozen tokenizer release already exists')
    folder.mkdir(parents=True,exist_ok=True);rows=read_catalog();keys,normalized=handbook_keys()
    chosen=[];excluded=[]
    for row in rows:
        reason=exclusion_reason(row,keys,normalized)
        if reason:excluded.append(dict(id=row['id'],reason=reason,split=row['split'],source_id=row['source_id']))
        else:chosen.append(row)
    chosen=sorted(chosen,key=lambda r:r['id'])
    if not chosen:raise ValueError('No independent broad-math training data')
    fields=['id','latex','source_id','source_url','source_revision','source_sha256','license','license_url','split','split_group','domain','template_sha256','canonical_sha256']
    selected_path=folder/'math_train.jsonl.gz'
    with gzip.open(selected_path,'wt',encoding='utf-8') as stream:
        for row in chosen:stream.write(json.dumps({k:row.get(k) for k in fields},ensure_ascii=False)+'\n')
    with gzip.open(folder/'excluded_ids.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in excluded:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest=dict(benchmark_release='active-benchmark-v1',benchmark_total=len(rows),
        original_train=sum(r['split']=='train' for r in rows),accepted=len(chosen),
        accepted_domains=dict(Counter(r['domain'] for r in chosen)),exclusions=dict(Counter(r['reason'] for r in excluded)),
        selected_ids_sha256=hashlib.sha256('\n'.join(r['id'] for r in chosen).encode()).hexdigest(),
        texts_sha256=text_digest([r['latex'] for r in chosen]),
        selected_archive_sha256=sha(selected_path),
        original_catalog_sha256=sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz'),
        handbook_sha256=sha(CODE/'PDE_dataset_0725.xlsx'),
        overlap_policy='All 221 handbook semantic keys screened; train-only; conservative derivative-equation/multiterm quarantine; never asserts complete semantic equivalence detection for arbitrary TeX',
        source_used_for_transformer_pretraining=False)
    (folder/'corpus_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    texts=[r['latex'] for r in chosen];started=time.perf_counter()
    adapters,reports=train_tokenizers(texts,folder/'tokenizers',mode='fixed',capacity=1024)
    source_files=['hard_ast_2026.py','ast_boundary_v8.py','atomic_bpe_v7.py','rl_route_v6.py','atoms_v4.py','frontend_v6.py','ast_route_v6.py',
                  'eqgpt/frozen_math.py','eqgpt/latex_tokenizers.py','eqgpt/latex_codec.py']
    files=[selected_path,folder/'corpus_manifest.json',folder/'excluded_ids.jsonl.gz',*sorted((folder/'tokenizers').glob('*.json'))]
    frozen=dict(version='eqgpt-round1-broad-math-1024-v1',vocab_size=1024,methods=list(METHODS),
        corpus=manifest,files={str(p.relative_to(folder)):sha(p) for p in files},
        tokenizer_source_sha256={p:sha(ROOT/'src/math_tokenizer'/p) for p in source_files},
        native_source_sha256={p.name:sha(p) for p in sorted((ROOT/'native').glob('*.c'))},
        elapsed_tokenizer_seconds=time.perf_counter()-started,
        policy='Load-only in model training. Shared across every target, seed and first-round arm. No automatic vocabulary/capacity search.')
    (folder/'FROZEN.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n')
    return frozen


def load_frozen(folder):
    folder=Path(folder).resolve();path=folder/'FROZEN.json';manifest=json.loads(path.read_text())
    if manifest['version']!='eqgpt-round1-broad-math-1024-v1' or manifest['vocab_size']!=1024:raise ValueError('Wrong tokenizer release')
    for name,expected in manifest['files'].items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder) or sha(p)!=expected:raise ValueError('Frozen asset mismatch: '+name)
    for name,expected in manifest['tokenizer_source_sha256'].items():
        if sha(ROOT/'src/math_tokenizer'/name)!=expected:raise ValueError('Tokenizer source changed: '+name)
    for name,expected in manifest['native_source_sha256'].items():
        if sha(ROOT/'native'/name)!=expected:raise ValueError('Native tokenizer source changed: '+name)
    adapters=load_tokenizers(folder/'tokenizers')
    if any(a.vocab_size!=1024 for a in adapters.values()):raise ValueError('Capacity mismatch')
    return adapters,manifest
