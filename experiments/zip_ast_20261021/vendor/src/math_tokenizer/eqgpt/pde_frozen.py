"""Frozen 896-ID vocabularies learned only from held-out EqGPT PDE LaTeX."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import time

from math_tokenizer.corpus import ROOT
from .author import CODE, verify_source
from .data import augment_training
from .frozen_math import sha, text_digest
from .latex_codec import structure_to_latex
from .latex_tokenizers import METHODS, load_tokenizers, train_tokenizers
from .shared8_config import SHARED8_TARGETS

VERSION = 'eqgpt-shared8-author-pde-latex-896-v1'


def prepare(folder):
    folder = Path(folder).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('PDE vocabulary directory must be empty')
    verify_source()
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='eqgpt_pde_vocab_') as temporary:
        sequences, source = augment_training(SHARED8_TARGETS, temporary, 525)
    if source['source_formulas'] != 221 or source['training_formulas'] != 212:
        raise ValueError('Unexpected EqGPT handbook exclusion count')
    if not set(SHARED8_TARGETS).issubset(set(source['removed_equations'])):
        raise ValueError('A held-out target survived exclusion')
    if len(sequences) != 32 * source['training_formulas']:
        raise ValueError('Author augmentation count changed')
    texts = [structure_to_latex(sequence) for sequence in sequences]
    archive = folder / 'pde_latex_train.jsonl.gz'
    with gzip.open(archive, 'wt', encoding='utf-8') as stream:
        for text in texts:
            stream.write(json.dumps(text, ensure_ascii=False) + '\n')
    corpus = dict(source='EqGPT author PDE_dataset_0725.xlsx', source_formulas=221,
                  excluded_equations=source['removed_equations'], training_formulas=212,
                  augmentations_per_formula=32, training_strings=len(texts),
                  unique_training_strings=len(set(texts)),
                  texts_sha256=text_digest(texts), source_sha256=sha(CODE / 'PDE_dataset_0725.xlsx'),
                  author_training_sha256=source['training_sha256'],
                  source_used_for_transformer_pretraining=True,
                  target_policy='Same eight targets and canonical aliases excluded from vocabulary and GPT training')
    started = time.perf_counter()
    adapters, _ = train_tokenizers(texts, folder / 'tokenizers', mode='fixed', capacity=896)
    if set(adapters) != set(METHODS) or any(a.vocab_size != 896 for a in adapters.values()):
        raise ValueError('Tokenizer capacity or methods changed')
    files = [archive, *(folder / 'tokenizers').glob('*.json')]
    sources = ['hard_ast_2026.py', 'ast_boundary_v8.py', 'atomic_bpe_v7.py',
               'rl_route_v6.py', 'atoms_v4.py', 'frontend_v6.py', 'ast_route_v6.py',
               'eqgpt/pde_frozen.py', 'eqgpt/latex_tokenizers.py', 'eqgpt/latex_codec.py']
    frozen = dict(version=VERSION, vocab_size=896, methods=list(METHODS), corpus=corpus,
                  files={str(path.relative_to(folder)): sha(path) for path in files},
                  tokenizer_source_sha256={name: sha(ROOT / 'src/math_tokenizer' / name) for name in sources},
                  native_source_sha256={path.name: sha(path) for path in sorted((ROOT / 'native').glob('*.c'))},
                  elapsed_tokenizer_seconds=time.perf_counter()-started)
    (folder / 'FROZEN.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + '\n')
    return frozen


def load_frozen(folder):
    folder = Path(folder).resolve()
    frozen = json.loads((folder / 'FROZEN.json').read_text())
    if frozen['version'] != VERSION or frozen['vocab_size'] != 896 or frozen['methods'] != list(METHODS):
        raise ValueError('Wrong PDE vocabulary release')
    if frozen['corpus']['source_sha256'] != sha(CODE / 'PDE_dataset_0725.xlsx'):
        raise ValueError('EqGPT handbook changed')
    for name, expected in frozen['files'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or sha(path) != expected:
            raise ValueError('Frozen PDE asset mismatch: ' + name)
    for name, expected in frozen['tokenizer_source_sha256'].items():
        if sha(ROOT / 'src/math_tokenizer' / name) != expected:
            raise ValueError('Frozen PDE tokenizer source changed: ' + name)
    for name, expected in frozen['native_source_sha256'].items():
        if sha(ROOT / 'native' / name) != expected:
            raise ValueError('Frozen PDE native source changed: ' + name)
    adapters = load_tokenizers(folder / 'tokenizers')
    if any(adapter.vocab_size != 896 for adapter in adapters.values()):
        raise ValueError('PDE vocabulary capacity changed')
    return adapters, frozen
