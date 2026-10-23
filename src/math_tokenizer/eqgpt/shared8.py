"""Train one frozen-weight generator per method after excluding all eight targets."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .author import verify_source
from .data import augment_training
from .pde_frozen import load_frozen
from .frozen_math import sha, text_digest
from .latex_codec import structure_to_latex
from .models import build_models
from .round1_config import LATEX_METHODS, METHODS
from .round1_runtime import runtime_sha256, train_round1
from .shared8_config import SHARED8_CASES, SHARED8_TARGETS, SHARED8_VERSION, SHARED8_EPOCHS, SHARED_POSITION_CAPACITY
from .tokenizer_adapters import Adapter


def prepare_shared_pretrain(folder, frozen_assets, seed=525, device='cpu', smoke=False, pilot_epochs=None):
    folder=Path(folder).resolve();frozen_assets=Path(frozen_assets).resolve()
    if seed!=525:raise ValueError('The eight-case protocol fixes seed 525')
    if device not in ('cpu','cuda'):raise ValueError('Unsupported pretraining device')
    if pilot_epochs is not None and (smoke or pilot_epochs<1):raise ValueError('Pilot pretraining requires full data and positive epochs')
    verify_source();adapters,frozen=load_frozen(frozen_assets)
    adapters={name:adapters[name] for name in LATEX_METHODS}
    adapters['eqgpt_reference']=Adapter('eqgpt')
    path=folder/'shared_pretrain.json'
    if path.exists():
        report=json.loads(path.read_text())
        if (report['smoke'],report['pilot_epochs'])!=(smoke,pilot_epochs):
            raise ValueError('Existing shared checkpoints use a different training budget')
        validate_shared_pretrain(folder,frozen_assets,seed)
        return report
    folder.mkdir(parents=True,exist_ok=True)
    sequences,source=augment_training(SHARED8_TARGETS,folder,seed)
    target_names=set(SHARED8_TARGETS)
    if not target_names.issubset(set(source['removed_equations'])):
        raise RuntimeError('A target handbook equation survived exclusion')
    texts=[structure_to_latex(s) for s in sequences]
    if text_digest(texts)!=frozen['corpus']['texts_sha256']:
        raise ValueError('Frozen PDE vocabulary was trained on different LaTeX formulas')
    if max(len(adapter.sequence(text)) for name,adapter in adapters.items() if name!='eqgpt_reference' for text in texts)>SHARED_POSITION_CAPACITY:
        raise ValueError('Training formula exceeds the common position capacity')
    if max(len(adapters['eqgpt_reference'].sequence(sequence)) for sequence in sequences)>SHARED_POSITION_CAPACITY:
        raise ValueError('Author GRE training formula exceeds the common position capacity')
    models,initial=build_models(adapters,SHARED_POSITION_CAPACITY,seed,device)
    training={}
    for name,model in models.items():
        encoded=[adapters[name].sequence(s) for s in (sequences if name=='eqgpt_reference' else texts)]
        if smoke:encoded=encoded[:4]
        epochs=pilot_epochs if pilot_epochs is not None else (1 if smoke else SHARED8_EPOCHS)
        training[name]=train_round1(model,encoded,folder/name,device,seed,smoke,epochs)
    report=dict(version=SHARED8_VERSION,cases=list(SHARED8_CASES),targets=list(SHARED8_TARGETS),
        seed=seed,smoke=smoke,pilot_epochs=pilot_epochs,device=device,methods=list(METHODS),source=source,
        frozen_sha256=sha(frozen_assets/'FROZEN.json'),runtime_sha256=runtime_sha256(shared8=True),
        training=training,model_initialization=initial,
        training_gre_sha256=sha(folder/'train_gre.json'),
        tokenizer_corpus_sha256=frozen['corpus']['texts_sha256'])
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    validate_shared_pretrain(folder,frozen_assets,seed)
    return report


def validate_shared_pretrain(folder,frozen_assets,seed=525):
    folder=Path(folder).resolve();frozen_assets=Path(frozen_assets).resolve()
    report=json.loads((folder/'shared_pretrain.json').read_text())
    expected=(SHARED8_VERSION,list(SHARED8_CASES),list(SHARED8_TARGETS),list(METHODS),seed)
    actual=tuple(report[k] for k in ('version','cases','targets','methods','seed'))
    if actual!=expected or report['frozen_sha256']!=sha(frozen_assets/'FROZEN.json'):
        raise ValueError('Shared eight-case pretraining configuration differs')
    if report['runtime_sha256']!=runtime_sha256(shared8=True) or report['training_gre_sha256']!=sha(folder/'train_gre.json'):
        raise ValueError('Shared eight-case source or formulas changed')
    sequences=json.loads((folder/'train_gre.json').read_text())
    if len(sequences)!=report['source']['training_sequences']:
        raise ValueError('Shared eight-case training length differs')
    for name in METHODS:
        recorded=report['training'][name]
        expected_epochs=report['pilot_epochs'] or (1 if report['smoke'] else SHARED8_EPOCHS)
        expected_samples=4 if report['smoke'] else len(sequences)
        if (recorded['epochs'],recorded['samples'],recorded['batch_size'])!=(expected_epochs,expected_samples,2 if report['smoke'] else 128):
            raise ValueError('Shared checkpoint training budget differs: '+name)
        weights=torch.load(folder/name/'pretrained.pt',map_location='cpu',weights_only=True)
        from .author import tensor_hash
        if tensor_hash(weights)!=recorded['final_sha256']:
            raise ValueError('Shared checkpoint differs: '+name)
    return report
