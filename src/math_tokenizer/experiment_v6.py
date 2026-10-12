"""Checkpoint registry and thin official baseline loaders for all v6 replays."""
import json
from pathlib import Path
import sentencepiece as spm
from tokenizers import Tokenizer
from .corpus import ROOT
from .current_models import load_checkpoint
from .baselines import HFTokenizer,SentencePieceTokenizer,TikToken,ByteTokenizer
from .baselines_v6 import PathPieceOfficial,ByteWordPiece,Llama3Official


def load_entry(entry):
    kind=entry['kind'];path=ROOT/entry['path'] if entry.get('path') else None
    if kind=='proposed':model=load_checkpoint(path)
    elif kind=='hf':model=HFTokenizer(Tokenizer.from_file(str(path)),entry.get('train_seconds',0))
    elif kind=='sp':model=SentencePieceTokenizer(spm.SentencePieceProcessor(model_file=str(path)),entry.get('train_seconds',0))
    elif kind=='wp':model=ByteWordPiece(Tokenizer.from_file(str(path)),entry.get('train_seconds',0))
    elif kind=='pathpiece':model=PathPieceOfficial(path,entry.get('train_seconds',0))
    elif kind=='tiktoken':model=TikToken(entry['encoding'])
    elif kind=='llama3_official':
        record=next(r for r in json.loads((ROOT/'data/pretrained_registry_v6.json').read_text()) if r['repo']==entry['name'])
        model=Llama3Official(record)
    elif kind=='bytes':model=ByteTokenizer()
    else:raise ValueError('unknown model kind '+kind)
    return model


def entries(capacity,scenario):
    out=ROOT/f'results/v6/{scenario}_{capacity}';folder=out/'models';result=[]
    final=json.loads((out/'final_selection.json').read_text());chosen=final['v5_branch']
    def add(name,path,seed=0,group='control',kind='proposed',**extra):
        if path is not None and not Path(path).exists():raise FileNotFoundError(path)
        result.append({'name':name,'path':str(Path(path).relative_to(ROOT)) if path else None,'seed':seed,'group':group,'kind':kind,'capacity':capacity,'scenario':scenario,**extra})
    add('ast_v4',folder/'ast_v4_selected.json',group='matched_primary')
    add('ast_v4_previous_dictionary',folder/'ast_v4_previous_dictionary.json')
    add('ast_v4_previous_base',folder/'ast_v4_previous.json')
    add('v5_previous_local',folder/'v5_previous_local.json',17)
    if (folder/'v5_previous_trajectory.json').exists():add('v5_previous_trajectory',folder/'v5_previous_trajectory.json',17)
    for seed in (17,29,43):
        if (folder/f'v5_selected_{seed}.json').exists():add('v5_trajectory',folder/f'v5_selected_{seed}.json',seed,group='matched_primary' if chosen=='trajectory' else 'research')
        if (folder/f'v5_barrier_selected_{seed}.json').exists():add('v5_barrier',folder/f'v5_barrier_selected_{seed}.json',seed,group='research')
    add('v5_structural',folder/'v5_structural_selected_17.json',0,group='matched_primary' if chosen=='structural' else 'research',deterministic_expected_gradient=True)
    for name,file in [('v5_no_update','v5_no_update.json'),('v5_search','v5_search_control.json'),('v5_atomic_frequency','v5_atomic_frequency.json'),
        ('v5_barrier_no_update','v5_barrier_no_update.json'),('v5_barrier_search','v5_barrier_search_control.json'),('v5_barrier_frequency','v5_barrier_frequency.json'),
        ('v5_structural_frequency','v5_structural_frequency.json'),('v5_structural_reward_greedy','v5_structural_reward_greedy.json')]:
        if (folder/file).exists():add(name,folder/file,17)
    for baseline in json.loads((out/'baselines.json').read_text()):
        kind={'hf_bpe':'hf','sentencepiece_bpe':'sp','sentencepiece_unigram':'sp','wordpiece_bytes':'wp','pathpiece_bpe_vocab':'pathpiece'}[baseline['name']]
        add(baseline['name'],ROOT/baseline['path'],group='matched_primary',kind=kind,train_seconds=baseline['train_seconds'])
    add('utf8_bytes',None,group='floor',kind='bytes')
    return result


def pretrained_entries():
    result=[]
    for r in json.loads((ROOT/'data/pretrained_registry_v6.json').read_text()):
        if r['status']=='downloaded':result.append({'name':r['repo'],'kind':r.get('kind','hf'),'path':r['local'],'group':'fixed_pretrained_reference','seed':0,'scenario':'pretrained','capacity':None})
    qwen=json.loads((ROOT/'data/baseline_registry.json').read_text())['qwen']
    result.append({'name':'Qwen2.5-0.5B','kind':'hf','path':qwen['local'],'group':'fixed_pretrained_reference','seed':0,'scenario':'pretrained','capacity':None})
    for encoding in ('gpt2','cl100k_base','o200k_base'):result.append({'name':'tiktoken_'+encoding,'kind':'tiktoken','encoding':encoding,'path':None,'group':'fixed_pretrained_reference','seed':0,'scenario':'pretrained','capacity':None})
    return result
