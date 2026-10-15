"""Frozen author augmentations, held-out equations, and numerical case loading."""
from __future__ import annotations
import ast
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from .author import (ROOT, CODE, UPSTREAM, DICTIONARY, namespace, definitions,
                     source_tree, seed_all)

CASES = ['KdV_equation','Burgers_equation','Wave_equation','Chaffee_Infante_equation',
         'KG_equation','Allen_Cahn','Convection_diffusion_equation','PDE_divide',
         'Eq_6_2_12','PDE_compound','Possion_x_y','Laplacian_smile',
         'Laplacian_EITech','Burgers_2D','Laplacian_H','Laplacian_shuttle']

@contextmanager
def working_directory(path):
    old=Path.cwd();os.chdir(path)
    try: yield
    finally: os.chdir(old)


def canonical(words):
    """Coefficient-free GRE term multiset; commutative +/* permutations."""
    words=[w for w in words if w not in ('S','E')]
    terms=[];factors=[];power=1
    for word in words+['+']:
        if word=='+':
            terms.append(tuple(sorted(factors)));factors=[];power=1
        elif word=='*': power=1
        elif word=='/': power=-1
        else: factors.append((word,power));power=1
    return tuple(sorted(terms))


def augment_training(target_name, folder, seed=525, extra_exclusions=()):
    df=pd.read_excel(CODE/'PDE_dataset_0725.xlsx')
    targets=df[df['Equation name']==target_name]
    forbidden={canonical(s.split(',')) for s in targets['Terms']}
    forbidden.update(canonical(s) for s in extra_exclusions)
    keep=df[[canonical(s.split(',')) not in forbidden for s in df['Terms']]]
    removed=df.loc[~df.index.isin(keep.index)]
    selected=[s.split(',') for s in keep['Terms']]
    env=definitions('read_dataset.py',namespace())
    # The author's get_train_dataset ignores its Equation_name argument. Supply
    # the already-filtered rows through its original read_dataset dependency.
    env['read_dataset']=lambda Equation_name='': [row.copy() for row in selected]
    state=random.getstate();random.seed(seed)
    try:
        with working_directory(CODE): seqs=env['get_train_dataset'](target_name,32)
    finally: random.setstate(state)
    assert len(seqs)==32*len(keep)
    for s in seqs:
        assert s[0]==5 and s.count(1)==1
        assert canonical([DICTIONARY['id2word'][i] for i in s]) not in forbidden
    path=Path(folder);path.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(seqs,separators=(',',':'))
    (path/'train_gre.json').write_text(payload+'\n')
    info=dict(source_formulas=len(df),training_formulas=len(keep),augmentations=32,
              training_sequences=len(seqs),target_name=target_name,
              removed_equations=removed['Equation name'].tolist(),
              source_sha256=hashlib.sha256((CODE/'PDE_dataset_0725.xlsx').read_bytes()).hexdigest(),
              training_sha256=hashlib.sha256(payload.encode()).hexdigest(),
              max_gre_length=max(map(len,seqs)),target_structures_removed=True,
              author_augmentation_interior_eos=sum(s[-1]!=1 for s in seqs),
              augmentation_policy='Unchanged author permutations, including EOS placement')
    (path/'training_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    return seqs,info


class RuntimeCompatibility(ast.NodeTransformer):
    """Execution compatibility only; numerical formulas stay as authored."""
    def __init__(self, cpu=False, smoke_steps=None, checkpoint_interval=None):
        self.cpu=cpu;self.smoke_steps=smoke_steps;self.interval=checkpoint_interval
    def visit_Call(self,node):
        self.generic_visit(node)
        if self.cpu and isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
            node.func.attr='to';node.args=[ast.Name(id='device',ctx=ast.Load())]
        if self.smoke_steps and isinstance(node.func,ast.Name) and node.func.id=='range' and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value in (500,100):
            node.args[0]=ast.Constant(value=self.smoke_steps)
        return node
    def visit_Assign(self,node):
        self.generic_visit(node)
        if self.smoke_steps and any(isinstance(x,ast.Name) and x.id=='iter_num' for x in node.targets):
            node.value=ast.Constant(value=self.smoke_steps)
        return node
    def visit_BinOp(self,node):
        self.generic_visit(node)
        if self.interval and isinstance(node.op,(ast.Mod,ast.Mult)) and isinstance(node.right,ast.Constant) and node.right.value==500:
            node.right=ast.Constant(value=self.interval)
        return node


def runtime_definitions(filename, env, names, device, smoke=False):
    nodes=[n for n in source_tree(filename).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    transform=RuntimeCompatibility(str(device)=='cpu',2 if smoke else None,1 if smoke else None)
    module=ast.fix_missing_locations(transform.visit(ast.Module(body=nodes,type_ignores=[])))
    exec(compile(module,str(CODE/filename),'exec'),env)
    return env


def data_root():
    candidates=[UPSTREAM,ROOT/'vendor/eqgpt_original']
    for p in candidates:
        if (p/'data').exists(): return p
    raise FileNotFoundError('Run scripts/eqgpt/fetch_data.py before the experiment')


def case_environment(equation, observations=10000, noise=0, device='cpu', seed=525):
    if equation not in CASES: raise ValueError(equation)
    env=namespace(device,Equation_name=equation,choose=observations,noise_level=noise,
                  noise_type='Gaussian',trail_num='paired',Learning_Rate=.001)
    tree=source_tree('surrogate_model.py');nodes=[];started=False
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='Equation_name' for t in node.targets):started=True
        if not started:continue
        if isinstance(node,ast.If) and 'gpt_model/PDEGPT_' in ast.unparse(node.test):break
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('Equation_name','choose','noise_level','noise_type','trail_num','Learning_Rate') for t in node.targets):continue
        nodes.append(node)
    module=ast.Module(body=nodes,type_ignores=[])
    class Paths(ast.NodeTransformer):
        def visit_JoinedStr(self,node):
            s=ast.unparse(node)
            if 'Equation_name}burgers_sine.mat' in s:
                return ast.parse("f'data/{Equation_name}/burgers_sine.mat'",mode='eval').body
            if s in ("f'{Equation_name}/Laplacian_H.dat'", "f'{Equation_name}/Burgers2D.mat'"):
                node.values.insert(0,ast.Constant(value='data/'))
            return node
    module=ast.fix_missing_locations(Paths().visit(module))
    if not hasattr(np,'NaN'):np.NaN=np.nan
    with working_directory(data_root()):exec(compile(module,str(CODE/'surrogate_model.py'),'exec'),env)
    if 'data_path' in env:env['data_path']=str((data_root()/env['data_path']).resolve())
    definitions('neural_network.py',env)
    definitions('calculate_terms.py',env)
    seed_all(seed)
    un=env['un']
    if hasattr(un,'to_numpy'):un=un.to_numpy()
    un=np.asarray(un).copy()
    # This is exactly the author's Gaussian expression, frozen before any arm.
    un=un+(noise/100)*np.std(un)*np.random.randn(*un.shape)
    env['un']=un
    if equation in env['canonical_PDEs']:
        env.update(x_num=len(env['x']),t_num=len(env['t']),choose_validate=5000,meta_data_num=10000)
        env['total']=env['x_num']*env['t_num']
    elif equation in env['irrgular_2D_regions']:
        env.update(x_num=len(env['x']),y_num=len(env['y']),choose_validate=1000,meta_data_num=10000,delete_num=10,window_num=10)
        env['total']=env['x_num']*env['y_num']
        env['t']=env['y'] # PINN's function calls its second coordinate t.
    else:
        env.update(x_num=len(env['x']),choose_validate=1000,meta_data_num=10000)
    return env


def build_surrogate(env, device='cpu', seed=525):
    seed_all(seed)
    dimensions=3 if env['Equation_name'] in ['Laplacian_shuttle','Laplacian_H','Burgers_2D'] else 2
    return env['NN'](Num_Hidden_Layers=5,Neurons_Per_Layer=50,Input_Dim=dimensions,
                     Output_Dim=1,Data_Type=torch.float32,Device=torch.device(device),
                     Activation_Function=env['Activation_function'],Batch_Norm=False)
