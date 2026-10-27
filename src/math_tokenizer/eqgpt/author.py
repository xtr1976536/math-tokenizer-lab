"""Load the author's actual definitions without executing script side effects.

Definitions are compiled from the pinned source AST, not reimplemented. The
original files remain byte-identical. The original source is stored in eqgpt_original/.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
import torch
from sympy import Matrix
from torch import nn, optim
from torch.autograd import Variable
from torch.utils import data as Data
from tqdm import tqdm

from math_tokenizer.corpus import ROOT

UPSTREAM = ROOT / "eqgpt_original"
CODE = UPSTREAM / "code"
DICTIONARY = json.loads((CODE / "dict_datas_0725.json").read_text())
ARCHITECTURE = dict(d_model=768, d_ff=2048, d_k=64, d_v=64,
                    n_layers=6, n_heads=8, max_pos=50, CLIP=1)


def verify_source():
    provenance = json.loads((UPSTREAM / "provenance.json").read_text())
    for path, expected in provenance["files"].items():
        actual = hashlib.sha256((UPSTREAM / path).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Author source changed: {path}")
    return provenance


def source_tree(name):
    text = (CODE / name).read_text()
    if name == "surrogate_model.py":
        # The checked-in release contains an incomplete, unused Left assignment.
        assert text.count("    Left = \n") == 1
        text = text.replace("    Left = \n", "    Left = 'u_t'\n")
    return ast.parse(text, filename=str(CODE / name))


def namespace(device="cpu", **extra):
    env = dict(json=json, torch=torch, nn=nn, optim=optim, np=np, numpy=np,
               pd=pd, scio=scio, random=random, itertools=itertools, math=math,
               time=time, os=os, pickle=pickle, tqdm=tqdm, plt=plt, Data=Data,
               Variable=Variable, Matrix=Matrix, F=nn.functional,
               TensorDataset=Data.TensorDataset, DataLoader=Data.DataLoader,
               device=torch.device(device), word2id=DICTIONARY["word2id"],
               id2word=DICTIONARY["id2word"], vocab_size=len(DICTIONARY["id2word"]),
               __name__="math_tokenizer.eqgpt.author_loaded")
    env.update(ARCHITECTURE)
    env.update(extra)
    return env


def definitions(name, env, names=None):
    tree = source_tree(name)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and (names is None or n.name in names)]
    if names is not None and {n.name for n in nodes} != set(names):
        raise RuntimeError(f"Unknown author definitions in {name}: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CODE/name), "exec"), env)
    return env


def model_namespace(vocab_size=57, max_pos=50, device="cpu", n_layers=None):
    env = namespace(device, vocab_size=vocab_size, max_pos=max_pos)
    if n_layers is not None:
        env['n_layers'] = n_layers
    definitions("gpt_model.py", env)
    return env


def training_namespace(device="cpu"):
    return definitions("train_gpt.py", namespace(device))


def numerical_namespace(device="cpu", equation="KdV_equation"):
    env = namespace(device, Equation_name=equation)
    definitions("neural_network.py", env)
    definitions("calculate_terms.py", env)
    definitions("continue_train_GPT.py", env,
                {"delete_duplicate", "delete_dulplicate_A_column", "calculate_reward",
                 "get_mask_invalid", "find_min_no_repeat", "train_step", "MyDataSet"})
    definitions("PINN_optimization.py", env,
                {"PINNLossFunc", "get_coefficients", "get_PDE_terms", "get_no_boundary"})
    return env


def definition_hashes():
    output = {}
    for filename in ["gpt_model.py", "train_gpt.py", "read_dataset.py",
                     "neural_network.py", "calculate_terms.py", "continue_train_GPT.py",
                     "PINN_optimization.py"]:
        output[filename] = {n.name: hashlib.sha256(ast.dump(n).encode()).hexdigest()
                            for n in source_tree(filename).body
                            if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    return output


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def tensor_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
