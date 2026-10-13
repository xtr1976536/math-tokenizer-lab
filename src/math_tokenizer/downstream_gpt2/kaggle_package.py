"""Build private Kaggle assets and one-script CPU/GPU kernel submissions."""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from math_tokenizer.corpus import ROOT
from .adapters import METHODS, kernel_slug
from .data import UPSTREAM_SHA
from .kaggle_runtime import ASSET_SLUG, DATA_KERNEL_SLUG, OWNER

BASE = "math-tokenizer-gpt2"
OUTPUT = ROOT / "tmp/kaggle_downstream_gpt2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formula_hashes(destination):
    catalog = set()
    for path in (ROOT / "data/benchmark/formula_only").rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            text = "".join(json.loads(line).split())
            catalog.add(hashlib.sha256(text.encode()).hexdigest())
    destination.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(destination,"wt") as stream:
        json.dump(sorted(catalog),stream)
    return len(catalog)


def bundle():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    staging = OUTPUT / "project"
    if staging.exists():
        shutil.rmtree(staging)
    project = staging / "math_tokenizer_project"
    project.mkdir(parents=True)
    for folder in ("src","native","models"):
        shutil.copytree(ROOT/folder,project/folder,
                        ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copy2(ROOT/"pyproject.toml",project/"pyproject.toml")
    (project/"tmp").mkdir()
    shutil.copytree(ROOT/"tmp/SymbolicMathematics",project/"tmp/SymbolicMathematics",
                    ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(ROOT/"tmp/gpt2-607a30d",project/"tmp/gpt2-607a30d",
                    ignore=shutil.ignore_patterns(".cache","*.lock"))
    tiktoken_cache=Path(tempfile.gettempdir())/"data-gym-cache"
    if not tiktoken_cache.exists():
        raise FileNotFoundError("populate local tiktoken ranks cache before Kaggle packaging")
    shutil.copytree(tiktoken_cache,project/"tiktoken_cache")
    count = _formula_hashes(project/"data/benchmark/formula_hashes.json.gz")
    manifest = {"repository_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                "source_sha":UPSTREAM_SHA, "gpt2_revision":"607a30d783dfa663caf39e06633721c8d4cfcd7e",
                "gpt2_weights_sha256":sha(project/"tmp/gpt2-607a30d/model.safetensors"),
                "formula_catalog_hash_count":count,
                "formula_catalog_hashes_sha256":sha(project/"data/benchmark/formula_hashes.json.gz")}
    (project/"asset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    archive = OUTPUT / "dataset/project_bundle.tar.gz"
    archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,"w:gz",compresslevel=1) as tar:
        tar.add(project,arcname="math_tokenizer_project")
    (archive.parent/"dataset-metadata.json").write_text(json.dumps({
        "title":"Private Math Tokenizer GPT-2 Research Assets",
        "id":f"{OWNER}/{ASSET_SLUG}",
        "licenses":[{"name":"other"}],
        "description":"Private academic research bundle. Includes OpenAI GPT-2 weights, pinned SymbolicMathematics source under CC BY-NC 4.0, project code and tokenizer checkpoints. Formula catalog strings are excluded; only normalized SHA-256 hashes are included."
    },indent=2)+"\n")
    (archive.parent/"asset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(archive,archive.stat().st_size,"bytes",flush=True)
    return archive


def _script(phase, method=None):
    return f'''import hashlib, json, pathlib, subprocess, sys, tarfile
ASSET = pathlib.Path("/kaggle/input/{ASSET_SLUG}/project_bundle.tar.gz")
if not ASSET.exists(): raise FileNotFoundError(ASSET)
subprocess.run([sys.executable,"-m","pip","install","-q","transformers==5.19.0","tokenizers==0.23.3",
                "sympy==1.14.0","numexpr==2.14.2","sentencepiece==0.2.2","tiktoken==0.14.0",
                "antlr4-python3-runtime==4.11.1","safetensors","pylatexenc==2.11","openpyxl==3.1.5"],check=True)
with tarfile.open(ASSET,"r:gz") as tar: tar.extractall("/kaggle/working")
project = pathlib.Path("/kaggle/working/math_tokenizer_project")
import os
os.environ["TIKTOKEN_CACHE_DIR"] = str(project/"tiktoken_cache")
manifest = json.loads((project/"asset_manifest.json").read_text())
weights = project/"tmp/gpt2-607a30d/model.safetensors"
if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["gpt2_weights_sha256"]:
    raise RuntimeError("GPT-2 weight hash mismatch")
sys.path.insert(0,str(project/"src"))
from math_tokenizer.downstream_gpt2.kaggle_runtime import main
main({phase!r},{method!r})
'''


def _kernel(directory,slug,title,phase,method=None,gpu=False,kernel_sources=()):
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"run.py").write_text(_script(phase,method))
    metadata={"id":f"{OWNER}/{slug}","title":title,"code_file":"run.py",
              "language":"python","kernel_type":"script","is_private":"true",
              "enable_gpu":"true" if gpu else "false","enable_tpu":"false",
              "enable_internet":"true","machine_shape":"",
              "dataset_sources":[f"{OWNER}/{ASSET_SLUG}"],
              "kernel_sources":list(kernel_sources),"competition_sources":[],"model_sources":[]}
    (directory/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")


def kernels():
    root=OUTPUT/"kernels"
    data_source=f"{OWNER}/{DATA_KERNEL_SLUG}"
    _kernel(root/"smoke",f"{BASE}-smoke-v1","Math Tokenizer GPT2 Smoke v1","smoke",gpu=True)
    _kernel(root/"data",DATA_KERNEL_SLUG,"Math Tokenizer GPT2 Data v1","data")
    evaluations=[]
    for method in METHODS:
        slug=f"{BASE}-{kernel_slug(method)}-v1"
        _kernel(root/"train"/kernel_slug(method),slug,
                f"Math Tokenizer GPT2 {kernel_slug(method)} v1","train",method,
                gpu=True,kernel_sources=(data_source,))
        eval_slug=f"{BASE}-eval-{kernel_slug(method)}-v1"
        evaluations.append(f"{OWNER}/{eval_slug}")
        _kernel(root/"eval"/kernel_slug(method),eval_slug,
                f"Math Tokenizer GPT2 Eval {kernel_slug(method)} v1","evaluate",method,
                gpu=True,kernel_sources=(data_source,f"{OWNER}/{slug}"))
    _kernel(root/"aggregate",f"{BASE}-aggregate-v1",
            "Math Tokenizer GPT2 Aggregate v1","aggregate",gpu=False,
            kernel_sources=tuple(evaluations))
    print(root)


if __name__=="__main__":
    bundle()
    kernels()
