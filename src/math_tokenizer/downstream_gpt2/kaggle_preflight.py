"""Final local package and DAG review before submitting private Kaggle jobs."""
from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

from math_tokenizer.corpus import ROOT
from .adapters import METHODS, kernel_slug
from .kaggle_package import OUTPUT, bundle, kernels


def run():
    archive=bundle()
    kernels()
    jobs=list((OUTPUT/"kernels").rglob("kernel-metadata.json"))
    assert len(jobs)==19,len(jobs)
    inventory={}
    for path in jobs:
        meta=json.loads(path.read_text())
        slug=meta["id"].split("/",1)[1]
        assert meta["title"].lower().replace(" ","-")==slug,(meta["title"],slug)
        assert meta["is_private"]=="true"
        assert meta["enable_internet"]=="true"
        assert len(meta["dataset_sources"])==1
        script=(path.parent/"run.py").read_text()
        compile(script,str(path.parent/"run.py"),"exec")
        assert 'INPUT.rglob("asset_manifest.json")' in script
        assert "shutil.copytree(PROJECTS[0],project)" in script
        inventory[slug]={"gpu":meta["enable_gpu"],"kernel_sources":meta["kernel_sources"]}
    # Execute the exact generated bootstrap locator against Kaggle's nested
    # /input/datasets-style layout before any remote push.
    smoke=(OUTPUT/"kernels/smoke/run.py").read_text()
    prefix=smoke.split("subprocess.run(",1)[0]
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        project=root/"datasets/xiaotongrenkk/private-asset/project_bundle/math_tokenizer_project"
        (project/"tmp/gpt2-607a30d").mkdir(parents=True)
        (project/"asset_manifest.json").write_text("{}")
        (project/"tmp/gpt2-607a30d/model.safetensors").write_bytes(b"test")
        simulated=prefix.replace('pathlib.Path("/kaggle/input")',f"pathlib.Path({str(root)!r})")
        namespace={}
        exec(compile(simulated,"bootstrap_locator","exec"),namespace)
        assert namespace["PROJECTS"]==[project]
    for method in METHODS:
        train=f"math-tokenizer-gpt2-{kernel_slug(method)}-v1"
        evaluate=f"math-tokenizer-gpt2-eval-{kernel_slug(method)}-v1"
        assert inventory[train]["gpu"]==inventory[evaluate]["gpu"]=="true"
        assert inventory[train]["kernel_sources"]==["xiaotongrenkk/math-tokenizer-gpt2-data-v1"]
        assert len(inventory[evaluate]["kernel_sources"])==2
    assert len(inventory["math-tokenizer-gpt2-aggregate-v1"]["kernel_sources"])==8
    with tarfile.open(archive,"r:gz") as tar:
        names=tar.getnames()
    assert any(name.endswith("model.safetensors") for name in names)
    assert any(name.endswith("formula_hashes.json.gz") for name in names)
    assert any(name.endswith("source_revision.txt") for name in names)
    assert not any("formula_only" in name or "kaggle.json" in name for name in names)
    report={"status":"passed","private_jobs":len(jobs),"dataset_asset_bytes":archive.stat().st_size,
            "contains_formula_strings":False,"contains_credentials":False,"jobs":inventory}
    destination=ROOT/"tmp/downstream_gpt2/local_review/round_5_kaggle_preflight.json"
    destination.write_text(json.dumps(report,indent=2)+"\n")
    print(destination)
    return report


if __name__=="__main__":
    run()
