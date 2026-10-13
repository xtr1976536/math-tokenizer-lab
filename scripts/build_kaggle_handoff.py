"""Create a self-contained user handoff without Kaggle credentials or full runs."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"tmp/kaggle_downstream_gpt2"
DEST=ROOT/"deliverables/math_tokenizer_gpt2_kaggle_handoff"
ARCHIVE=ROOT/"deliverables/math_tokenizer_gpt2_kaggle_handoff.zip"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def build():
    if DEST.exists():
        shutil.rmtree(DEST)
    dataset=DEST/"tmp/kaggle_downstream_gpt2/dataset"
    kernels=DEST/"tmp/kaggle_downstream_gpt2/kernels"
    dataset.parent.mkdir(parents=True)
    shutil.copytree(SOURCE/"dataset",dataset)
    shutil.copytree(SOURCE/"kernels",kernels)
    (DEST/"scripts").mkdir()
    shutil.copy2(ROOT/"scripts/drive_kaggle_downstream.py",DEST/"scripts/drive_kaggle_downstream.py")
    shutil.copy2(ROOT/"docs/downstream_gpt2_kaggle.md",DEST/"PROTOCOL.md")
    audit=DEST/"audits"
    audit.mkdir()
    for name in ("round_1.json","round_2.json","round_3.json","round_4.json",
                 "round_5_kaggle_preflight.json","post_template_audit.json","summary.json"):
        path=ROOT/"tmp/downstream_gpt2/local_review"/name
        if not path.exists():
            raise FileNotFoundError(path)
        shutil.copy2(path,audit/name)
    pilot=DEST/"pilot_data"
    pilot.mkdir()
    for path in (ROOT/"tmp/downstream_gpt2/local_review/data").glob("*.json*"):
        shutil.copy2(path,pilot/path.name)
    revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    bundled=json.loads((dataset/"asset_manifest.json").read_text())
    if bundled["repository_commit"]!=revision:
        raise RuntimeError("rebuild Kaggle assets after committing source changes")
    manifest={"repository_commit":revision,
              "dataset_asset_sha256":sha(dataset/"project_bundle.tar.gz"),
              "dataset_asset_bytes":(dataset/"project_bundle.tar.gz").stat().st_size,
              "pilot_records":sum(1 for p in pilot.glob("*.jsonl") for _ in p.open()),
              "full_data_included":False,"full_data_source_bundled":True,
              "kaggle_credentials_included":False}
    (DEST/"HANDOFF_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    readme=f"""# Mathematical tokenizer GPT-2 Kaggle handoff

This package is pinned to local commit `{revision[:12]}`. It contains the full
research code, frozen seven tokenizers, native GPT-2 calibration tokenizer,
GPT-2 weights, the pinned SymbolicMathematics source generator, Kaggle job
metadata/scripts, local audit records, and a tiny verified example dataset.
The full task dataset is generated on Kaggle from the bundled source; no local
full-data run or external download of the 65M-equation paper archives is needed.

## Verify the archive

`HANDOFF_MANIFEST.json` records the SHA-256 and byte size of
`tmp/kaggle_downstream_gpt2/dataset/project_bundle.tar.gz`. `pilot_data/`
contains {manifest['pilot_records']} small task records plus checksums and
the common-coverage list. `audits/` contains four local smoke-review rounds,
an independent split audit and the Kaggle package review. These are smoke
checks, not downstream performance results.

## Run on the same Kaggle account

The metadata is private and targets account `xiaotongrenkk`. Kaggle already has
an older private asset dataset with this slug; update it from the supplied
folder, then submit the supplied kernels in order:

```bash
kaggle datasets version -p tmp/kaggle_downstream_gpt2/dataset -m 'Audited handoff bundle' -t
kaggle datasets status xiaotongrenkk/math-tokenizer-gpt2-assets-v1-20261013 --format json
kaggle kernels push -p tmp/kaggle_downstream_gpt2/kernels/smoke
kaggle kernels status xiaotongrenkk/math-tokenizer-gpt2-smoke-v1
kaggle kernels logs xiaotongrenkk/math-tokenizer-gpt2-smoke-v1
kaggle kernels push -p tmp/kaggle_downstream_gpt2/kernels/data
kaggle kernels status xiaotongrenkk/math-tokenizer-gpt2-data-v1
```

Submit the eight `kernels/train/*` directories only after both smoke and data
jobs finish successfully. When all eight training jobs finish, submit the eight
`kernels/eval/*` directories, then `kernels/aggregate`. The resumable driver
in `scripts/drive_kaggle_downstream.py` can monitor this sequence after the
initial smoke/data jobs are successful. Failed older Kaggle versions exist;
`kaggle kernels push` creates a new version of the same private kernel.

The Kaggle jobs install pinned Python libraries with Internet enabled. The
mathematical source code and GPT-2 weights are already in the private resource
bundle. Kaggle may unpack the tar/gzip files automatically; the run scripts
discover the mounted bundle by its manifest and verify the weight hash.

Source generator: https://github.com/facebookresearch/SymbolicMathematics at
`4596d070e1a9a1c2239c923d7d68fda577c8c007` (CC BY-NC 4.0). Its original
public datasets are optional and not needed by this experiment. The ODE data
uses the explicitly documented linear-in-constant subfamily of its method.
"""
    (DEST/"README.md").write_text(readme)
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as packed:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                packed.write(path,arcname=str(Path(DEST.name)/path.relative_to(DEST)))
    print(ARCHIVE,ARCHIVE.stat().st_size,sha(ARCHIVE))


if __name__=="__main__":
    build()
