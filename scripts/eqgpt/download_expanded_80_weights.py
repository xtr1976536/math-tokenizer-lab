"""Download and checksum all checkpoints for the expanded 80-token run."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments/expanded_downstream_80_v1"
MANIFEST = RUN / "weight_manifest.json"
VERSION = "expanded-downstream-80-v1-weights"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    missing = []
    for row in data["files"]:
        target = RUN / row["path"]
        if target.exists() and target.stat().st_size == row["size_bytes"] and sha(target) == row["sha256"]:
            continue
        if args.verify_only:
            missing.append(row["path"])
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        url = ("https://github.com/xtr1976536/math-tokenizer-lab/releases/download/"
               f"{VERSION}/{quote(row['asset'])}")
        request = urllib.request.Request(url, headers={"User-Agent": "math-tokenizer-lab"})
        with urllib.request.urlopen(request, timeout=180) as response, tmp.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if tmp.stat().st_size != row["size_bytes"] or sha(tmp) != row["sha256"]:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"Release checkpoint hash/size mismatch: {row['asset']}")
        tmp.replace(target)
        print("Downloaded and verified", row["path"])
    if missing:
        raise SystemExit("Missing or invalid checkpoints:\n" + "\n".join(missing))
    print(f"All {len(data['files'])} checkpoints verified ({data['total_bytes']} bytes).")


if __name__ == "__main__":
    main()
