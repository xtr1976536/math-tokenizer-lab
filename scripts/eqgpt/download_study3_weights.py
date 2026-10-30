"""Download and verify the de-duplicated Study 3 checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments/study3_corpus_effect_v1"
MANIFEST = RUN / "weight_manifest.json"
VERSION = "study3-corpus-effect-v1-weights"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--cohort", choices=("small", "large", "both"), default="both")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    rows = [row for row in data["files"] if args.cohort == "both" or row["cohort"] == args.cohort]
    missing = []
    for row in rows:
        target = ROOT / row["path"]
        if target.exists() and target.stat().st_size == row["size_bytes"] and sha(target) == row["sha256"]:
            continue
        if args.verify_only:
            missing.append(row["path"])
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        url = ("https://github.com/xtr1976536/math-tokenizer-lab/releases/download/"
               f"{VERSION}/{quote(row['asset'])}")
        request = urllib.request.Request(url, headers={"User-Agent": "math-tokenizer-lab"})
        with urllib.request.urlopen(request, timeout=600) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if temporary.stat().st_size != row["size_bytes"] or sha(temporary) != row["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Release checkpoint hash/size mismatch: {row['asset']}")
        temporary.replace(target)
        print("Downloaded and verified", row["path"])
    if missing:
        raise SystemExit("Missing or invalid checkpoints:\n" + "\n".join(missing))
    print(f"Verified {len(rows)} Study 3 checkpoints ({sum(row['size_bytes'] for row in rows)} bytes).")


if __name__ == "__main__":
    main()
