"""User-authorized model cleanup; history/results/corpus remain auditable.

Only results/<run>/models files are removed. Selected AST-v4 and v5 checkpoints
are copied byte-for-byte to models/current before removing their old locations.
Git retains the previous tracked snapshots; this does not rewrite Git history.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
KEEP = {
    "results/v4_512/models/ast-atom256.json": "models/current/ast_v4/base-512.json",
    "results/v4_1024/models/ast-atom256.json": "models/current/ast_v4/base-1024.json",
}
for scenario in ("broad", "specialized"):
    for suffix in (".json", ".tokenizer.json"):
        KEEP[f"results/v5_1024/models/{scenario}-rl-selected-17{suffix}"] = f"models/current/v5/{scenario}-base{suffix}"
        KEEP[f"results/v5_1024/models/{scenario}-reward_greedy-17{suffix}"] = f"models/current/v5/controls/{scenario}-greedy{suffix}"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    registry = ROOT / "models/retirement_manifest.json"
    if registry.exists():
        raise RuntimeError("Cleanup already recorded; never rerun automatically on new experiments.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for original, destination in KEEP.items():
        source = ROOT / original
        assert source.is_file(), source
        target = ROOT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
    directories = sorted((ROOT / "results").glob("*/models"))
    entries = {}
    for directory in directories:
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(ROOT))
            entries[relative] = {"sha256": sha(path), "bytes": path.stat().st_size,
                                 "relocated_to": KEEP.get(relative)}
    payload = {"reason": "User requested only AST-v4 and v5 active model families",
               "recoverable_git_commit": commit, "files": entries,
               "removed_files": len(entries), "removed_bytes": sum(e["bytes"] for e in entries.values()),
               "preserved_checkpoints": KEEP,
               "scope": "checkpoint files only; historical metrics/protocols and all corpus files unchanged"}
    registry.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    active = {"families": ["ast_v4", "v5"], "base_commit": commit,
              "ast_v4": {str(cap): f"models/current/ast_v4/base-{cap}.json" for cap in (512, 1024)},
              "v5": {scenario: f"models/current/v5/{scenario}-base.json" for scenario in ("broad", "specialized")},
              "controls": {scenario: f"models/current/v5/controls/{scenario}-greedy.json" for scenario in ("broad", "specialized")}}
    (ROOT / "models/current/registry.json").write_text(json.dumps(active, indent=2) + "\n")
    # Mutations begin only after all retained checkpoints and the full manifest
    # are saved and validated. Never delete outside the exact inventoried paths.
    for relative, entry in entries.items():
        path = ROOT / relative
        assert sha(path) == entry["sha256"]
        path.unlink()
    for directory in directories:
        for path in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir():
                path.rmdir()
        directory.rmdir()
    print(json.dumps({"removed_files": len(entries), "removed_bytes": payload["removed_bytes"],
                      "retained_files": len(KEEP), "recovery_commit": commit}))


if __name__ == "__main__":
    main()
