"""Resolve relocated current models and honestly classify retired evidence."""
import hashlib
import json
from pathlib import Path
from .corpus import ROOT


def retirement_entry(path):
    try:
        relative = str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return None
    manifest = ROOT / "models/retirement_manifest.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text())["files"].get(relative)


def resolve_checkpoint(path):
    path = Path(path)
    if path.exists():
        return path
    entry = retirement_entry(path)
    if entry and entry["relocated_to"]:
        target = ROOT / entry["relocated_to"]
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Relocated historical checkpoint changed: " + str(target))
        return target
    raise FileNotFoundError(f"Checkpoint retired or missing: {path}. Historical model replay requires the pre-cleanup Git revision; current models are in models/current/registry.json.")


def verify_saved_hash(path, expected):
    """Return actual/relocated/retired; a retired hash is provenance, not replay."""
    path = Path(path)
    if path.exists():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
        return "actual"
    entry = retirement_entry(path)
    assert entry is not None and entry["sha256"] == expected, (path, "unrecorded missing file")
    if entry["relocated_to"]:
        resolve_checkpoint(path)
        return "relocated"
    return "retired"
