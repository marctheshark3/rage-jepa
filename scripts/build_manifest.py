#!/usr/bin/env python3
"""Freeze the v0.2 source/evidence inventory and the tested release wheel."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "release-v0.2-manifest.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                           cwd=ROOT, check=True, capture_output=True).stdout.decode().split("\0")
    files = []
    for name in sorted(set(paths) - {"", MANIFEST}):
        path = ROOT / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"invalid release path: {name}")
        files.append({"path": name, "sha256": sha(path), "bytes": path.stat().st_size})
    wheel = ROOT / "dist/rage_jepa-0.2.0-py3-none-any.whl"
    manifest = {"schema": "rage-jepa.public-runtime-release.v1", "version": "0.2.0",
                "historical_release": "v0.1.0", "historical_manifest_sha256": sha(ROOT / "release-manifest.json"),
                "cohort_definition_sha256": sha(ROOT / "evidence/runtime-v0.2/cohort/definition.json"),
                "license": "MIT: code, model weights, authored simulation data and documentation",
                "tested_wheel": {"name": wheel.name, "sha256": sha(wheel), "bytes": wheel.stat().st_size},
                "files": files}
    (ROOT / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(files), "bytes": sum(f["bytes"] for f in files), "wheel": manifest["tested_wheel"]}, indent=2))


if __name__ == "__main__":
    main()
