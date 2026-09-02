#!/usr/bin/env python3
"""Emit a repodata fragment for one .conda file.

The channel's repodata.json is assembled from per-package JSON fragments
committed under meta/<subdir>/<filename>.json, so publishing never has to
re-download release assets. This tool produces one fragment: the package's
info/index.json plus the sha256/md5/size of the artifact itself.

Usage: fragment.py <pkg.conda> <subdir> [--meta-dir meta]

Requires the `zstd` CLI (the .conda info tarball is zstd-compressed and
python 3.12 has no stdlib zstd).
"""

from __future__ import annotations  # runs under the target python; may be 3.8

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def read_index_json(conda_path: Path) -> dict:
    with zipfile.ZipFile(conda_path) as zf:
        info_names = [n for n in zf.namelist() if n.startswith("info-") and n.endswith(".tar.zst")]
        if len(info_names) != 1:
            sys.exit(f"expected exactly one info-*.tar.zst in {conda_path.name}, found {info_names}")
        zst = zf.read(info_names[0])
    tar = subprocess.run(["zstd", "-d", "--stdout"], input=zst, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        member = tf.extractfile("info/index.json")
        if member is None:
            sys.exit(f"{conda_path.name}: info/index.json missing")
        return json.load(member)


def hashes(path: Path) -> tuple[str, str, int]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("conda_file", type=Path)
    ap.add_argument("subdir")
    ap.add_argument("--meta-dir", type=Path, default=Path("meta"))
    args = ap.parse_args()

    index = read_index_json(args.conda_file)
    if index.get("subdir", args.subdir) != args.subdir:
        sys.exit(f"index.json says subdir={index['subdir']!r} but you passed {args.subdir!r}")
    if "+" in str(index.get("version", "")):
        sys.exit(f"refusing local-version '+' in version {index['version']!r}: "
                 "conda orders 2.8.0+cu128 BELOW 2.8.0; encode flavour in the build string")

    sha256, md5, size = hashes(args.conda_file)
    entry = dict(index)
    entry.update({"sha256": sha256, "md5": md5, "size": size, "subdir": args.subdir})

    out = args.meta_dir / args.subdir / f"{args.conda_file.name}.json"
    if out.exists():
        have = json.loads(out.read_text()).get("sha256")
        if have == sha256:
            print(f"{out}: already present with same sha256 — no-op")
            return
        sys.exit(f"{out}: exists with DIFFERENT sha256 ({have} vs {sha256}); "
                 "published artifacts are immutable — bump the build number instead")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entry, indent=1, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
