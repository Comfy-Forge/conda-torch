#!/usr/bin/env python3
"""Index a directory of .conda files as a minimal conda channel.

Used by the win-64 clean-environment gate: the artifacts a repack run just
built are not published yet (publishing is what the gate protects), so the
solver is handed them through a file:// channel that carries exactly those
files. Layout written:

  <out>/<subdir>/<file>.conda        (copied)
  <out>/<subdir>/repodata.json       (index.json + sha256/md5/size per file)
  <out>/noarch/repodata.json         (empty; solvers request it unconditionally)

Pure python (zstandard module), no zstd CLI: it runs on a stock Windows runner.

Usage: local_channel.py --subdir win-64 --out <dir> <pkg.conda> [...]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import zstandard


def index_json(conda_path: Path) -> dict:
    with zipfile.ZipFile(conda_path) as zf:
        info = [n for n in zf.namelist() if n.startswith("info-") and n.endswith(".tar.zst")]
        if len(info) != 1:
            raise SystemExit(f"{conda_path.name}: expected one info-*.tar.zst, found {info}")
        raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(zf.read(info[0]))).read()
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        member = tf.extractfile("info/index.json")
        if member is None:
            raise SystemExit(f"{conda_path.name}: info/index.json missing")
        return json.load(member)


def hashes(path: Path) -> tuple[str, str, int]:
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()

    sub = args.out / args.subdir
    sub.mkdir(parents=True, exist_ok=True)
    packages = {}
    for f in args.files:
        entry = index_json(f)
        if entry.get("subdir", args.subdir) != args.subdir:
            raise SystemExit(f"{f.name}: subdir {entry.get('subdir')!r} != {args.subdir!r}")
        sha, md5, size = hashes(f)
        entry.update({"sha256": sha, "md5": md5, "size": size, "subdir": args.subdir})
        shutil.copyfile(f, sub / f.name)
        packages[f.name] = entry
        print(f"indexed {f.name}: {entry['name']} {entry['version']} {entry['build']}")
    (sub / "repodata.json").write_text(json.dumps({
        "info": {"subdir": args.subdir}, "packages": {},
        "packages.conda": packages, "repodata_version": 1}, indent=1, sort_keys=True))
    noarch = args.out / "noarch"
    noarch.mkdir(exist_ok=True)
    (noarch / "repodata.json").write_text(json.dumps({
        "info": {"subdir": "noarch"}, "packages": {}, "packages.conda": {},
        "repodata_version": 1}))
    print(f"channel -> {args.out.resolve().as_uri()} ({len(packages)} packages)")


if __name__ == "__main__":
    main()
