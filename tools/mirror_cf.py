#!/usr/bin/env python3
"""Mirror conda-forge artifacts byte-identical into this channel.

For grid cells conda-forge already built at the exact PyPI CUDA flavour we
re-host their artifact unchanged: download from conda.anaconda.org, verify
sha256 AND size against conda-forge's own repodata entry, upload to our
release (tag = subdir), and commit the conda-forge repodata entry verbatim
as our fragment (plus provenance). Dependencies are left untouched — this
channel is always used alongside conda-forge, so their deps resolve there.

Published artifacts are immutable: an existing fragment with a different
sha256 is a hard error, never an overwrite; a matching one is a no-op.

Usage:
  mirror_cf.py --subdir linux-64 <filename.conda> [...]
  mirror_cf.py --manifest mirror.txt        # lines: "<subdir> <filename>"

Needs: zstd CLI, gh CLI (authenticated), network to conda.anaconda.org.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

CF = "https://conda.anaconda.org/conda-forge"
REPO = "Comfy-Forge/conda-torch"
UA = {"User-Agent": "conda-torch-mirror/1 (github.com/Comfy-Forge/conda-torch)"}


def fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def repodata(subdir: str, cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    js = cache / f"{subdir}.repodata.json"
    if not js.exists():
        zst = cache / f"{subdir}.repodata.json.zst"
        print(f"fetching conda-forge {subdir} repodata ...", file=sys.stderr)
        fetch(f"{CF}/{subdir}/repodata.json.zst", zst)
        subprocess.run(["zstd", "-d", "-f", str(zst), "-o", str(js)], check=True,
                       capture_output=True)
    return json.loads(js.read_text())


def file_hashes(path: Path) -> tuple[str, str, int]:
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), size


def gh_json(args: list[str]):
    r = subprocess.run(["gh"] + args, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout) if r.stdout.strip() else {}


def release_assets(subdir: str) -> dict[str, int]:
    """name -> size for assets on the subdir release; creates the release if absent."""
    data = gh_json(["release", "view", subdir, "-R", REPO, "--json", "assets"])
    if data is None:
        subprocess.run(
            ["gh", "release", "create", subdir, "-R", REPO, "--title", f"{subdir} packages",
             "--notes", "Storage release for this subdir. Do not delete: repodata base_url points here."],
            check=True, capture_output=True)
        return {}
    return {a["name"]: a["size"] for a in data.get("assets", [])}


def mirror_one(subdir: str, filename: str, rd: dict, assets: dict[str, int],
               meta_dir: Path, cache: Path) -> str:
    key = "packages.conda" if filename.endswith(".conda") else "packages"
    entry = rd.get(key, {}).get(filename)
    if entry is None:
        sys.exit(f"{subdir}/{filename}: not in conda-forge repodata — check the exact filename")
    want_sha, want_size = entry["sha256"], entry["size"]

    frag_path = meta_dir / subdir / f"{filename}.json"
    if frag_path.exists():
        have = json.loads(frag_path.read_text())
        if have.get("sha256") == want_sha:
            return "fragment already present (same sha256) — no-op"
        sys.exit(f"{frag_path}: exists with DIFFERENT sha256 "
                 f"({have.get('sha256')} vs {want_sha}); published artifacts are immutable")

    if filename in assets:
        if assets[filename] != want_size:
            sys.exit(f"{subdir}/{filename}: already on our release with size {assets[filename]} "
                     f"but conda-forge says {want_size} — refusing to write a fragment for it")
        status = "asset already uploaded (size matches)"
    else:
        local = cache / subdir / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            print(f"downloading {filename} ({want_size/1e6:.0f} MB) ...", file=sys.stderr)
            fetch(f"{CF}/{subdir}/{filename}", local)
        sha, md5, size = file_hashes(local)
        if sha != want_sha or size != want_size:
            local.rename(local.with_suffix(local.suffix + ".corrupt"))
            sys.exit(f"{subdir}/{filename}: hash/size mismatch vs conda-forge repodata "
                     f"(got sha256={sha} size={size}, want {want_sha} {want_size})")
        if entry.get("md5") and md5 != entry["md5"]:
            sys.exit(f"{subdir}/{filename}: md5 mismatch")
        subprocess.run(["gh", "release", "upload", subdir, str(local), "-R", REPO],
                       check=True, capture_output=True)
        assets[filename] = size
        status = "downloaded, verified, uploaded"

    frag = dict(entry)
    frag["subdir"] = subdir
    frag["mirrored_from"] = "conda-forge"
    frag_path.parent.mkdir(parents=True, exist_ok=True)
    frag_path.write_text(json.dumps(frag, indent=1, sort_keys=True) + "\n")
    return f"{status}; fragment written"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("filenames", nargs="*")
    ap.add_argument("--subdir")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--meta-dir", type=Path, default=Path("meta"))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(os.environ.get("MIRROR_CACHE", "/tmp/conda-torch-mirror-cache")))
    args = ap.parse_args()

    jobs: list[tuple[str, str]] = []
    if args.manifest:
        for line in args.manifest.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                subdir, fn = line.split()
                jobs.append((subdir, fn))
    for fn in args.filenames:
        if not args.subdir:
            sys.exit("--subdir is required with positional filenames")
        jobs.append((args.subdir, fn))
    if not jobs:
        sys.exit("nothing to do")

    rds: dict[str, dict] = {}
    assets: dict[str, dict[str, int]] = {}
    for subdir, fn in jobs:
        if subdir not in rds:
            rds[subdir] = repodata(subdir, args.cache_dir)
            assets[subdir] = release_assets(subdir)
        result = mirror_one(subdir, fn, rds[subdir], assets[subdir], args.meta_dir, args.cache_dir)
        print(f"{subdir}/{fn}: {result}")


if __name__ == "__main__":
    main()
