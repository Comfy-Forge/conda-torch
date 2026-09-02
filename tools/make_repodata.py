#!/usr/bin/env python3
"""Assemble the static channel site from committed repodata fragments.

Reads meta/<subdir>/*.json (one fragment per .conda, produced by
fragment.py) and writes site/<subdir>/repodata.json with CEP-15
info.base_url pointing at the matching GitHub release, so packages are
fetched from release assets while repodata is served by GitHub Pages.

Metadata overlay: patches/<subdir>/patches.json may override selected
repodata keys per exact .conda filename. Fragments on disk stay untouched
(published artifacts are immutable); the overlay is the metadata-fix path.
A patch naming a filename with no fragment is a hard error, so a typo
cannot silently no-op.

Also emits repodata.json.zst (zstd -19) beside each repodata.json.

Usage: make_repodata.py [--meta-dir meta] [--site-dir site] [--patches-dir patches]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

RELEASES = "https://github.com/Comfy-Forge/conda-torch/releases/download"
CHANNEL = "https://comfy-forge.github.io/conda-torch"
# every subdir a client might request must exist (404s abort some solvers)
ALWAYS_SUBDIRS = {"noarch", "linux-64", "linux-aarch64", "win-64", "osx-arm64", "osx-64"}
# keys a patch may override; anything else in a patch entry is a hard error
PATCHABLE_KEYS = {"depends", "constrains", "purls"}


def load_patches(patches_dir: Path, subdir: str, packages_conda: dict) -> int:
    pfile = patches_dir / subdir / "patches.json"
    if not pfile.is_file():
        return 0
    patches = json.loads(pfile.read_text())
    applied = 0
    for filename, override in patches.items():
        if filename not in packages_conda:
            sys.exit(f"{pfile}: patch targets {filename!r} but no such fragment exists "
                     f"in meta/{subdir}/ — fix the filename or drop the patch")
        bad = set(override) - PATCHABLE_KEYS
        if bad:
            sys.exit(f"{pfile}: {filename}: keys {sorted(bad)} are not patchable "
                     f"(allowed: {sorted(PATCHABLE_KEYS)})")
        packages_conda[filename].update(override)
        applied += 1
    return applied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, default=Path("meta"))
    ap.add_argument("--site-dir", type=Path, default=Path("site"))
    ap.add_argument("--patches-dir", type=Path, default=Path("patches"))
    args = ap.parse_args()

    subdirs = {p.name for p in args.meta_dir.iterdir() if p.is_dir()} | ALWAYS_SUBDIRS
    summary = []
    for subdir in sorted(subdirs):
        packages_conda = {}
        for frag in sorted((args.meta_dir / subdir).glob("*.json")) if (args.meta_dir / subdir).is_dir() else []:
            filename = frag.name[: -len(".json")]
            packages_conda[filename] = json.loads(frag.read_text())
        patched = load_patches(args.patches_dir, subdir, packages_conda)
        repodata = {
            "info": {"subdir": subdir, "base_url": f"{RELEASES}/{subdir}/"},
            "packages": {},
            "packages.conda": packages_conda,
            "repodata_version": 2,
        }
        out = args.site_dir / subdir / "repodata.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(repodata, indent=1, sort_keys=True) + "\n"
        out.write_text(body)
        zst = subprocess.run(["zstd", "-19", "--stdout"], input=body.encode(),
                             capture_output=True, check=True).stdout
        (args.site_dir / subdir / "repodata.json.zst").write_bytes(zst)
        summary.append((subdir, len(packages_conda), patched))

    lines = "".join(
        f"<tr><td>{s}</td><td>{n}</td><td><a href='{s}/repodata.json'>repodata.json</a></td></tr>"
        for s, n, _ in summary
    )
    (args.site_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>comfy-forge conda channel</title>"
        "<style>body{font-family:system-ui;margin:3em auto;max-width:40em}"
        "td{padding:.3em 1em;border-bottom:1px solid #ccc}</style>"
        f"<h1>comfy-forge conda channel</h1><p>Add <code>{CHANNEL}</code> as a conda channel. "
        "Packages are served as GitHub release assets via CEP-15 <code>base_url</code>.</p>"
        f"<table><tr><th>subdir</th><th>packages</th><th></th></tr>{lines}</table>"
    )
    for s, n, p in summary:
        print(f"{s}: {n} packages" + (f" ({p} patched)" if p else ""))


if __name__ == "__main__":
    main()
