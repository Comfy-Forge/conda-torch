#!/usr/bin/env python3
"""Assemble the static channel site from committed repodata fragments.

Reads meta/<subdir>/*.json (one fragment per .conda, produced by
fragment.py) and writes site/<subdir>/repodata.json with CEP-15
info.base_url pointing at the matching GitHub release, so packages are
fetched from release assets while repodata is served by GitHub Pages.

Usage: make_repodata.py [--meta-dir meta] [--site-dir site]
"""

import argparse
import json
from pathlib import Path

RELEASES = "https://github.com/Comfy-Forge/conda-torch/releases/download"
CHANNEL = "https://comfy-forge.github.io/conda-torch"
# every subdir a client might request must exist (404s abort some solvers)
ALWAYS_SUBDIRS = {"noarch", "linux-64", "linux-aarch64", "win-64", "osx-arm64", "osx-64"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, default=Path("meta"))
    ap.add_argument("--site-dir", type=Path, default=Path("site"))
    args = ap.parse_args()

    subdirs = {p.name for p in args.meta_dir.iterdir() if p.is_dir()} | ALWAYS_SUBDIRS
    summary = []
    for subdir in sorted(subdirs):
        packages_conda = {}
        for frag in sorted((args.meta_dir / subdir).glob("*.json")) if (args.meta_dir / subdir).is_dir() else []:
            filename = frag.name[: -len(".json")]
            packages_conda[filename] = json.loads(frag.read_text())
        repodata = {
            "info": {"subdir": subdir, "base_url": f"{RELEASES}/{subdir}/"},
            "packages": {},
            "packages.conda": packages_conda,
            "repodata_version": 2,
        }
        out = args.site_dir / subdir / "repodata.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(repodata, indent=1, sort_keys=True) + "\n")
        summary.append((subdir, len(packages_conda)))

    lines = "".join(
        f"<tr><td>{s}</td><td>{n}</td><td><a href='{s}/repodata.json'>repodata.json</a></td></tr>"
        for s, n in summary
    )
    (args.site_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>comfy-forge conda channel</title>"
        "<style>body{font-family:system-ui;margin:3em auto;max-width:40em}"
        "td{padding:.3em 1em;border-bottom:1px solid #ccc}</style>"
        f"<h1>comfy-forge conda channel</h1><p>Add <code>{CHANNEL}</code> as a conda channel. "
        "Packages are served as GitHub release assets via CEP-15 <code>base_url</code>.</p>"
        f"<table><tr><th>subdir</th><th>packages</th><th></th></tr>{lines}</table>"
    )
    for s, n in summary:
        print(f"{s}: {n} packages")


if __name__ == "__main__":
    main()
