#!/usr/bin/env python3
"""Scan a pixi.lock (or conda-lock/explicit) file for known-defective builds.

Published artifacts in this channel are immutable, and metadata patches in
patches/ only protect FUTURE solves — a lockfile that already pins a bad
build keeps installing it silently. This is the self-serve check: run it
against your own lockfile, re-lock anything it flags.

Usage: check_lock.py [pixi.lock ...]        (default: ./pixi.lock)
Exit status: 0 clean, 1 defective build(s) found, 2 usage error.

Matches on the exact .conda filename anywhere in the file, so it works on
pixi.lock, conda-lock.yml, and @EXPLICIT env files alike, with no parser
to keep in sync with lockfile format versions.
"""

import json
import sys
import urllib.request
from pathlib import Path

KNOWN_BAD_URL = "https://raw.githubusercontent.com/Comfy-Forge/conda-torch/main/known_bad.json"


def load_known_bad() -> dict:
    local = Path(__file__).resolve().parent.parent / "known_bad.json"
    if local.is_file():
        return json.loads(local.read_text())
    with urllib.request.urlopen(KNOWN_BAD_URL, timeout=30) as r:
        return json.load(r)


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or [Path("pixi.lock")]
    for p in paths:
        if not p.is_file():
            print(f"error: {p} not found", file=sys.stderr)
            return 2
    bad = load_known_bad()
    # the same build string can be defective under several subdirs (linux-64
    # and linux-aarch64 share naming), so keep every (subdir, info) per name
    index: dict[str, list[tuple[str, dict]]] = {}
    for subdir, entries in bad.items():
        if subdir.startswith("_"):
            continue
        for fn, info in entries.items():
            index.setdefault(fn, []).append((subdir, info))

    hits = 0
    for p in paths:
        text = p.read_text(errors="replace")
        for fn, variants in index.items():
            if fn not in text:
                continue
            # prefer the subdir whose full asset URL actually appears; fall
            # back to every variant for formats that carry bare filenames
            matched = [(s, i) for s, i in variants if f"/{s}/{fn}" in text] or variants
            for subdir, info in matched:
                hits += 1
                print(f"{p}: DEFECTIVE {subdir}/{fn}")
                print(f"    reason:   {info['reason']}")
                print(f"    fixed by: {info['fixed_by']}")
                if info.get("patched"):
                    print(f"    note:     {info['patched']} — re-lock to pick it up")
    if hits:
        print(f"\n{hits} known-defective build(s) pinned. Re-lock (pixi update / delete the "
              f"pin and re-solve) onto the replacement(s) above.")
        return 1
    print("clean: no known-defective builds pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
