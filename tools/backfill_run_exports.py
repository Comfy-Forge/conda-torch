#!/usr/bin/env python3
"""One-time: add `run_exports` to fragments published before it was captured.

fragment.py now lifts info/run_exports.json out of every artifact, but the
1,832 fragments published before that carry nothing, and re-downloading
~200 GiB of artifacts to read a two-line JSON is absurd. Both sources of
truth are available without touching the artifacts:

  * our own repacks — deterministic. torch_repack.py emits, for version V
    with next minor N:
        libtorch  {"weak": ["libtorch >=V,<N.0a0"]}
        pytorch   {"weak": ["pytorch >=V,<N.0a0", "libtorch >=V,<N.0a0"]}
    and passes run_exports=None for every side artifact (triton, cudnn,
    nvshmem) and the selector metapackages. Verified against the published
    bytes of libtorch 2.8.0 _4, pytorch 2.8.0 _4 and libtorch 2.4.1 _1.

  * mirrors — they are conda-forge's own bytes, so conda-forge's published
    run_exports.json is authoritative for them, keyed by exact filename.

Anything this script cannot source is left absent rather than guessed;
`--check` reports what that is without writing.

Usage: backfill_run_exports.py [--meta-dir meta] [--check]
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

CF = "https://conda.anaconda.org/conda-forge"
# names whose artifacts our pipeline builds with run_exports=None
NO_RUN_EXPORTS = {"triton", "libcudnn", "nvidia-cudnn", "nvidia-nvshmem",
                  "cuda-bindings", "comfy-channel-smoke"}


def next_minor(version: str) -> str:
    parts = version.split(".")
    return f"{parts[0]}.{int(parts[1]) + 1}"


def ours(name: str, version: str) -> dict:
    """What torch_repack.py emitted for this artifact, or {} if nothing."""
    if name == "libtorch":
        return {"weak": [f"libtorch >={version},<{next_minor(version)}.0a0"]}
    if name == "pytorch":
        n = next_minor(version)
        return {"weak": [f"pytorch >={version},<{n}.0a0",
                         f"libtorch >={version},<{n}.0a0"]}
    if name in NO_RUN_EXPORTS or name.startswith("pytorch-cuda"):
        return {}
    return {}


def cf_run_exports(subdir: str, cache: Path) -> dict:
    """conda-forge's run_exports.json for a subdir, cached on disk."""
    local = cache / f"cf-run_exports-{subdir}.json"
    if not local.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        url = f"{CF}/{subdir}/run_exports.json"
        print(f"  fetching {url} ...", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "conda-torch/backfill"})
        local.write_bytes(urllib.request.urlopen(req, timeout=600).read())
    return json.loads(local.read_text())["packages.conda"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, default=Path("meta"))
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/conda-torch-backfill"))
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    unsourced, written, already = [], 0, 0
    for sub_dir in sorted(p for p in args.meta_dir.iterdir() if p.is_dir()):
        subdir = sub_dir.name
        frags = sorted(sub_dir.glob("*.json"))
        if not frags:
            continue
        need_cf = any(json.loads(f.read_text()).get("mirrored_from") for f in frags)
        cf = cf_run_exports(subdir, args.cache_dir) if need_cf else {}

        for frag in frags:
            entry = json.loads(frag.read_text())
            if "run_exports" in entry:
                already += 1
                continue
            filename = frag.name[: -len(".json")]
            if entry.get("mirrored_from"):
                hit = cf.get(filename)
                if hit is None:
                    unsourced.append(f"{subdir}/{filename} (mirror absent from conda-forge)")
                    continue
                rex = hit.get("run_exports") or {}
            else:
                rex = ours(entry["name"], entry["version"])
            if not rex:
                continue  # nothing to record; absent means empty downstream
            if not args.check:
                entry["run_exports"] = rex
                frag.write_text(json.dumps(entry, indent=1, sort_keys=True) + "\n")
            written += 1
        print(f"{subdir}: {len(frags)} fragments")

    verb = "would write" if args.check else "wrote"
    print(f"\n{verb} run_exports into {written} fragments"
          f" ({already} already had it, {len(unsourced)} unsourced)")
    for u in unsourced[:20]:
        print(f"  unsourced: {u}")
    return 1 if unsourced else 0


if __name__ == "__main__":
    sys.exit(main())
