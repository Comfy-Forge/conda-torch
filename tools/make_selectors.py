#!/usr/bin/env python3
"""Build the per-flavour selector metapackages.

One noarch-generic package per CUDA flavour in the grid — pytorch-cuda128
depends on `pytorch * cuda128_*` — so a manifest can pin a CUDA minor with
a single name instead of per-cell build-string globs. The build glob
matches every convention in the channel (cudaNNN_repack_*, cudaNNN_mkl_*,
cudaNNN_generic_*). Without a selector, conda semantics let the solver
pick any flavour whose major matches the host (build-number tiebreak
decides), which violates the channel's minor-match doctrine.

Usage: make_selectors.py [--out-dir DIR]   # writes <name>-<ver>-sel_0.conda
"""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

FLAVOURS = ["cu124", "cu126", "cu128", "cu129", "cu130", "cu132"]


def tar_zst(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for arc, b in members.items():
            ti = tarfile.TarInfo(arc)
            ti.size = len(b)
            ti.mtime = 0
            tf.addfile(ti, io.BytesIO(b))
    return subprocess.run(["zstd", "-19", "--stdout"], input=buf.getvalue(),
                          capture_output=True, check=True).stdout


def build_selector(flavour: str, out_dir: Path) -> Path:
    minor = flavour[2:3] + flavour[3:4] + "." + flavour[4:]  # cu124 -> 12.4, cu132 -> 13.2
    name = f"pytorch-cuda{flavour[2:]}"
    fn_base = f"{name}-{minor}-sel_0"
    index = {
        "name": name, "version": minor, "build": "sel_0", "build_number": 0,
        "subdir": "noarch", "noarch": "generic", "platform": None, "arch": None,
        "depends": [f"pytorch * cuda{flavour[2:]}_*"],
        "license": "MIT", "timestamp": int(time.time() * 1000),
    }
    about = {"summary": f"Selector metapackage: pins pytorch to CUDA {minor} "
                        f"(cuda{flavour[2:]}_*) builds of this channel"}
    paths = {"paths": [], "paths_version": 1}
    out = out_dir / f"{fn_base}.conda"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("metadata.json", json.dumps({"conda_pkg_format_version": 2}))
        zf.writestr(f"pkg-{fn_base}.tar.zst", tar_zst({}))
        zf.writestr(f"info-{fn_base}.tar.zst", tar_zst({
            "info/index.json": json.dumps(index).encode(),
            "info/paths.json": json.dumps(paths).encode(),
            "info/about.json": json.dumps(about).encode(),
        }))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("selectors-out"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fl in FLAVOURS:
        p = build_selector(fl, args.out_dir)
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{p} {p.stat().st_size}B sha256={sha[:16]}…")


if __name__ == "__main__":
    main()
