#!/usr/bin/env python3
"""Per-entry solve sweep against the live channel.

For every non-hole grid entry, pick the newest python whose pytorch fragment
exists (skipping 3.15, which conda-forge cannot yet solve), prefer the
HIGHEST build number of that record, write a minimal pixi workspace pinning
that exact (version, build) on that platform with the flavour's CUDA
declared, run `pixi lock`, and assert the lockfile contains the expected
.conda filename served from OUR release URL.

Usage: sweep_solve.py [--grid grid/grid.json] [--work DIR] [--only SUBSTR]
Exit: 0 all OK, 1 otherwise. Prints one line per entry and a summary.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIXI = os.environ.get("PIXI", os.path.expanduser("~/.pixi/bin/pixi"))
CHANNEL = "https://comfy-forge.github.io/conda-torch"
OURS = "https://github.com/Comfy-Forge/conda-torch/releases/download/"


def newest(paths: list[str]) -> str:
    def bn(p: str) -> int:
        m = re.search(r"_(\d+)\.conda\.json$", p)
        return int(m.group(1)) if m else -1
    return max(paths, key=bn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path, default=REPO / "grid/grid.json")
    ap.add_argument("--work", type=Path, default=Path("/tmp/conda-torch-sweep"))
    ap.add_argument("--only", default="", help="substring filter on 'version/flavour/platform'")
    args = ap.parse_args()
    os.environ.setdefault("PIXI_CACHE_DIR", str(args.work / "cache"))
    g = json.loads(args.grid.read_text())

    results = []
    for ent in g["entries"]:
        if ent.get("action") == "hole":
            continue
        ver, fl, plat = ent["torch_version"], ent["flavour"], ent["platform"]
        key = f"{ver}/{fl}/{plat}"
        if args.only and args.only not in key:
            continue
        cuda = fl[2:4] + "." + fl[4:]
        pick = None
        for r in sorted(ent.get("records", []), key=lambda r: int(r["python"]), reverse=True):
            if r["python"] == "315":
                continue
            if r.get("action") == "mirror" and r.get("pytorch_file"):
                if (REPO / f"meta/{plat}/{r['pytorch_file']}.json").exists():
                    pick = r["pytorch_file"]
                    break
            elif r.get("action") == "repack":
                hits = glob.glob(str(REPO / f"meta/{plat}/pytorch-{ver}-cuda{fl[2:]}_repack_py{r['python']}_*.conda.json"))
                if hits:
                    pick = Path(newest(hits)).name[: -len(".json")]
                    break
        if pick is None:
            results.append((key, "SKIP", "no published record"))
            print(*results[-1], sep="\t", flush=True)
            continue
        build = re.match(rf"pytorch-{re.escape(ver)}-(.+)\.conda$", pick).group(1)
        proj = args.work / "proj"
        shutil.rmtree(proj, ignore_errors=True)
        proj.mkdir(parents=True)
        (proj / "pixi.toml").write_text(
            f'[workspace]\nname = "sweep"\nchannels = ["{CHANNEL}", "conda-forge"]\n'
            f'platforms = [{{ platform = "{plat}", cuda = "{cuda}" }}]\n\n'
            f'[dependencies]\npytorch = {{ version = "=={ver}", build = "{build}" }}\n')
        p = subprocess.run([PIXI, "lock"], cwd=proj, capture_output=True, text=True, timeout=900)
        if p.returncode != 0:
            tail = (p.stderr or p.stdout).strip().splitlines()[-1][:160] if (p.stderr or p.stdout).strip() else "?"
            results.append((key, "UNSAT", tail))
        else:
            lock = (proj / "pixi.lock").read_text()
            if f"{OURS}{plat}/{pick}" in lock:
                results.append((key, "OK", pick))
            elif pick in lock:
                results.append((key, "WRONG-SOURCE", pick))
            else:
                results.append((key, "MISSING-IN-LOCK", pick))
        print(*results[-1], sep="\t", flush=True)

    ok = sum(1 for r in results if r[1] == "OK")
    print(f"\n=== SWEEP: {ok}/{len(results)} OK ===")
    for r in results:
        if r[1] != "OK":
            print(*r, sep="\t")
    (args.work / "results.json").write_text(json.dumps(results, indent=1))
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
