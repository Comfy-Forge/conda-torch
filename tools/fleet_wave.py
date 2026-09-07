#!/usr/bin/env python3
"""Drive a fleet-wide rebuild wave through repack-torch.yml.

Every repack cell in the grid is bumped to its NEXT build number and rebuilt.
Published artifacts are immutable, so a wave never republishes a filename: it
always moves to max(published build number) + 1.

Phasing is donor-then-shim. A cell's libtorch half is shared by every shim of
that entry, so the donor is dispatched and allowed to finish first; the shims
then find libtorch already published and skip rebuilding it. Dispatching them
together would have every shim rebuild the same libtorch and throw it away.

The done check is fragment existence, not the workflow's own conclusion: a run
can go green having published nothing, and a run can die after uploading but
before committing its fragment. `meta/<subdir>/<stem>.conda.json` is the only
statement that a cell actually landed. The build-string hash is not knowable
before rendering, so the check globs `..._h*_<n>.conda.json`.

Every GitHub API call is fail-closed. An error is never read as "not done" --
that would re-dispatch work that may be in flight -- it is retried with
backoff and, if it keeps failing, aborts the wave.

Usage:
  fleet_wave.py --phase donor --concurrency 8
  fleet_wave.py --phase shim  --concurrency 8
  fleet_wave.py --phase all --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = "repack-torch.yml"


def log(msg: str) -> None:
    print(f"[fleet] {msg}", flush=True)


def gh(*args: str, retries: int = 5) -> str:
    """Run gh, retrying transient failures. Raises rather than returning a
    value the caller could mistake for 'nothing there'."""
    last = ""
    for attempt in range(retries):
        p = subprocess.run(["gh", *args], capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout
        last = (p.stderr or p.stdout).strip()
        wait = min(60, 2 ** attempt) + random.random() * 2
        log(f"gh {' '.join(args[:3])} failed ({last.splitlines()[-1][:120] if last else '?'}); "
            f"retry {attempt + 1}/{retries} in {wait:.1f}s")
        time.sleep(wait)
    raise RuntimeError(f"gh {' '.join(args)} failed after {retries} tries: {last}")


def cells(grid: dict) -> list[dict]:
    """Every repack cell, in a stable order."""
    out = []
    for e in grid["entries"]:
        if e["action"] == "hole":
            continue
        for r in e["records"]:
            if r["action"] != "repack":
                continue
            out.append({
                "version": e["torch_version"],
                "flavour": e["flavour"],
                "platform": e["platform"],
                "python": f"{r['python'][0]}.{r['python'][1:]}",
                "pytag": f"py{r['python']}",
                "donor": bool(r.get("libtorch_donor")),
            })
    return out


def published_max(subdir: str) -> dict[tuple, int]:
    """Highest published build number per (version, flavour, pytag), read from
    the channel's own repodata."""
    import urllib.request
    url = f"https://comfy-forge.github.io/conda-torch/{subdir}/repodata.json"
    with urllib.request.urlopen(url, timeout=120) as fh:
        data = json.load(fh)
    best: dict[tuple, int] = {}
    for fn in data.get("packages.conda", {}):
        if not fn.startswith("pytorch-"):
            continue
        stem = fn[:-len(".conda")]
        try:
            _, version, build = stem.split("-", 2)
            flavour, rest = build.split("_repack_", 1)
            pytag, _, num = rest.split("_")
        except ValueError:
            continue
        key = (version, flavour, pytag)
        best[key] = max(best.get(key, -1), int(num))
    return best


def fragment_exists(cell: dict, num: int) -> bool:
    """The only trustworthy done check: the cell's repodata fragment is in the
    tree. The build-string hash is decided during rendering, so glob it."""
    d = REPO / "meta" / cell["platform"]
    pat = (f"pytorch-{cell['version']}-{cell['flavour_build']}_repack_"
           f"{cell['pytag']}_h*_{num}.conda.json")
    return any(d.glob(pat))


def in_flight() -> int:
    out = gh("run", "list", f"--workflow={WORKFLOW}", "--limit", "100",
             "--json", "status", "-q",
             '[.[] | select(.status == "in_progress" or .status == "queued")] | length')
    return int(out.strip() or 0)


def dispatch(cell: dict, num: int) -> None:
    gh("workflow", "run", WORKFLOW, "--ref", "main",
       "-f", f"version={cell['version']}",
       "-f", f"flavour={cell['flavour']}",
       "-f", f"python={cell['python']}",
       "-f", f"platform={cell['platform']}",
       "-f", f"build_number={num}",
       "-f", "side_build_number=0")


def sync() -> None:
    subprocess.run(["git", "pull", "--rebase", "--quiet", "origin", "main"],
                   cwd=REPO, check=False)


def run_phase(todo: list[tuple[dict, int]], concurrency: int,
              dry_run: bool) -> list[tuple[dict, int]]:
    """Dispatch every cell, respecting the concurrency gate. Returns the ones
    still missing a fragment when the phase drains."""
    pending = list(todo)
    while pending:
        while in_flight() >= concurrency:
            time.sleep(30)
        cell, num = pending.pop(0)
        label = (f"{cell['version']} {cell['flavour']} {cell['python']} "
                 f"{cell['platform']} -> _{num}")
        if dry_run:
            log(f"WOULD dispatch {label}")
            continue
        dispatch(cell, num)
        log(f"dispatched {label}")
        time.sleep(2)          # be kind to the dispatch API

    if dry_run:
        return []
    while in_flight() > 0:
        time.sleep(60)
    sync()
    return [(c, n) for c, n in todo if not fragment_exists(c, n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path, default=REPO / "grid" / "grid.json")
    ap.add_argument("--phase", choices=["donor", "shim", "all"], default="all")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2,
                    help="extra passes over cells still missing a fragment")
    ap.add_argument("--only", default="",
                    help="substring filter over 'version flavour python platform'")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grid = json.loads(args.grid.read_text())
    all_cells = cells(grid)

    maxima: dict[str, dict] = {}
    for subdir in sorted({c["platform"] for c in all_cells}):
        maxima[subdir] = published_max(subdir)
        log(f"{subdir}: {len(maxima[subdir])} published cells")

    todo: list[tuple[dict, int]] = []
    for c in all_cells:
        c["flavour_build"] = c["flavour"].replace("cu", "cuda")
        key = (c["version"], c["flavour_build"], c["pytag"])
        nxt = maxima[c["platform"]].get(key, -1) + 1
        label = f"{c['version']} {c['flavour']} {c['python']} {c['platform']}"
        if args.only and args.only not in label:
            continue
        if fragment_exists(c, nxt):
            log(f"already landed {label} _{nxt}")
            continue
        todo.append((c, nxt))

    phases = []
    if args.phase in ("donor", "all"):
        phases.append(("donor", [t for t in todo if t[0]["donor"]]))
    if args.phase in ("shim", "all"):
        phases.append(("shim", [t for t in todo if not t[0]["donor"]]))

    failed: list[tuple[dict, int]] = []
    for name, batch in phases:
        if not batch:
            log(f"phase {name}: nothing to do")
            continue
        log(f"phase {name}: {len(batch)} cells")
        left = run_phase(batch, args.concurrency, args.dry_run)
        for attempt in range(args.retries):
            if not left:
                break
            log(f"phase {name}: retry {attempt + 1} over {len(left)} cells")
            left = run_phase(left, args.concurrency, args.dry_run)
        failed += left
        log(f"phase {name}: {len(batch) - len(left)}/{len(batch)} landed")

    if failed:
        log(f"{len(failed)} cells still missing a fragment:")
        for c, n in failed:
            log(f"  RED {c['version']} {c['flavour']} {c['python']} "
                f"{c['platform']} _{n}")
        sys.exit(1)
    log("wave complete: every dispatched cell has a fragment")


if __name__ == "__main__":
    main()
