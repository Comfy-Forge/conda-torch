#!/usr/bin/env python3
"""win-64 clean-environment gate: solve, resolve every DLL import, import torch.

The one check that would have caught the shipped defect: the win-64
repack had never been imported in a clean environment (every consumer
built and tested against the conda-forge mirror, which outranks it), and
`import torch` died with "Error loading shm.dll or one of its
dependencies" -- a message that names the first DLL the loader gave up on
and never the dependency that is actually absent.

Three stages, each fatal:

  1. SOLVE   pixi resolves the requested pytorch build into a fresh env
             from the given channels under strict priority, with the
             flavour's CUDA as the virtual __cuda (GPU-less runner). When
             --expect-build is given the solved libtorch/pytorch build
             numbers must match: a solve that silently lands on an older
             wide-window build is exactly the failure mode being gated.
  2. RESOLVE tools/win_dll_audit.py walks the PE import tree of every DLL
             under torch/lib against the env the way the loader would, and
             prints every basename that resolves NOWHERE. That list is the
             diagnosis; a non-empty list fails before python is even asked.
  3. IMPORT  `import torch`, a CPU op, and torch.cuda.is_available(), which
             must not raise (False is the right answer without a GPU).
             Optionally a consumer (torchvision et al.) is solved on top
             and imported too.

Runs on Windows. Requires pixi on PATH. Exit 1 = gate failed; exit 3 = the
requested python minor is not installable on win-64 from these channels
(untestable, not defective).

Usage:
  win_gate.py --workdir W --python 3.12 --cuda 12.8 \
      --spec "pytorch ==2.8.0 cuda128_repack_py312_*" \
      -c file:///D:/a/_temp/localchan -c https://comfy-forge.github.io/conda-torch -c conda-forge \
      [--expect-build 5] [--consumer "torchvision 0.23.* cuda128_torch28_py312_*"]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"[win-gate] {msg}", flush=True)


def run(cmd: list[str], cwd: Path, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=env, check=check, text=True)


def toml_str(s: str) -> str:
    return json.dumps(s)  # JSON string literals are valid TOML basic strings


def write_manifest(workdir: Path, channels: list[str], python: str, specs: list[str],
                   cuda: str) -> None:
    deps = [f"python = {toml_str(python + '.*')}", "pefile = \"*\""]
    for spec in specs:
        # "name ==version build_glob" / "name version build_glob" / "name"
        parts = spec.split()
        name = parts[0]
        if len(parts) == 1:
            deps.append(f"{name} = \"*\"")
        elif len(parts) == 2:
            deps.append(f"{name} = {toml_str(parts[1])}")
        else:
            ver = parts[1]
            deps.append(f"{name} = {{ version = {toml_str(ver)}, build = {toml_str(parts[2])} }}")
    # the virtual __cuda rides the platform entry (pixi >= 0.79 syntax; the
    # old [system-requirements] table is deprecated), CONDA_OVERRIDE_CUDA is
    # exported as well for anything that reads the env instead
    text = f"""[workspace]
name = "win-gate"
channels = [{", ".join(toml_str(c) for c in channels)}]
platforms = [{{ name = "win-gate", platform = "win-64", cuda = {toml_str(cuda)} }}]
channel-priority = "strict"

[dependencies]
{chr(10).join(deps)}
"""
    (workdir / "pixi.toml").write_text(text)
    log("pixi.toml:\n" + text)


def solved(workdir: Path, env: dict) -> dict[str, dict]:
    out = subprocess.run(["pixi", "list", "--json", "--platform", "win-gate"],
                         cwd=workdir, env=env, capture_output=True, text=True, check=True).stdout
    return {p["name"]: p for p in json.loads(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--python", required=True, help="python minor, e.g. 3.12")
    ap.add_argument("--cuda", required=True, help="virtual __cuda version, e.g. 12.8")
    ap.add_argument("--spec", action="append", required=True,
                    help="match spec, e.g. 'pytorch ==2.8.0 cuda128_repack_py312_*'")
    ap.add_argument("-c", "--channel", action="append", required=True)
    ap.add_argument("--expect-build", type=int, default=None,
                    help="libtorch AND pytorch must solve at exactly this build number")
    ap.add_argument("--consumer", action="append", default=[],
                    help="extra spec solved on top (a package that imports torch); "
                         "its top-level module is imported too (name before the "
                         "first space, '-' -> '_')")
    ap.add_argument("--consumer-channel", action="append", default=[],
                    help="channels prepended for the consumer solve only")
    args = ap.parse_args()

    if os.name != "nt":
        sys.exit("win_gate.py runs on Windows (the loader being tested is Windows')")
    args.workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CONDA_OVERRIDE_CUDA=args.cuda)

    # ---- 0. is the interpreter even available? ------------------------------
    # A python minor with no win-64 candidate on any channel (py3.15 until
    # conda-forge ships it) makes the cell untestable, not defective: exit
    # 3 so the caller can tell that apart from a gate failure. The shared
    # libtorch half is still gated through the line's other pythons.
    probe_dir = args.workdir / "python-probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(probe_dir, args.channel, args.python, [], args.cuda)
    if subprocess.run(["pixi", "lock"], cwd=probe_dir, env=env).returncode != 0:
        log(f"python {args.python} has no win-64 candidate on {args.channel}: "
            "the cell cannot be installed anywhere yet, so it cannot be gated")
        sys.exit(3)

    # ---- 1. solve -----------------------------------------------------------
    write_manifest(args.workdir, args.channel, args.python, args.spec, args.cuda)
    run(["pixi", "install", "-v"], args.workdir, env)
    pk = solved(args.workdir, env)
    for name in ("pytorch", "libtorch"):
        if name not in pk:
            sys.exit(f"{name} not in the solved env: {sorted(pk)}")
        p = pk[name]
        log(f"solved {name} {p['version']} {p['build']} from {p.get('source')}")
        if args.expect_build is not None and int(p["build_number"]) != args.expect_build:
            sys.exit(f"{name} solved to build {p['build']!r}, expected build number "
                     f"{args.expect_build}: the requested build did not win the solve")
    for name in sorted(pk):
        p = pk[name]
        if p["name"].startswith(("cuda", "lib", "intel", "vc", "ucrt", "python")):
            log(f"  {p['name']:<22} {p['version']:<14} {p['build']:<34} {p.get('source')}")

    prefix = args.workdir / ".pixi" / "envs" / "default"
    envpy = prefix / "python.exe"
    if not envpy.is_file():
        sys.exit(f"no python.exe under {prefix}")

    # ---- 2. resolve every DLL import ----------------------------------------
    audit = run([str(envpy), str(HERE / "win_dll_audit.py"), "--prefix", str(prefix),
                 "--json", str(args.workdir / "dll-audit.json")], args.workdir, env, check=False)
    if audit.returncode != 0:
        sys.exit("DLL audit found imports that resolve nowhere (list above); "
                 "the package cannot load in this environment")

    # ---- 3. import ----------------------------------------------------------
    probe = (
        "import torch, sys\n"
        "print('torch', torch.__version__, 'cuda', torch.version.cuda, 'py', sys.version.split()[0])\n"
        "print('torch/lib:', torch.__file__)\n"
        "x = torch.ones(3, 3) @ torch.ones(3, 3)\n"
        "assert float(x.sum()) == 27.0, x\n"
        "print('cpu matmul ok')\n"
        "print('torch.cuda.is_available():', torch.cuda.is_available())\n"
        "import torch.nn, torch.utils.data, torch.profiler\n"
        "print('submodules ok')\n"
    )
    run([str(envpy), "-c", probe], args.workdir, env)
    log("import torch: OK")

    # ---- consumer (optional) ------------------------------------------------
    if args.consumer:
        cdir = args.workdir / "consumer"
        cdir.mkdir(exist_ok=True)
        write_manifest(cdir, args.consumer_channel + args.channel, args.python,
                       args.spec + args.consumer, args.cuda)
        run(["pixi", "install", "-v"], cdir, env)
        cpk = solved(cdir, env)
        for name in ("pytorch", "libtorch"):
            log(f"consumer env: {name} {cpk[name]['version']} {cpk[name]['build']}")
            if args.expect_build is not None and int(cpk[name]["build_number"]) != args.expect_build:
                sys.exit(f"consumer solve landed on {name} {cpk[name]['build']!r}, "
                         f"expected build number {args.expect_build}")
        cprefix = cdir / ".pixi" / "envs" / "default"
        cpy = cprefix / "python.exe"
        audit = run([str(cpy), str(HERE / "win_dll_audit.py"), "--prefix", str(cprefix)],
                    cdir, env, check=False)
        if audit.returncode != 0:
            sys.exit("DLL audit failed in the consumer environment")
        mods = [c.split()[0].replace("-", "_") for c in args.consumer]
        run([str(cpy), "-c", "import torch\n" + "".join(
            f"import {m}; print({m!r}, getattr({m}, '__version__', '?'))\n" for m in mods)
             + "print('torch.cuda.is_available():', torch.cuda.is_available())\n"], cdir, env)
        log("consumer import: OK")

    log("GATE PASSED: solve + every DLL import resolves + import torch")


if __name__ == "__main__":
    main()
