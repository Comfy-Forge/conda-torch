#!/usr/bin/env python3
"""linux clean-environment gate: solve, resolve every NEEDED soname, import torch.

The linux twin of tools/win_gate.py, and it exists for the same reason that
one did: a linux repack is assembled on an x86 runner by unzipping a wheel
and rewriting ELF headers with patchelf -- nothing in the pipeline ever
LOADS the result. On linux-aarch64 that gap is total: until this gate, no
aarch64 repack had ever been imported anywhere, by anyone, in a clean
environment. The structural checks (verify_repack.py, the RPATH lint, the
sleef gate) all read the bytes; only the loader can say they resolve.

Three stages, each fatal:

  1. SOLVE   pixi resolves the requested pytorch build into a fresh env from
             the given channels under strict priority, with the flavour's
             CUDA declared as the virtual __cuda (the runner has no GPU and
             our libtorch hard-depends on __cuda). With --expect-build the
             solved libtorch AND pytorch build numbers must match -- a solve
             that silently lands on an older build, or on the conda-forge
             mirror that outranks the repack by build number, is exactly the
             failure this gates.
  2. RESOLVE every DT_NEEDED of every ELF in the environment's lib/ and
             site-packages/torch/lib is resolved with ldd, as the loader
             would. Anything reported "not found" fails the gate, with two
             named exceptions: the sonames a GPU-less machine legitimately
             lacks (libcuda.so.1, libnvidia-ml.so.1 -- shipped by the NVIDIA
             driver, never by conda) and a soname needed ONLY by nvshmem's
             dlopen'd transport/bootstrap plugins, whose absence nvshmem
             tolerates by design. Both are reported, never silent.
  3. IMPORT  `import torch`, a CPU matmul, torch.cuda.is_available() (False
             is the right answer without a GPU, raising is not), and a few
             submodules. Optionally a consumer package is solved on top and
             imported too.

The runner's architecture must match --platform: this gate proves the
artifact loads, which cannot be done from another architecture.

Exit 0 = passed, 1 = failed, 3 = the requested python minor is not
installable on this platform from these channels (untestable, not
defective -- same contract as win_gate.py).

Usage:
  linux_gate.py --workdir W --python 3.12 --cuda 13.0 --platform linux-aarch64 \
      --spec "pytorch ==2.11.0 cuda130_repack_py312_*" \
      -c https://comfy-forge.github.io/conda-torch -c conda-forge [--expect-build 0]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Sonames the NVIDIA DRIVER installs, never a conda package. On a GPU-less
# runner they are absent by construction and `ldd` says "not found" for
# every CUDA-linked library; that is the expected state of the machine, not
# a defect in the artifact. Everything else must resolve.
DRIVER_PROVIDED = {"libcuda.so.1", "libnvidia-ml.so.1"}

# nvshmem probes its transports at run time: nvshmem_bootstrap_*/transport_*
# are dlopen'd opportunistically and an absent one is a capability that is
# simply not offered, never a load failure of anything torch links. They are
# the one class of object allowed to NEED something the env does not have
# (torch_repack.py drops the plugins whose dependencies are unobtainable;
# one that survives because the env COULD supply the soname -- an MPI
# bootstrap next to conda-forge openmpi -- is legitimate).
OPTIONAL_PLUGIN = re.compile(r"^nvshmem_(transport|bootstrap)_")

ARCH_OF = {"linux-64": "x86_64", "linux-aarch64": "aarch64"}


def log(msg: str) -> None:
    print(f"[linux-gate] {msg}", flush=True)


def run(cmd: list[str], cwd: Path, env: dict | None = None,
        check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=env, check=check, text=True)


def toml_str(s: str) -> str:
    return json.dumps(s)  # JSON string literals are valid TOML basic strings


def write_manifest(workdir: Path, channels: list[str], python: str,
                   specs: list[str], cuda: str, subdir: str) -> None:
    deps = [f"python = {toml_str(python + '.*')}"]
    for spec in specs:
        parts = spec.split()
        name = parts[0]
        if len(parts) == 1:
            deps.append(f"{name} = \"*\"")
        elif len(parts) == 2:
            deps.append(f"{name} = {toml_str(parts[1])}")
        else:
            deps.append(f"{name} = {{ version = {toml_str(parts[1])}, "
                        f"build = {toml_str(parts[2])} }}")
    text = f"""[workspace]
name = "linux-gate"
channels = [{", ".join(toml_str(c) for c in channels)}]
platforms = [{{ name = "linux-gate", platform = {toml_str(subdir)}, cuda = {toml_str(cuda)} }}]
channel-priority = "strict"

[dependencies]
{chr(10).join(deps)}
"""
    (workdir / "pixi.toml").write_text(text)
    log("pixi.toml:\n" + text)


def solved(workdir: Path, env: dict) -> dict[str, dict]:
    out = subprocess.run(["pixi", "list", "--json", "--platform", "linux-gate"],
                         cwd=workdir, env=env, capture_output=True, text=True,
                         check=True).stdout
    return {p["name"]: p for p in json.loads(out)}


def elf_audit(prefix: Path, out_json: Path | None) -> list[str]:
    """Every DT_NEEDED of every ELF in the env, resolved the way the loader
    would (ldd honours DT_RPATH/$ORIGIN exactly as ld.so does). Returns the
    unresolved ones, driver-provided sonames excluded."""
    roots = [prefix / "lib"]
    roots += sorted({p.resolve() for p in prefix.glob("lib/python3.*/site-packages/torch/lib")})
    seen: dict[str, list[str]] = {}
    scanned = 0
    for root in roots:
        if not root.is_dir():
            continue
        for so in sorted(root.rglob("*.so*")):
            if not so.is_file() or so.is_symlink():
                continue
            with open(so, "rb") as fh:
                if fh.read(4) != b"\x7fELF":
                    continue
            scanned += 1
            p = subprocess.run(["ldd", str(so)], capture_output=True, text=True)
            for line in p.stdout.splitlines():
                m = re.match(r"\s*(\S+) => not found", line)
                if m:
                    seen.setdefault(m.group(1), []).append(so.name)
    log(f"ELF audit: scanned {scanned} objects under {[str(r) for r in roots]}")
    fatal = []
    for soname, users in sorted(seen.items()):
        if soname in DRIVER_PROVIDED:
            kind = "driver-provided (expected on a GPU-less runner)"
        elif all(OPTIONAL_PLUGIN.match(u) for u in users):
            kind = "optional nvshmem plugin dependency (dlopen'd, absence tolerated)"
        else:
            kind = "UNRESOLVED"
            fatal.append(soname)
        log(f"  {soname}: {kind}; needed by {len(users)} object(s), e.g. {sorted(set(users))[:3]}")
    if out_json is not None:
        out_json.write_text(json.dumps(
            {"scanned": scanned,
             "unresolved": {k: sorted(set(v)) for k, v in seen.items()},
             "fatal": fatal,
             "driver_provided": sorted(DRIVER_PROVIDED)}, indent=2, sort_keys=True) + "\n")
    return sorted(set(fatal))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--python", required=True, help="python minor, e.g. 3.12")
    ap.add_argument("--cuda", required=True, help="virtual __cuda version, e.g. 13.0")
    ap.add_argument("--platform", default="linux-aarch64", choices=sorted(ARCH_OF))
    ap.add_argument("--spec", action="append", required=True,
                    help="match spec, e.g. 'pytorch ==2.11.0 cuda130_repack_py312_*'")
    ap.add_argument("-c", "--channel", action="append", required=True)
    ap.add_argument("--expect-build", type=int, default=None,
                    help="libtorch AND pytorch must solve at exactly this build number")
    ap.add_argument("--consumer", action="append", default=[])
    ap.add_argument("--consumer-channel", action="append", default=[])
    ap.add_argument("--json", type=Path, default=None, help="write the ELF audit here")
    args = ap.parse_args()

    if sys.platform != "linux":
        sys.exit("linux_gate.py runs on linux (the loader being tested is ld.so)")
    want = ARCH_OF[args.platform]
    if platform.machine() != want:
        sys.exit(f"--platform {args.platform} needs a {want} runner; this is "
                 f"{platform.machine()}. The gate proves the artifact LOADS, which "
                 "no other architecture can answer.")
    args.workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CONDA_OVERRIDE_CUDA=args.cuda)

    # ---- 0. is the interpreter even available? -----------------------------
    probe = args.workdir / "python-probe"
    probe.mkdir(parents=True, exist_ok=True)
    write_manifest(probe, args.channel, args.python, [], args.cuda, args.platform)
    if subprocess.run(["pixi", "lock"], cwd=probe, env=env).returncode != 0:
        log(f"python {args.python} has no {args.platform} candidate on {args.channel}: "
            "the cell cannot be installed anywhere yet, so it cannot be gated")
        sys.exit(3)

    # ---- 1. solve ----------------------------------------------------------
    write_manifest(args.workdir, args.channel, args.python, args.spec, args.cuda,
                   args.platform)
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
        if p["name"].startswith(("cuda", "lib", "nvidia", "sleef", "triton", "python")):
            log(f"  {p['name']:<22} {p['version']:<14} {p['build']:<34} {p.get('source')}")

    prefix = args.workdir / ".pixi" / "envs" / "default"
    envpy = prefix / "bin" / "python"
    if not envpy.is_file():
        sys.exit(f"no bin/python under {prefix}")

    # ---- 2. resolve every NEEDED soname ------------------------------------
    unresolved = elf_audit(prefix, args.json)
    if unresolved:
        sys.exit(f"sonames that resolve nowhere in the installed env: {unresolved}")

    # ---- 3. import ---------------------------------------------------------
    probe_src = (
        "import torch, sys\n"
        "print('torch', torch.__version__, 'cuda', torch.version.cuda, "
        "'py', sys.version.split()[0], 'machine', __import__('platform').machine())\n"
        "print('torch package:', torch.__file__)\n"
        "x = torch.ones(3, 3) @ torch.ones(3, 3)\n"
        "assert float(x.sum()) == 27.0, x\n"
        "print('cpu matmul ok')\n"
        "y = torch.nn.Linear(4, 2)(torch.zeros(1, 4))\n"
        "assert y.shape == (1, 2), y.shape\n"
        "print('nn forward ok')\n"
        "print('torch.cuda.is_available():', torch.cuda.is_available())\n"
        "import torch.nn, torch.utils.data, torch.profiler\n"
        "print('submodules ok')\n"
    )
    run([str(envpy), "-c", probe_src], args.workdir, env)
    log("import torch: OK")

    if args.consumer:
        cdir = args.workdir / "consumer"
        cdir.mkdir(exist_ok=True)
        write_manifest(cdir, args.consumer_channel + args.channel, args.python,
                       args.spec + args.consumer, args.cuda, args.platform)
        run(["pixi", "install", "-v"], cdir, env)
        cpk = solved(cdir, env)
        for name in ("pytorch", "libtorch"):
            log(f"consumer env: {name} {cpk[name]['version']} {cpk[name]['build']}")
            if args.expect_build is not None and int(cpk[name]["build_number"]) != args.expect_build:
                sys.exit(f"consumer solve landed on {name} {cpk[name]['build']!r}, "
                         f"expected build number {args.expect_build}")
        cprefix = cdir / ".pixi" / "envs" / "default"
        left = elf_audit(cprefix, None)
        if left:
            sys.exit(f"consumer env has unresolved sonames: {left}")
        mods = [c.split()[0].replace("-", "_") for c in args.consumer]
        run([str(cprefix / "bin" / "python"), "-c", "import torch\n" + "".join(
            f"import {m}; print({m!r}, getattr({m}, '__version__', '?'))\n" for m in mods)
            + "print('torch.cuda.is_available():', torch.cuda.is_available())\n"], cdir, env)
        log("consumer import: OK")

    log("GATE PASSED: solve + every NEEDED soname resolves + import torch")


if __name__ == "__main__":
    main()
