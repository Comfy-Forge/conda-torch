#!/usr/bin/env python3
"""Repack an official PyPI torch wheel into two conda packages.

Split at conda-forge's line (measured against their real pytorch/libtorch
packages, 2026-09):

  libtorch  — the 9 big python-independent .so at $PREFIX/lib, real headers
              at $PREFIX/include, cmake at $PREFIX/share/cmake, plus the
              CUDA/openmp run deps. One per (version, flavour, platform).
  pytorch   — the site-packages tree, with the big libs/headers/cmake as
              relative symlinks back into $PREFIX, real libtorch_python.so
              at $PREFIX/lib, byte-compiled .pyc, torchrun entry point.
              One per python.

Every transformation here is a mitigation from the five-hacker investigation;
see README.md for the rationale of each. linux-64 only for now — other
platforms fail loudly rather than half-work.

Usage:
  torch_repack.py --version 2.8.0 --flavour cu128 --python 3.12 -o dist/

Must run under the exact python minor being targeted (for .pyc magic).
Requires: zstandard (pip), patchelf >= 0.14 on PATH.
"""
from __future__ import annotations

import argparse
import compileall
import py_compile
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import zstandard

EMPTY_SHA = hashlib.sha256(b"").hexdigest()

# 255-char dummy prefix; installers rewrite this byte range in place.
_P = "/opt/conda-torch/_h_env" + "_placehold" * 40
PLACEHOLDER = _P[:255]
assert len(PLACEHOLDER) == 255

# /bin/sh polyglot: the 255-char placeholder cannot live in a shebang
# (kernel truncates at 127 bytes), so line 2 re-execs through sh.
TORCHRUN = """#!/bin/sh
'''exec' {prefix}/bin/python "$0" "$@"
' '''
import sys
from torch.distributed.run import main
if __name__ == '__main__':
    sys.argv[0] = sys.argv[0].removesuffix('.exe')
    sys.exit(main())
""".format(prefix=PLACEHOLDER)

# The python-independent heavyweights (zero Py* symbols, verified): these move
# to $PREFIX/lib in libtorch. libtorch_python.so is per-python and stays in
# the pytorch package (also at $PREFIX/lib, conda-forge-style).
BIG_LIBS = [
    "libtorch_cuda.so",
    "libtorch_cpu.so",
    "libtorch_cuda_linalg.so",
    "libtorch.so",
    "libc10.so",
    "libc10_cuda.so",
    "libshm.so",
    "libcaffe2_nvrtc.so",
    "libtorch_global_deps.so",
]
# Exactly the objects with a direct CUDA DT_NEEDED (patch set is proven
# minimal; RPATH does not propagate between siblings).
PATCH_LIBS = [
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libtorch_cuda_linalg.so",
    "libc10_cuda.so",
    "libtorch_global_deps.so",
    "libcaffe2_nvrtc.so",
    "libtorch_python.so",
]
# From site-packages/torch/lib, $PREFIX/lib is exactly four levels up.
ADDED_RPATH = "$ORIGIN/../../../.."

# nvidia-*-cu12 wheel pin -> conda-forge package (all 14 verified to exist
# with matching versions). 'cudnn' on conda-forge is a metapackage squatting
# on the 8.x name — the real library is 'libcudnn'.
NVIDIA_MAP = {
    "nvidia-cuda-runtime-cu12": "cuda-cudart",
    "nvidia-cublas-cu12": "libcublas",
    "nvidia-cudnn-cu12": "libcudnn",
    "nvidia-cusparse-cu12": "libcusparse",
    "nvidia-cufft-cu12": "libcufft",
    "nvidia-curand-cu12": "libcurand",
    "nvidia-cusolver-cu12": "libcusolver",
    "nvidia-nccl-cu12": "nccl",
    "nvidia-cuda-cupti-cu12": "cuda-cupti",
    "nvidia-cuda-nvrtc-cu12": "cuda-nvrtc",
    "nvidia-nvjitlink-cu12": "libnvjitlink",
    "nvidia-cufile-cu12": "libcufile",
    "nvidia-cusparselt-cu12": "cusparselt",
    "nvidia-nvtx-cu12": "cuda-nvtx",
}
# Pure-python runtime deps: PyPI name -> conda-forge name.
PY_DEP_MAP = {
    "filelock": "filelock",
    "typing-extensions": "typing_extensions",
    "sympy": "sympy",
    "networkx": "networkx",
    "jinja2": "jinja2",
    "fsspec": "fsspec",
    "setuptools": "setuptools",
    "triton": "triton",
    "numpy": "numpy",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


def log(msg: str) -> None:
    print(f"[torch_repack] {msg}", flush=True)


# --------------------------------------------------------------------------
# wheel acquisition
# --------------------------------------------------------------------------

def wheel_url(version: str, flavour: str, py: str) -> str:
    cp = "cp" + py.replace(".", "")
    idx = f"https://download.pytorch.org/whl/{flavour}/torch/"
    html = urllib.request.urlopen(idx, timeout=60).read().decode()
    pat = re.compile(r'href="([^"]*torch-%s%%2B%s-%s-%s-manylinux[^"]*x86_64\.whl)[#"]'
                     % (re.escape(version), flavour, cp, cp))
    m = pat.search(html)
    if not m:
        sys.exit(f"no linux x86_64 wheel for torch {version}+{flavour} {cp} on {idx}")
    return m.group(1)


def download(url: str, dest: Path) -> None:
    if dest.exists():
        log(f"wheel already present: {dest}")
        return
    log(f"downloading {url}")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)
    tmp.rename(dest)
    log(f"downloaded {dest.stat().st_size} bytes")


# --------------------------------------------------------------------------
# metadata translation
# --------------------------------------------------------------------------

def parse_requires(dist_info: Path, py: str) -> list[tuple[str, str]]:
    """Marker-evaluated Requires-Dist for linux-64/cpython: [(name, spec)]."""
    env = {
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "sys_platform": "linux",
        "os_name": "posix",
        "python_version": py,
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
    }
    out = []
    for line in (dist_info / "METADATA").read_text(errors="replace").splitlines():
        if not line.strip():
            break
        if not line.lower().startswith("requires-dist:"):
            continue
        req = line.split(":", 1)[1].strip()
        marker = ""
        if ";" in req:
            req, marker = (s.strip() for s in req.split(";", 1))
        if "extra" in marker:
            continue
        if marker and not eval_marker(marker, env):
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)\s*(.*)$", req)
        if not m:
            sys.exit(f"unparseable Requires-Dist: {line}")
        name = m.group(1).lower().replace("_", "-").replace(".", "-")
        spec = m.group(2).strip().strip("()").strip()
        out.append((name, spec))
    return out


def eval_marker(marker: str, env: dict[str, str]) -> bool:
    """Tiny PEP 508 marker evaluator: ==, !=, >=, and/or/parens only."""
    expr = marker
    for k, v in env.items():
        expr = re.sub(rf"\b{k}\b", repr(v), expr)
    expr = re.sub(r'"([^"]*)"', r"'\1'", expr)
    if re.search(r"[A-Za-z_]{2,}", re.sub(r"\b(and|or|not|in)\b", "", re.sub(r"'[^']*'", "", expr))):
        sys.exit(f"marker has unknown variables, refusing to guess: {marker}")
    return bool(eval(expr, {"__builtins__": {}}))


def translate_deps(requires: list[tuple[str, str]]) -> list[str]:
    """PyPI requirement list -> conda depends for the *pytorch* package.

    nvidia-*/CUDA deps are handled separately (they belong to libtorch).
    """
    deps = []
    for name, spec in requires:
        if name in NVIDIA_MAP:
            continue  # carried by libtorch
        if name not in PY_DEP_MAP:
            sys.exit(f"unmapped PyPI dependency {name!r} ({spec!r}); add it to PY_DEP_MAP")
        conda = PY_DEP_MAP[name]
        spec = spec.replace("~=", ">=")
        if conda == "setuptools":
            # runtime dep on 3.12+; new setuptools removed pkg_resources
            spec = (spec + "," if spec else "") + "<82"
        deps.append(f"{conda} {spec}".strip())
    return deps


def cuda_deps(requires: list[tuple[str, str]], flavour: str) -> list[str]:
    """nvidia-* wheel pins -> conda-forge CUDA deps for the *libtorch* package.

    Wheel pin becomes the lower bound; cap at the next major (CUDA minor
    compatibility — an exact-minor cap is UNSAT because e.g. conda-forge
    triton 3.4.0 only exists as a cuda129 build).
    """
    cuda_minor = f"{flavour[2:-1]}.{flavour[-1]}"  # cu128 -> 12.8
    deps = ["__cuda", f"cuda-version >={cuda_minor},<{int(flavour[2:-1]) + 1}"]
    seen = set()
    for name, spec in requires:
        if name not in NVIDIA_MAP:
            continue
        m = re.match(r"^==\s*([0-9][0-9.]*)$", spec)
        if not m:
            sys.exit(f"expected an exact pin for {name}, got {spec!r}")
        v = m.group(1)
        cap = int(v.split(".")[0]) + 1
        deps.append(f"{NVIDIA_MAP[name]} >={v},<{cap}.0a0")
        seen.add(name)
    missing = {n for n, _ in requires if n.startswith("nvidia-")} - seen
    if missing:
        sys.exit(f"nvidia deps without a conda mapping: {missing}")
    return deps


# --------------------------------------------------------------------------
# tree surgery
# --------------------------------------------------------------------------

def sed_cmake(cmake_root: Path) -> None:
    """Neutralise the four absolute-path / ABI traps in shipped cmake files."""
    aten = cmake_root / "ATen" / "ATenConfig.cmake"
    txt = aten.read_text()
    new = re.sub(
        r'set\(ATEN_INCLUDE_DIR "[^"]*"\)',
        'get_filename_component(ATEN_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../include" ABSOLUTE)',
        txt,
    )
    if new == txt:
        sys.exit("ATenConfig.cmake: hard-coded include dir not found, layout changed?")
    aten.write_text(new)

    for tp in cmake_root.glob("*/TensorpipeTargets.cmake"):
        tp.write_text(tp.read_text().replace('"/usr/local/cuda/include"', '""'))

    mkl = cmake_root / "Caffe2" / "public" / "mkl.cmake"
    txt = mkl.read_text()
    i = txt.find("set_property(")
    while i != -1:
        depth, j = 0, i
        while j < len(txt):
            if txt[j] == "(":
                depth += 1
            elif txt[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        blk = txt[i:j + 1]
        if "MKL_ROOT" in blk:
            txt = txt[:i] + "if(MKL_ROOT)\n" + blk + "\nendif()" + txt[j + 1:]
            mkl.write_text(txt)
            break
        i = txt.find("set_property(", j)
    else:
        sys.exit("mkl.cmake: MKL_ROOT set_property block not found")

    tc = cmake_root / "Torch" / "TorchConfig.cmake"
    txt = tc.read_text()
    anchor = "if(TORCH_CXX_FLAGS)"
    if anchor not in txt:
        sys.exit("TorchConfig.cmake: TORCH_CXX_FLAGS anchor not found")
    txt = txt.replace(
        anchor,
        'set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")\n' + anchor,
        1,
    )
    tc.write_text(txt)


def make_symlink(path: Path, target: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)


# --------------------------------------------------------------------------
# package emission
# --------------------------------------------------------------------------

def collect_paths(stage: Path, prefix_files: dict[str, str]) -> tuple[list[dict], list[str]]:
    files: list[str] = []
    for root, dirs, names in os.walk(stage):
        dirs.sort()
        for n in sorted(names):
            files.append(str(Path(root, n).relative_to(stage)))
    # symlinked dirs are not walked into but must be listed
    for root, dirs, _ in os.walk(stage):
        for d in list(dirs):
            p = Path(root, d)
            if p.is_symlink():
                files.append(str(p.relative_to(stage)))
                dirs.remove(d)
    files.sort()

    paths = []
    for rel in files:
        p = stage / rel
        if p.is_symlink():
            entry = {"_path": rel, "path_type": "softlink"}
            tgt = p.resolve()
            if tgt.is_file():
                entry["sha256"] = sha256_file(tgt)
                entry["size_in_bytes"] = tgt.stat().st_size
            else:  # directory symlink (or dangling at build time)
                entry["sha256"] = EMPTY_SHA
                entry["size_in_bytes"] = 0
        else:
            entry = {"_path": rel, "path_type": "hardlink",
                     "sha256": sha256_file(p), "size_in_bytes": p.stat().st_size}
        if rel in prefix_files:
            entry["file_mode"] = prefix_files[rel]
            entry["prefix_placeholder"] = PLACEHOLDER
        paths.append(entry)
    return paths, files


def emit_conda(stage: Path, outdir: Path, index: dict, about: dict,
               prefix_files: dict[str, str]) -> Path:
    paths, files = collect_paths(stage, prefix_files)
    name, version, build = index["name"], index["version"], index["build"]
    stem = f"{name}-{version}-{build}"

    infodir = outdir / f"_info_{name}"
    if infodir.exists():
        shutil.rmtree(infodir)
    (infodir / "info").mkdir(parents=True)
    w = lambda n, s: (infodir / "info" / n).write_text(s)
    w("index.json", json.dumps(index, indent=2, sort_keys=True))
    w("paths.json", json.dumps({"paths": paths, "paths_version": 1}, indent=2))
    w("about.json", json.dumps(about, indent=2, sort_keys=True))
    w("files", "".join(f + "\n" for f in files))
    if prefix_files:
        w("has_prefix", "".join(
            f"{PLACEHOLDER} {mode} {rel}\n" for rel, mode in sorted(prefix_files.items())))

    cctx = zstandard.ZstdCompressor(level=19, threads=-1)

    def make_tar_zst(base: Path, members: list[str], dest: Path) -> Path:
        with open(dest, "wb") as raw, cctx.stream_writer(raw) as zw, \
                tarfile.open(fileobj=zw, mode="w|", format=tarfile.GNU_FORMAT) as tf:
            for rel in members:
                p = base / rel
                if p.is_symlink():
                    ti = tarfile.TarInfo(rel)
                    ti.type = tarfile.SYMTYPE
                    ti.linkname = os.readlink(p)
                    ti.mode = 0o777
                    ti.mtime = 0
                    tf.addfile(ti)
                    continue
                ti = tf.gettarinfo(str(p), arcname=rel)
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                ti.mtime = 0
                with open(p, "rb") as fh:
                    tf.addfile(ti, fh)
        return dest

    out = outdir / f"{stem}.conda"
    tmp = outdir / f"_tmp_{name}"
    tmp.mkdir(exist_ok=True)
    log(f"compressing {stem} ({len(files)} files)...")
    pkg_blob = make_tar_zst(stage, files, tmp / "pkg.tar.zst")
    info_members = sorted(str(p.relative_to(infodir)) for p in infodir.rglob("*") if p.is_file())
    info_blob = make_tar_zst(infodir, info_members, tmp / "info.tar.zst")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        z.writestr("metadata.json", json.dumps({"conda_pkg_format_version": 2}))
        z.write(pkg_blob, f"pkg-{stem}.tar.zst")
        z.write(info_blob, f"info-{stem}.tar.zst")
    shutil.rmtree(tmp)
    shutil.rmtree(infodir)
    log(f"built {out.name}: {out.stat().st_size} bytes, sha256 {sha256_file(out)}")
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--flavour", required=True, help="e.g. cu128")
    ap.add_argument("--python", dest="py", required=True, help="e.g. 3.12")
    ap.add_argument("--subdir", default="linux-64")
    ap.add_argument("--build-number", type=int, default=0)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("dist"))
    ap.add_argument("--wheel", type=Path, default=None,
                    help="use a pre-downloaded wheel instead of fetching")
    ap.add_argument("--work", type=Path, default=Path("work"))
    args = ap.parse_args()

    if args.subdir != "linux-64":
        sys.exit(f"only linux-64 is implemented; {args.subdir} needs its own trap review "
                 "(aarch64: nvshmem + torch.libs; win: DLL strategy)")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != args.py:
        sys.exit(f"must run under python {args.py} for .pyc magic "
                 f"(running {sys.version_info.major}.{sys.version_info.minor})")
    if shutil.which("patchelf") is None:
        sys.exit("patchelf not on PATH")

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    wheel = args.wheel
    if wheel is None:
        url = wheel_url(args.version, args.flavour, args.py)
        wheel = args.work / PurePosixPath(url.split("#")[0]).name.replace("%2B", "+")
        download(url, wheel)

    # ---- extract full wheel into the pytorch stage -------------------------
    pt_stage = args.work / "pytorch_stage"
    lt_stage = args.work / "libtorch_stage"
    for s in (pt_stage, lt_stage):
        if s.exists():
            shutil.rmtree(s)
    sp_rel = Path(f"lib/python{args.py}/site-packages")
    sp = pt_stage / sp_rel
    sp.mkdir(parents=True)

    log("extracting wheel (enumerating the zip namelist, never top_level.txt)...")
    zf = zipfile.ZipFile(wheel)
    for zi in zf.infolist():
        if zi.is_dir():
            continue
        dest = sp / zi.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(zi) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        mode = (zi.external_attr >> 16) & 0o7777
        if mode:
            os.chmod(dest, mode)
    zf.close()

    dist_info = next(sp.glob("torch-*.dist-info"))
    requires = parse_requires(dist_info, args.py)

    # ---- dist-info: pip must never think it owns this ----------------------
    (dist_info / "RECORD").unlink()
    (dist_info / "INSTALLER").write_bytes(b"conda")  # exactly 5 bytes, no \n
    for junk in ("direct_url.json", "RECORD.jws", "REQUESTED"):
        (dist_info / junk).unlink(missing_ok=True)

    # ---- byte-compile with the target interpreter (checked-hash: immune to
    # the mtimes conda extraction produces) ---------------------------------
    log("byte-compiling...")
    for top in ("torch", "torchgen", "functorch"):
        compileall.compile_dir(sp / top, quiet=2, workers=0,
                               invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)

    # ---- split: big libs to $PREFIX/lib, symlinks back ---------------------
    tlib = sp / "torch" / "lib"
    (lt_stage / "lib").mkdir(parents=True)
    for so in BIG_LIBS:
        shutil.move(tlib / so, lt_stage / "lib" / so)
        make_symlink(tlib / so, f"../../../../{so}")
    # vendored OpenMP out, conda-forge libgomp in (soname-identical, dedupes)
    gomp = list(tlib.glob("libgomp*"))
    if len(gomp) != 1:
        sys.exit(f"expected one vendored libgomp, found {gomp}")
    gomp[0].unlink()

    # libtorch_python.so is per-python: real at $PREFIX/lib in *pytorch*
    (pt_stage / "lib").mkdir(exist_ok=True)
    shutil.move(tlib / "libtorch_python.so", pt_stage / "lib" / "libtorch_python.so")
    make_symlink(tlib / "libtorch_python.so", "../../../../libtorch_python.so")

    # ---- headers: real in libtorch at $PREFIX/include, dir-symlinks back.
    # pybind11 stays REAL in the package: moving it would clobber
    # $PREFIX/include/pybind11 from the pybind11 package.
    tinc = sp / "torch" / "include"
    (lt_stage / "include").mkdir()
    for entry in sorted(tinc.iterdir()):
        if entry.name == "pybind11":
            continue
        shutil.move(entry, lt_stage / "include" / entry.name)
        make_symlink(tinc / entry.name, f"../../../../../include/{entry.name}")

    # ---- cmake: seds first, then real tree to $PREFIX/share/cmake ----------
    tcmake = sp / "torch" / "share" / "cmake"
    sed_cmake(tcmake)
    (lt_stage / "share").mkdir()
    shutil.move(tcmake, lt_stage / "share" / "cmake")
    # link lives at torch/share/cmake, resolved from torch/share: 5 ups to $PREFIX
    make_symlink(tcmake, "../../../../../share/cmake")

    # torch/bin stays fully real inside pytorch: torch_shm_manager MUST exist
    # at $SP_DIR/torch/bin/ (import hard-fails otherwise) and protoc must NOT
    # go to $PREFIX/bin (it would clobber libprotobuf's).

    # ---- patchelf ----------------------------------------------------------
    log("patching RPATHs (--force-rpath: RUNPATH would lose to LD_LIBRARY_PATH)...")
    for so in PATCH_LIBS:
        target = (pt_stage / "lib" / so) if so == "libtorch_python.so" else (lt_stage / "lib" / so)
        run("patchelf", "--force-rpath", "--add-rpath", ADDED_RPATH, str(target))
        got = subprocess.run(["patchelf", "--print-rpath", str(target)],
                             capture_output=True, text=True, check=True).stdout
        if ADDED_RPATH not in got:
            sys.exit(f"rpath patch did not stick on {so}: {got}")

    # ---- entry point -------------------------------------------------------
    bindir = pt_stage / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "torchrun").write_text(TORCHRUN)
    os.chmod(bindir / "torchrun", 0o755)
    prefix_files = {"bin/torchrun": "text"}
    # torchfrtrace deliberately NOT generated: its module (tools.flight_recorder)
    # is not in the wheel; the upstream entry point is broken.

    # ---- metadata ----------------------------------------------------------
    flavour, py = args.flavour, args.py
    pytag = "py" + py.replace(".", "")
    now = int(time.time() * 1000)
    lt_deps = cuda_deps(requires, flavour) + [
        "libgomp", "libgcc >=12", "libstdcxx >=12", "__glibc >=2.28",
    ]
    hsh = lambda s: hashlib.sha256(s.encode()).hexdigest()[:8]
    lt_build = f"cuda{flavour[2:]}_repack_h{hsh('libtorch|' + args.version + '|' + flavour)}_{args.build_number}"
    pt_build = f"cuda{flavour[2:]}_repack_{pytag}_h{hsh('pytorch|' + args.version + '|' + flavour + '|' + py)}_{args.build_number}"

    lt_index = {
        "arch": "x86_64", "platform": "linux", "subdir": args.subdir,
        "name": "libtorch", "version": args.version,
        "build": lt_build, "build_number": args.build_number,
        "depends": sorted(set(lt_deps)),
        "constrains": [f"pytorch {args.version} cuda{flavour[2:]}_repack_*"],
        "license": "BSD-3-Clause", "license_family": "BSD",
        "timestamp": now,
    }
    nxt = f"{py.split('.')[0]}.{int(py.split('.')[1]) + 1}"
    pt_deps = translate_deps(requires) + [
        f"python >={py},<{nxt}.0a0",
        f"python_abi {py}.* *_cp{py.replace('.', '')}",
        f"libtorch {args.version} {lt_build}",
    ]
    pt_index = {
        "arch": "x86_64", "platform": "linux", "subdir": args.subdir,
        "name": "pytorch", "version": args.version,
        "build": pt_build, "build_number": args.build_number,
        "depends": sorted(set(pt_deps)),
        "constrains": ["pytorch-cpu <0.0a0", "pytorch-gpu <0.0a0"],
        "license": "BSD-3-Clause", "license_family": "BSD",
        "timestamp": now,
    }
    about = {
        "home": "https://pytorch.org",
        "license": "BSD-3-Clause",
        "summary": "PyTorch, repacked byte-for-byte from the official PyPI wheel",
        "description": f"Repacked from {wheel.name}. See github.com/Comfy-Forge/conda-torch.",
        "extra": {"repacked_from": wheel.name, "wheel_sha256": sha256_file(wheel)},
    }

    emit_conda(lt_stage, args.outdir, lt_index, about, {})
    emit_conda(pt_stage, args.outdir, pt_index, about, prefix_files)
    log("done")


if __name__ == "__main__":
    main()
