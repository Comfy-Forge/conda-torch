#!/usr/bin/env python3
"""Repack an official PyPI torch wheel into two conda packages.

Split at conda-forge's line (measured against their real pytorch/libtorch
packages, 2026-09):

  libtorch  — the big python-independent binaries, real headers, cmake,
              plus the CUDA/openmp run deps. One per (version, flavour,
              platform).
  pytorch   — the site-packages tree, byte-compiled .pyc, torchrun entry
              point, libtorch_python. One per python.

Per-platform layout:

  linux-64        big .so at $PREFIX/lib, symlinks back into torch/lib.
  linux-aarch64   same, plus: vendored NVPL/ACL/gfortran go to the private
                  $PREFIX/lib/libtorch-vendored/ (their top-level names
                  would clobber libgfortran5 et al.), and the hard
                  nvshmem dependency is repacked as its own package
                  (conda-forge has no nvshmem at all).
  win-64          site-packages is python-version-independent on Windows
                  (Lib/site-packages), so libtorch ships the big DLLs,
                  import libs, headers and cmake AT their wheel locations
                  inside torch/ — no relocation, no symlinks, and both
                  cpp_extension's -L/-I and the cmake IMPORTED_LOCATION
                  checks keep working. Vendored CUDA DLLs and 2.7 GB of
                  dead .lib archives are stripped; CUDA comes from
                  conda-forge packages whose DLLs land in Library/bin
                  (already on torch's own search path).

Every transformation here is a mitigation from the five-hacker
investigation; see README.md for the rationale of each.

Usage:
  torch_repack.py --version 2.8.0 --flavour cu128 --python 3.12 \
                  --subdir linux-64 -o dist/

Must run under the exact python minor being targeted (for .pyc magic).
Requires: zstandard (pip), patchelf >= 0.14 on PATH (linux subdirs), and
pip's vendored distlib t64.exe (win-64 entry point launcher).
"""
from __future__ import annotations

import argparse
import compileall
import py_compile
import hashlib
import io
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
    if sys.argv[0].endswith('.exe'):  # not str.removesuffix: py3.8 targets
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())
""".format(prefix=PLACEHOLDER)

# Windows entry point: the classic conda-build PAIR — a plain setuptools
# cli-64.exe launcher (runs its sibling torchrun-script.py) plus the
# script as a TEXT file with a registered placeholder shebang. The
# previous appended-archive form (distlib t64 + dead C:\bld shebang) was
# dead on arrival: launcher.c passes unknown shebangs verbatim to
# CreateProcessW with no fallback, and neither conda nor rattler rewrites
# an unregistered binary shebang. Text prefix replacement is the one
# mechanism both installers implement reliably on Windows. (conda-forge's
# own torchrun.exe has the same dead-shebang defect.)
WIN_MAIN = (
    b"import sys\n"
    b"from torch.distributed.run import main\n"
    b"if __name__ == '__main__':\n"
    b"    if sys.argv[0].endswith('.exe'):\n"
    b"        sys.argv[0] = sys.argv[0][:-4]\n"
    b"    sys.exit(main())\n"
)

PLATFORMS = {
    "linux-64": {
        "arch": "x86_64", "platform": "linux",
        # torch <= 2.5 published plain linux_x86_64 tags, not manylinux
        "wheel_re": r"(?:manylinux[^\"]*|linux)_x86_64\.whl",
        "markers": {"platform_system": "Linux", "platform_machine": "x86_64",
                    "sys_platform": "linux", "os_name": "posix"},
    },
    "linux-aarch64": {
        "arch": "aarch64", "platform": "linux",
        "wheel_re": r"(?:manylinux[^\"]*|linux)_aarch64\.whl",
        "markers": {"platform_system": "Linux", "platform_machine": "aarch64",
                    "sys_platform": "linux", "os_name": "posix"},
    },
    "win-64": {
        "arch": "x86_64", "platform": "win",
        "wheel_re": r"win_amd64\.whl",
        "markers": {"platform_system": "Windows", "platform_machine": "AMD64",
                    "sys_platform": "win32", "os_name": "nt"},
    },
}

# The big python-independent .so set (zero Py* symbols, verified) is
# enumerated from the wheel at split time — torch >=2.9 grew
# libtorch_nvshmem on x86, old versions ship fewer libs. libtorch_python
# is per-python and stays in the pytorch package (at $PREFIX/lib,
# conda-forge-style).
# From site-packages/torch/lib, $PREFIX/lib is exactly four levels up.
# RPATHs are SET wholesale, never appended: the wheel's inherited
# $ORIGIN/../../nvidia/* entries resurrect the moment a pip nvidia-*
# wheel lands in the env and outrank $PREFIX/lib (proven with a planted
# cudart), and old wheels carry absolute /usr/local/cuda entries that
# shadow the prefix on Jetson/SBSA boxes.
ADDED_RPATH = "$ORIGIN/../../../.."
# torch's own header trees — the ONLY ones relocated to $PREFIX/include.
# Every third-party tree (google/, fmt/, dnnl*, sleef*, asmjit, fbgemm,
# kineto, oneapi, KleidiAI, pybind11, ...) stays REAL in-package: at
# $PREFIX/include it would silently clobber the libprotobuf / fmt /
# onednn / sleef packages' headers on co-install.
TORCH_OWN_HEADERS = {"ATen", "c10", "caffe2", "torch", "tensorpipe"}
# aarch64 private dir for wheel-vendored libs whose top-level names would
# clobber real conda-forge files ($PREFIX/lib/libgfortran.so.5 belongs to
# libgfortran5, NVPL/ACL names could belong to future feedstocks). ABI
# fidelity matters more than dedup here: torch pins an exact ACL build.
VENDOR_DIR = "libtorch-vendored"
VENDOR_PRIVATE = re.compile(r"^lib(arm_compute|nvpl_|gfortran|openblas)")

# The vendored-CUDA era (old aarch64 wheels, e.g. 2.4.1/2.5.1 cu124):
# torch/lib carries the whole CUDA stack with stock sonames. Those are
# STRIPPED and replaced by conda-forge deps — moving them to $PREFIX/lib
# would clobber conda-forge's own files, keeping them would double-load
# against any other CUDA-using conda package, and the package would blow
# the 2 GiB asset cap (~2.3 GB compressed vendored).
CUDA_VENDORED = re.compile(
    r"^lib(cudart|cublasLt|cublas|cudnn(?:_\w+)?|cufftw|cufft|cupti|curand"
    r"|cusolver(?:Mg)?|cusparseLt|cusparse|nccl|nvJitLink|nvToolsExt"
    r"|nvrtc(?:-builtins)?|cufile|nvperf\w*)\.so")
CUDA_SONAME_PKG = {
    "cudart": "cuda-cudart", "cublas": "libcublas", "cublasLt": "libcublas",
    "cufft": "libcufft", "cufftw": "libcufft", "cupti": "cuda-cupti",
    "curand": "libcurand", "cusolver": "libcusolver", "cusolverMg": "libcusolver",
    "cusparse": "libcusparse", "cusparseLt": "cusparselt", "nccl": "nccl",
    "nvJitLink": "libnvjitlink", "nvToolsExt": "cuda-nvtx", "nvrtc": "cuda-nvrtc",
    "nvrtc-builtins": "cuda-nvrtc", "cufile": "libcufile",
}

# nvidia-* wheel pin -> conda-forge package (verified to exist with
# matching versions; -cuNN suffix stripped before lookup). 'cudnn' on
# conda-forge is a metapackage squatting on the 8.x name — the real
# library is 'libcudnn'. 'nvidia-nvshmem' maps to OUR OWN repack (no
# conda-forge nvshmem exists, verified 404).
NVIDIA_MAP = {
    "nvidia-cuda-runtime": "cuda-cudart",
    "nvidia-cublas": "libcublas",
    "nvidia-cudnn": "libcudnn",
    "nvidia-cusparse": "libcusparse",
    "nvidia-cufft": "libcufft",
    "nvidia-curand": "libcurand",
    "nvidia-cusolver": "libcusolver",
    "nvidia-nccl": "nccl",
    "nvidia-cuda-cupti": "cuda-cupti",
    "nvidia-cuda-nvrtc": "cuda-nvrtc",
    "nvidia-nvjitlink": "libnvjitlink",
    "nvidia-cufile": "libcufile",
    "nvidia-cusparselt": "cusparselt",
    "nvidia-nvtx": "cuda-nvtx",
    "nvidia-nvshmem": "nvidia-nvshmem",
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
    "importlib-metadata": "importlib-metadata",
    "packaging": "packaging",
    "cuda-bindings": "cuda-bindings",
}

# --- win-64 payload surgery ------------------------------------------------
# Vendored NVIDIA/OpenMP DLLs stripped from torch/lib: supplied instead by
# conda-forge packages whose win builds ship the identical DLL basenames
# into Library/bin (on torch's own search path via sys.exec_prefix). The
# libiomp* strip (runtime AND stubs — shipping a colliding copy of our own
# intel-openmp dependency's DLL is indefensible) pairs with the
# intel-openmp dep — one OpenMP runtime in the process, conda's.
# zlibwapi.dll is deliberately KEPT (tiny, and conda-forge cudnn does not
# ship it under that name).
WIN_STRIP_DLL = re.compile(
    r"^(cudart64|cublas64|cublasLt64|cudnn|cufft64|cufftw64|cupti64"
    r"|curand64|cusolver64|cusolverMg64|cusparse64|nvJitLink|nvrtc64"
    r"|nvrtc-builtins64|libiomp)", re.IGNORECASE)
# imported-basename -> conda-forge package. On Windows the basename IS
# the soname: a version that exists is NOT enough, the exact DLL name
# torch's PE import table demands must ship in the pinned window (CUPTI's
# basename changes every CUDA minor — cupti64_2025.1.1.dll vs _2025.2.1;
# the audit runs for every stripped basename on every flavour).
WIN_DLL_PKG = [
    (re.compile(r"^cudart64", re.I), "cuda-cudart"),
    (re.compile(r"^cublas(lt)?64", re.I), "libcublas"),
    (re.compile(r"^cudnn", re.I), "libcudnn"),
    (re.compile(r"^cufftw?64", re.I), "libcufft"),
    (re.compile(r"^cupti64", re.I), "cuda-cupti"),
    (re.compile(r"^curand64", re.I), "libcurand"),
    (re.compile(r"^cusolver(mg)?64", re.I), "libcusolver"),
    (re.compile(r"^cusparse64", re.I), "libcusparse"),
    (re.compile(r"^nvjitlink", re.I), "libnvjitlink"),
    (re.compile(r"^nvrtc", re.I), "cuda-nvrtc"),
    (re.compile(r"^libiomp", re.I), "intel-openmp"),
]
# Dead static archives (~2.7 GB): nothing consumes them — cpp_extension
# links exactly the torch six, and the cmake IMPORTED_IMPLIB checks cover
# only c10/c10_cuda/torch_cpu/torch_cuda/torch (+ an optional kineto
# lookup that warns, not errors). Strip by explicit list, never pattern.
WIN_STRIP_LIB = {
    "dnnl.lib", "libprotoc.lib", "libprotobuf.lib", "libprotobuf-lite.lib",
    "sleef.lib", "microkernels-prod.lib", "XNNPACK.lib", "fmt.lib",
    "fbgemm.lib", "pthreadpool.lib", "cpuinfo.lib", "libittnotify.lib",
    "asmjit.lib", "kineto.lib",
}
# kineto.lib is in the strip list above ONLY if absent from this keep set;
# cmake's unconditional append_torchlib_if_found(kineto) merely warns when
# it is missing, but keeping it (99 MB) silences that for consumers.
WIN_KEEP_LIB = {
    "c10.lib", "c10_cuda.lib", "torch_cpu.lib", "torch_cuda.lib",
    "torch.lib", "torch_python.lib", "_C.lib", "shm.lib",
    "caffe2_nvrtc.lib", "kineto.lib",
}
# conda-forge CUDA packages backing the stripped DLLs (win has no
# nccl/cufile/cusparselt/nvshmem). Lower bounds come from the linux
# wheel's METADATA pins for the same (version, flavour).
WIN_CUDA_PKGS = {
    "cuda-cudart", "libcublas", "libcudnn", "libcusparse", "libcufft",
    "libcurand", "libcusolver", "cuda-nvrtc", "libnvjitlink", "cuda-cupti",
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
# licenses, provenance, ABI floors (hostile-review items 2, 4, 13)
# --------------------------------------------------------------------------

_LICENSE_NAME = re.compile(
    r"(^|/)(AUTHORS|COPYING|COPYRIGHT|LICEN[CS]E|NOTICE|EULA|(THIRD|3RD)[-_]?PARTY)"
    r"[^/]*$", re.I)
_LICENSE_EXCLUDE = (".py", ".pyc", ".pyi", ".h", ".hpp", ".so", ".dll", ".lib",
                    ".cmake", ".c", ".cpp", ".cu", ".cuh", ".exe")


def wheel_license_files(wheel: Path) -> dict[str, bytes]:
    """Every license-ish file in a wheel, keyed by a flattened unique name.

    NVIDIA's redistribution terms permit shipping the runtime libraries ON
    CONDITION the license text accompanies them — an output package must
    never carry fewer license files than its source wheel."""
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(wheel) as zf:
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            n = zi.filename
            if n.lower().endswith(_LICENSE_EXCLUDE):
                continue
            if _LICENSE_NAME.search(n) or "/licenses/" in n.lower():
                out[n.replace("/", "_")] = zf.read(zi)
    return out


def license_gate(wheel: Path, pkg_name: str) -> dict[str, bytes]:
    lics = wheel_license_files(wheel)
    if not lics:
        sys.exit(f"{wheel.name}: NO license files found in the source wheel — "
                 f"refusing to publish {pkg_name} without knowing why")
    log(f"{pkg_name}: carrying {len(lics)} license file(s): {sorted(lics)[:4]}...")
    return lics


def provenance() -> dict:
    """CI build provenance for about.extra + the fragment (a published
    .conda must be traceable to the workflow run and commit that built it)."""
    out: dict[str, str] = {}
    rid = os.environ.get("GITHUB_RUN_ID")
    if rid:
        repo = os.environ.get("GITHUB_REPOSITORY", "Comfy-Forge/conda-torch")
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        out["run_id"] = rid
        out["run_url"] = f"{server}/{repo}/actions/runs/{rid}"
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        out["source_commit"] = sha
    return out


# GLIBCXX symbol version -> minimal GCC major (gcc.gnu.org abi.html)
_GLIBCXX_GCC = {
    "3.4.19": 5, "3.4.20": 5, "3.4.21": 5, "3.4.22": 6, "3.4.23": 7,
    "3.4.24": 7, "3.4.25": 8, "3.4.26": 9, "3.4.27": 9, "3.4.28": 10,
    "3.4.29": 11, "3.4.30": 12, "3.4.31": 13, "3.4.32": 13, "3.4.33": 14,
    "3.4.34": 15,
}


def elf_floors(roots: list[Path]) -> tuple[str | None, int | None]:
    """(max GLIBC_x.y, GCC major implied by max GLIBCXX) across every ELF
    under roots — measured from the binaries, never pasted. A pasted
    constant either excludes old distros for nothing (torch 2.4 needs only
    2.17) or silently under-declares the next wheel era."""
    max_glibc: V | None = None
    max_cxx: str | None = None
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.is_symlink() or not p.is_file():
                continue
            with open(p, "rb") as fh:
                if fh.read(4) != b"\x7fELF":
                    continue
            txt = subprocess.run(["readelf", "-V", str(p)],
                                 capture_output=True, text=True).stdout
            for m in re.finditer(r"GLIBC_(\d+\.\d+)\b", txt):
                v = V(m.group(1))
                if max_glibc is None or v > max_glibc:
                    max_glibc = v
            for m in re.finditer(r"GLIBCXX_(3\.4\.\d+)\b", txt):
                if max_cxx is None or V(m.group(1)) > V(max_cxx):
                    max_cxx = m.group(1)
    gcc = None
    if max_cxx is not None:
        gcc = _GLIBCXX_GCC.get(max_cxx)
        if gcc is None:
            log(f"WARNING: unknown GLIBCXX_{max_cxx}; assuming GCC 15 floor")
            gcc = 15
    return (max_glibc.s if max_glibc else None), gcc


def floor_deps(roots: list[Path]) -> list[str]:
    glibc, gcc = elf_floors(roots)
    deps = []
    if glibc:
        deps.append(f"__glibc >={glibc}")
    if gcc:
        deps += [f"libstdcxx >={gcc}", "libgcc"]
    log(f"measured ABI floors: glibc={glibc} gcc={gcc}")
    return deps


# --------------------------------------------------------------------------
# wheel acquisition
# --------------------------------------------------------------------------

def fetch(url: str, timeout: int, headers: dict | None = None):
    # download.pytorch.org 403s the default Python-urllib User-Agent
    h = {"User-Agent": "conda-torch/0.1 (+https://github.com/Comfy-Forge/conda-torch)"}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h)
    return urllib.request.urlopen(req, timeout=timeout)


def full_url(u: str) -> str:
    return u if u.startswith("http") else "https://download.pytorch.org/" + u.lstrip("/")


def wheel_url(version: str, flavour: str, py: str, subdir: str) -> tuple[str, str | None]:
    """Returns (url, sha256-from-index-anchor-or-None)."""
    cp = "cp" + py.replace(".", "")
    idx = f"https://download.pytorch.org/whl/{flavour}/torch/"
    html = fetch(idx, 60).read().decode()
    # some flavour-dir wheels carry no +cuNN local tag (old aarch64 CUDA
    # builds): the flavour is encoded only by the directory. Prefer the
    # tagged name; fall back to tag-less. version.py is verified against
    # the flavour after extraction either way.
    for ver_pat in (re.escape(version) + "%2B" + flavour, re.escape(version)):
        pat = re.compile(r'href="([^"]*torch-%s-%s-%s-%s)(?:#sha256=([0-9a-f]{64}))?"'
                         % (ver_pat, cp, cp, PLATFORMS[subdir]["wheel_re"]))
        m = pat.search(html)
        if m:
            return full_url(m.group(1)), m.group(2)
    sys.exit(f"no {subdir} wheel for torch {version}+{flavour} {cp} on {idx}")


def download(url: str, dest: Path, expect_sha: str | None = None) -> None:
    """Download with mandatory hash verification when a hash is known.

    A cached file is re-verified too: the hash protects against a poisoned
    CDN response, and the recorded provenance must never notarize whatever
    happened to arrive.
    """
    if not dest.exists():
        log(f"downloading {url}")
        tmp = dest.with_suffix(".part")
        with fetch(url, 120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 22)
        tmp.rename(dest)
        log(f"downloaded {dest.stat().st_size} bytes")
    else:
        log(f"wheel already present: {dest}")
    if expect_sha:
        got = sha256_file(dest)
        if got != expect_sha:
            dest.unlink()
            sys.exit(f"SHA256 MISMATCH for {dest.name}: expected {expect_sha}, "
                     f"got {got} — refusing to build from unverified bytes")


def grid_wheel_sha(version: str, flavour: str, subdir: str, py: str) -> str | None:
    """The wheel_sha256 grid.json recorded for this cell, if any."""
    grid = Path(__file__).resolve().parent.parent / "grid" / "grid.json"
    if not grid.exists():
        return None
    g = json.loads(grid.read_text())
    pynum = py.replace(".", "")
    for ent in g.get("entries", []):
        if (ent.get("torch_version") == version and ent.get("flavour") == flavour
                and ent.get("platform") == subdir):
            for r in ent.get("records", []):
                if r.get("python") == pynum and r.get("wheel_sha256"):
                    return r["wheel_sha256"]
    return None


class RangeFile(io.RawIOBase):
    """Minimal random-access HTTP reader for zipfile (readinto-based)."""

    def __init__(self, url: str):
        self.url = url
        with fetch(url, 60, {"Range": "bytes=0-0"}) as r:
            cr = r.headers.get("Content-Range", "")
        self._size = int(cr.rsplit("/", 1)[1])
        self._pos = 0

    def seek(self, off: int, whence: int = 0) -> int:
        self._pos = {0: off, 1: self._pos + off, 2: self._size + off}[whence]
        return self._pos

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        if self._pos >= self._size:
            return 0
        end = min(self._pos + len(b), self._size) - 1
        with fetch(self.url, 120, {"Range": f"bytes={self._pos}-{end}"}) as r:
            data = r.read()
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)


def remote_metadata(version: str, flavour: str, py: str, subdir: str) -> str:
    """Range-read METADATA out of a wheel without downloading it."""
    url, _ = wheel_url(version, flavour, py, subdir)
    zf = zipfile.ZipFile(io.BufferedReader(RangeFile(url), 256 * 1024))
    name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
    return zf.read(name).decode(errors="replace")


# --------------------------------------------------------------------------
# metadata translation
# --------------------------------------------------------------------------

class V:
    """Marker/spec comparison value: numeric-dotted versions compare as
    version tuples ('3.9' < '3.15'); everything else as plain strings.
    PEP 508 says version-y markers compare as versions — a lexical '3.9' <
    '3.15' would silently drop deps on old pythons."""

    def __init__(self, s: str):
        self.s = str(s)
        self.t = None
        if re.fullmatch(r"\d+(\.\d+)*", self.s):
            self.t = tuple(int(x) for x in self.s.split("."))

    def _k(self, other):
        if self.t is not None and other.t is not None:
            n = max(len(self.t), len(other.t))
            pad = lambda t: t + (0,) * (n - len(t))
            return pad(self.t), pad(other.t)
        return self.s, other.s

    def __eq__(self, o): a, b = self._k(o); return a == b
    def __ne__(self, o): a, b = self._k(o); return a != b
    def __lt__(self, o): a, b = self._k(o); return a < b
    def __le__(self, o): a, b = self._k(o); return a <= b
    def __gt__(self, o): a, b = self._k(o); return a > b
    def __ge__(self, o): a, b = self._k(o); return a >= b
    def __hash__(self): return hash(self.s)


def parse_requires(metadata_text: str, py: str, subdir: str) -> list[tuple[str, list[str], str]]:
    """Marker-evaluated Requires-Dist for the target platform.

    Returns [(normalized_name, extras, spec)] — extras matter for torch
    2.14+'s `cuda-toolkit[cublas,...]==13.2.1` dependency shape.
    """
    env = dict(PLATFORMS[subdir]["markers"])
    env.update({
        "python_version": py,
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
    })
    out = []
    for line in metadata_text.splitlines():
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
        m = re.match(r"^([A-Za-z0-9._-]+)\s*(\[[^\]]*\])?\s*(.*)$", req)
        if not m:
            sys.exit(f"unparseable Requires-Dist: {line}")
        name = m.group(1).lower().replace("_", "-").replace(".", "-")
        extras = [e.strip() for e in m.group(2)[1:-1].split(",")] if m.group(2) else []
        spec = m.group(3).strip().strip("()").strip()
        out.append((name, extras, spec))
    return out


def eval_marker(marker: str, env: dict[str, str]) -> bool:
    """Tiny PEP 508 marker evaluator: comparisons, and/or/parens only.
    Values are wrapped in V() so python_version compares numerically
    ('3.9' < '3.15' must be True — a lexical compare drops deps)."""
    # wrap ALL string literals first (before any V( exists), then names
    expr = marker.replace('"', "'")
    expr = re.sub(r"'([^']*)'", r"V('\1')", expr)
    for k, v in env.items():
        expr = re.sub(rf"\b{k}\b", f"V({v!r})", expr)
    leftover = re.sub(r"\bV\('[^']*'\)", "", expr)
    if re.search(r"[A-Za-z_]{2,}", re.sub(r"\b(and|or|not|in)\b", "", leftover)):
        sys.exit(f"marker has unknown variables, refusing to guess: {marker}")
    return bool(eval(expr, {"__builtins__": {}, "V": V}))


def pep_clauses(spec: str) -> list[tuple[str, str]]:
    """'<13,>=12.6.85' -> [('<','13'), ('>=','12.6.85')]; '~=' expanded."""
    clauses = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        m = re.match(r"^(~=|==|!=|<=|>=|<|>)\s*([0-9][0-9a-zA-Z.*+!-]*)$", part)
        if not m:
            sys.exit(f"unsupported version clause {part!r} in {spec!r}")
        op, ver = m.group(1), m.group(2)
        if op == "~=":
            base = ver.split(".")
            if len(base) < 2:
                sys.exit(f"~= needs at least two components: {part}")
            upper = ".".join(base[:-2] + [str(int(base[-2]) + 1)])
            clauses += [(">=", ver), ("<", upper + ".0a0")]
        else:
            clauses.append((op, ver))
    return clauses


def conda_spec(spec: str) -> str:
    """PEP 440 specifier -> conda version spec (same ops, ~= expanded)."""
    if not spec:
        return ""
    return ",".join(op + ver for op, ver in pep_clauses(spec))


def spec_floor(spec: str) -> str | None:
    """The == or >= version in a specifier, if any."""
    for op, ver in pep_clauses(spec):
        if op in ("==", ">="):
            return ver.rstrip(".*")
    return None


def nvidia_key(name: str) -> str | None:
    base = re.sub(r"-cu\d+$", "", name)
    return base if base in NVIDIA_MAP else None


# torch 2.14+ pins `cuda-toolkit[extras]==X.Y.Z` instead of individual
# nvidia-* wheels. Each extra maps to the conda-forge component; versions
# ride the flavour's cuda-version window (conda-forge CUDA libs constrain
# their own cuda-version per release line).
CUDA_TOOLKIT_EXTRAS = {
    "cublas": "libcublas", "cudart": "cuda-cudart", "cufft": "libcufft",
    "cufile": "libcufile", "cupti": "cuda-cupti", "curand": "libcurand",
    "cusolver": "libcusolver", "cusparse": "libcusparse",
    "nvjitlink": "libnvjitlink", "nvrtc": "cuda-nvrtc", "nvtx": "cuda-nvtx",
}
# python-level deps that live in the *pytorch* package but are CUDA-family
# and routinely missing from conda-forge for a given line — candidates for
# a PyPI side-repack when the translated range is unsatisfiable there.
SIDE_REPACK_OK = {"triton": "triton", "cuda-bindings": "cuda-bindings"}


def translate_deps(requires: list[tuple[str, list[str], str]]) -> list[str]:
    """PyPI requirement list -> conda depends for the *pytorch* package.

    nvidia-*/cuda-toolkit deps are handled separately (they belong to
    libtorch).
    """
    deps = []
    for name, extras, spec in requires:
        if nvidia_key(name) or name == "cuda-toolkit":
            continue  # carried by libtorch
        if name in SIDE_REPACK_OK:
            deps.append(f"{SIDE_REPACK_OK[name]} {conda_spec(spec)}".strip())
            continue
        if name not in PY_DEP_MAP:
            sys.exit(f"unmapped PyPI dependency {name!r} ({spec!r}); add it to PY_DEP_MAP")
        conda = PY_DEP_MAP[name]
        spec = conda_spec(spec)
        if conda == "setuptools" and "<" not in spec:
            # runtime dep on 3.12+; new setuptools removed pkg_resources
            spec = (spec + "," if spec else "") + "<82"
        deps.append(f"{conda} {spec}".strip())
    return deps


def cuda_bound(name: str, spec: str) -> str:
    """Exact pin -> lower bound + next-major cap (CUDA minor compat).
    torch 2.14 moved some pins to ranges — translate those faithfully."""
    clauses = pep_clauses(spec)
    if len(clauses) == 1 and clauses[0][0] == "==":
        v = clauses[0][1]
        return f">={v},<{int(v.split('.')[0]) + 1}.0a0"
    return ",".join(op + ver for op, ver in clauses)


def cuda_dep_map(requires: list[tuple[str, list[str], str]],
                 flavour: str) -> dict[str, str]:
    """All CUDA-side deps as {conda_name: conda_version_spec_or_''}."""
    out: dict[str, str] = {}
    for name, extras, spec in requires:
        if name == "cuda-toolkit":
            pin = spec_floor(spec)
            want = f"{flavour[2:-1]}.{flavour[-1]}"
            if pin and not pin.startswith(want):
                sys.exit(f"cuda-toolkit pin {pin} does not match flavour {flavour}")
            for extra in extras:
                if extra not in CUDA_TOOLKIT_EXTRAS:
                    sys.exit(f"unknown cuda-toolkit extra {extra!r}; extend the map")
                out[CUDA_TOOLKIT_EXTRAS[extra]] = ""
            continue
        key = nvidia_key(name)
        if key:
            out[NVIDIA_MAP[key]] = cuda_bound(name, spec)
    unmapped = {n for n, _, _ in requires if n.startswith("nvidia-") and not nvidia_key(n)}
    if unmapped:
        sys.exit(f"nvidia deps without a conda mapping: {unmapped}")
    return out


def cuda_deps(requires: list[tuple[str, list[str], str]], flavour: str) -> list[str]:
    """CUDA deps for the *libtorch* package, bounded by the flavour window."""
    cuda_minor = f"{flavour[2:-1]}.{flavour[-1]}"  # cu128 -> 12.8
    deps = ["__cuda", f"cuda-version >={cuda_minor},<{int(flavour[2:-1]) + 1}"]
    for pkg, bound in cuda_dep_map(requires, flavour).items():
        deps.append(f"{pkg} {bound}".strip())
    return deps


def win_cuda_deps(version: str, flavour: str, py: str) -> list[str]:
    """win-64 libtorch CUDA deps.

    The Windows wheel declares ZERO nvidia requirements (everything is
    vendored), so the dep set is borrowed from the linux wheel of the
    same (version, flavour) — the same upstream build set — filtered to
    the packages that actually back a stripped DLL.
    """
    cuda_minor = f"{flavour[2:-1]}.{flavour[-1]}"
    deps = ["__cuda", f"cuda-version >={cuda_minor},<{int(flavour[2:-1]) + 1}"]
    log("range-reading linux METADATA for CUDA lower bounds...")
    linux_req = parse_requires(remote_metadata(version, flavour, py, "linux-64"),
                               py, "linux-64")
    cmap = cuda_dep_map(linux_req, flavour)
    found = set()
    for pkg, bound in cmap.items():
        if pkg in WIN_CUDA_PKGS:
            deps.append(f"{pkg} {bound}".strip())
            found.add(pkg)
    missing = WIN_CUDA_PKGS - found
    if missing:
        sys.exit(f"linux METADATA gave no pins for {missing}; cannot bound win deps")
    return deps


_CF_FILES: dict[str, list] = {}


def cf_files(pkg: str) -> list:
    if pkg not in _CF_FILES:
        for attempt in range(5):
            try:
                _CF_FILES[pkg] = json.load(fetch(
                    f"https://api.anaconda.org/package/conda-forge/{pkg}/files", 60))
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:  # package genuinely absent from conda-forge
                    _CF_FILES[pkg] = []
                    break
                if attempt == 4:
                    raise  # a transient error must NOT read as "unsatisfiable"
                time.sleep(10 * (attempt + 1))
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))
    return _CF_FILES[pkg]


def cf_versions(pkg: str, subdir: str) -> list[str]:
    """All conda-forge versions of pkg for subdir (anaconda.org API)."""
    return sorted({f["version"] for f in cf_files(pkg)
                   if f.get("attrs", {}).get("subdir") == subdir})


def cf_satisfiable(pkg: str, spec: str, subdir: str) -> bool:
    versions = cf_versions(pkg, subdir)
    ops = {"==": V.__eq__, "!=": V.__ne__, "<": V.__lt__, "<=": V.__le__,
           ">": V.__gt__, ">=": V.__ge__}
    clauses = pep_clauses(spec) if spec else []
    # strip .* and alpha caps (<10.0a0) so bounds compare numerically
    clean = lambda ver: re.match(r"[0-9.]*", ver).group(0).rstrip(".")
    return any(all(ops[op](V(v), V(clean(ver))) for op, ver in clauses)
               for v in versions)


_SPEC_OPS = {"==": V.__eq__, "!=": V.__ne__, "<": V.__lt__, "<=": V.__le__,
             ">": V.__gt__, ">=": V.__ge__}
_clean_ver = lambda ver: re.match(r"[0-9.]*", ver).group(0).rstrip(".")


def _spec_admits(version: str, spec: str) -> bool:
    if not spec:
        return True
    return all(_SPEC_OPS[op](V(version), V(_clean_ver(v))) for op, v in pep_clauses(spec))


def _windows_overlap(a: str, b: str) -> bool:
    """Both are conda specs like '>=12.8,<13'."""
    def bounds(s):
        lo, hi = None, None
        for op, v in pep_clauses(s):
            if op == ">=":
                lo = _clean_ver(v)
            elif op == "<":
                hi = _clean_ver(v)
        return lo, hi
    alo, ahi = bounds(a)
    blo, bhi = bounds(b)
    lo = max(V(alo or "0"), V(blo or "0"))
    his = [V(x) for x in (ahi, bhi) if x]
    return not his or lo < min(his)


def cf_satisfiable_in_window(pkg: str, spec: str, subdir: str, window: str) -> bool:
    """A version satisfying spec is not enough: the BUILD's own
    cuda-version constraint must intersect the entry's window (conda-forge
    ships e.g. libcudnn 9.19+ only as cuda13 builds — UNSAT inside a
    '>=12.8,<13' window even though the version exists)."""
    for f in cf_files(pkg):
        a = f.get("attrs", {})
        if a.get("subdir") != subdir or not _spec_admits(f["version"], spec):
            continue
        cv = next((d.split(None, 1)[1] for d in a.get("depends", [])
                   if d.startswith("cuda-version ")), None)
        if cv is None or _windows_overlap(cv, window):
            return True
    return False


# conda-forge CUDA lib names that have a PyPI-wheel fallback when no
# conda-forge build fits the entry's cuda-version window. The fallback
# publishes under the SAME conda-forge name (libcudnn): a second name
# with the same lib paths made an ordinary solve install both copies
# with no mutex and clobber torch's cudnn (proven env-breaker). Carrying
# the cf name obliges the completeness rule — the channel mirrors cf's
# libcudnn line alongside (the triton doctrine).
NVIDIA_FALLBACK = {"libcudnn": "libcudnn"}
_CF_CUDA_PKGS = set(NVIDIA_MAP.values()) | set(CUDA_TOOLKIT_EXTRAS.values())


def fix_window_unsat(deps: list[str], requires, flavour: str, subdir: str,
                     outdir: Path, work: Path) -> list[str]:
    """Replace conda-forge CUDA deps that no build can satisfy inside the
    flavour window with a side-repacked nvidia-* package."""
    window = f">={flavour[2:-1]}.{flavour[-1]},<{int(flavour[2:-1]) + 1}"
    out = []
    for d in deps:
        pkg, _, spec = d.partition(" ")
        if pkg not in _CF_CUDA_PKGS or pkg == "nvidia-nvshmem" \
                or cf_satisfiable_in_window(pkg, spec, subdir, window):
            out.append(d)
            continue
        if pkg not in NVIDIA_FALLBACK:
            sys.exit(f"{pkg} {spec!r} has no conda-forge build inside cuda-version "
                     f"{window} on {subdir} and no PyPI fallback is defined: grid hole")
        if subdir == "win-64":
            sys.exit(f"{pkg} {spec!r} window-UNSAT on win-64: extend the fallback "
                     "for Windows (Library/bin DLL placement) before publishing")
        req = next(((n, s) for n, _, s in requires
                    if nvidia_key(n) and NVIDIA_MAP[nvidia_key(n)] == pkg), None)
        if not req:
            sys.exit(f"cannot find the wheel pin behind {pkg} to side-repack")
        pin = spec_floor(req[1])
        log(f"{pkg} {spec!r} UNSAT inside cuda-version {window} on {subdir}: "
            f"side-repacking {req[0]} {pin} as {NVIDIA_FALLBACK[pkg]}")
        build_nvidia_lib(req[0], pin, subdir, outdir, work, NVIDIA_FALLBACK[pkg])
        out.append(f"{NVIDIA_FALLBACK[pkg]} >={pin},<{int(pin.split('.')[0]) + 1}.0a0")
    return out


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


def patch_rpath(target: Path, rpath: str) -> None:
    """REPLACE the RPATH wholesale (--set-rpath + --force-rpath).

    Never --add-rpath: appending keeps the wheel's entries in front (the
    pip-hijack and /usr/local/cuda hazards, see ADDED_RPATH comment)."""
    for entry in rpath.split(":"):
        if not entry or not entry.startswith("$ORIGIN"):
            sys.exit(f"refusing non-$ORIGIN rpath entry {entry!r} for {target.name}")
    run("patchelf", "--force-rpath", "--set-rpath", rpath, str(target))
    got = subprocess.run(["patchelf", "--print-rpath", str(target)],
                         capture_output=True, text=True, check=True).stdout.strip()
    if got != rpath:
        sys.exit(f"rpath set did not stick on {target.name}: {got!r}")


def scrub_site_elfs(pt_stage: Path, sp: Path) -> None:
    """Every remaining ELF in site-packages gets a clean RPATH: $ORIGIN
    plus the relative hop to $PREFIX/lib. Kills the wheel's dormant
    nvidia/* entries on torch/_C and upstream's /lib/intel64 +
    /usr/local/cuda junk on functorch/_C (which --force-rpath would
    otherwise have PROMOTED from RUNPATH to RPATH)."""
    lib = pt_stage / "lib"
    n = 0
    for p in sorted(sp.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        with open(p, "rb") as fh:
            if fh.read(4) != b"\x7fELF":
                continue
        rel = os.path.relpath(lib, p.parent)
        patch_rpath(p, f"$ORIGIN:$ORIGIN/{rel}")
        n += 1
    log(f"scrubbed RPATHs on {n} site-packages ELFs")


def extract_wheel(wheel: Path, sp: Path) -> None:
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


def scrub_dist_info(sp: Path) -> Path:
    """pip must never think it owns this. Returns the dist-info dir."""
    dist_info = next(sp.glob("torch-*.dist-info"))
    (dist_info / "RECORD").unlink()
    (dist_info / "INSTALLER").write_bytes(b"conda")  # exactly 5 bytes, no \n
    for junk in ("direct_url.json", "RECORD.jws", "REQUESTED"):
        (dist_info / junk).unlink(missing_ok=True)
    return dist_info


def byte_compile(sp: Path, sp_rel: str) -> None:
    # checked-hash .pyc: immune to the mtimes conda extraction produces.
    # ddir makes co_filename ENV-RELATIVE — without it every traceback,
    # pdb session and inspect.getsource on a user machine points at the
    # phantom CI staging dir with no source lines.
    log("byte-compiling (env-relative co_filename)...")
    for top in ("torch", "torchgen", "functorch"):
        if (sp / top).is_dir():
            # str(), not Path: 3.9-3.11 compileall calls stripdir.split()
            compileall.compile_dir(str(sp / top), quiet=2, workers=0,
                                   ddir=f"{sp_rel}/{top}",
                                   invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
    pyc_check(sp, sp_rel)


def pyc_check(sp: Path, sp_rel: str) -> None:
    import marshal
    pyc = next((sp / "torch" / "__pycache__").glob("__init__.*.pyc"), None)
    if pyc is None:
        sys.exit("byte-compile produced no torch/__pycache__/__init__ .pyc")
    code = marshal.loads(pyc.read_bytes()[16:])
    if not code.co_filename.startswith(sp_rel):
        sys.exit(f".pyc co_filename {code.co_filename!r} is not env-relative "
                 f"(wanted prefix {sp_rel!r})")
    log(f".pyc co_filename verified: {code.co_filename}")


def win_launcher_stub() -> bytes:
    """setuptools' cli-64.exe: a plain launcher that runs its sibling
    <name>-script.py, reading the script's own (prefix-rewritten) shebang.
    Debian-patched installs strip the .exe files — fall back to pulling
    the wheel straight from PyPI."""
    try:
        m = __import__("setuptools")
        cand = Path(m.__file__).parent / "cli-64.exe"
        if cand.exists():
            return cand.read_bytes()
    except ImportError:
        pass
    log("installed setuptools has no cli-64.exe (Debian strip?); fetching the wheel")
    url, sha = pypi_wheel_url("setuptools", "80.9.0", ["py3-none-any"])
    dest = Path("/tmp/_setuptools_launcher.whl")
    download(url, dest, sha)
    with zipfile.ZipFile(dest) as zf:
        return zf.read("setuptools/cli-64.exe")


def win_launcher_pair(scripts: Path, prefix_files: dict[str, str]) -> None:
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "torchrun.exe").write_bytes(win_launcher_stub())
    (scripts / "torchrun-script.py").write_text(
        "#!" + PLACEHOLDER + "/python.exe\n" + WIN_MAIN.decode())
    prefix_files["Scripts/torchrun-script.py"] = "text"


def pe_imports(path: Path) -> set[str]:
    """Imported DLL basenames from a PE's import + delay-import tables."""
    import pefile
    pe = pefile.PE(str(path), fast_load=True)
    pe.parse_data_directories(directories=[
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]])
    out = set()
    for attr in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
        for entry in getattr(pe, attr, None) or []:
            out.add(entry.dll.decode(errors="replace"))
    pe.close()
    return out


def cf_dll_basenames(pkg: str, subdir: str, version: str) -> set[str]:
    """Library/bin DLL basenames shipped by the newest conda-forge build of
    pkg==version on subdir (paths.json range-read from the artifact)."""
    cands = sorted((f for f in cf_files(pkg)
                    if f.get("attrs", {}).get("subdir") == subdir
                    and f["version"] == version and f["basename"].endswith(".conda")),
                   key=lambda f: f.get("attrs", {}).get("build_number", 0),
                   reverse=True)
    for cand in cands:
        # the files API lists builds that were patched OUT of repodata and
        # can 404 on the CDN — skip ghosts, try the next build
        url = "https://conda.anaconda.org/conda-forge/" + cand["basename"]
        try:
            zf = zipfile.ZipFile(io.BufferedReader(RangeFile(url), 256 * 1024))
            info = next(n for n in zf.namelist() if n.startswith("info-"))
            raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(zf.read(info))).read()
        except (urllib.error.HTTPError, urllib.error.URLError, StopIteration) as e:
            log(f"cf_dll_basenames: skipping ghost {cand['basename']} ({e})")
            continue
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            paths = json.load(tf.extractfile("info/paths.json"))
        return {PurePosixPath(p["_path"]).name.lower() for p in paths["paths"]
                if p["_path"].lower().startswith("library/bin/")
                and p["_path"].lower().endswith(".dll")}
    return set()


def win_basename_audit(imports: set[str], stripped: list[str],
                       deps: list[str], flavour: str) -> list[str]:
    """Narrow win CUDA dep bounds so every needed DLL BASENAME provably
    ships inside the pinned window. A satisfiable version is not enough:
    cupti's basename changes every CUDA minor, so the wide window installs
    a package that does not contain the name torch imports and the eager
    loader 'tolerates' the miss into `DLL load failed`."""
    strippedset = {s.lower() for s in stripped}
    needed: dict[str, set[str]] = {}
    for imp in sorted(imports):
        if imp.lower() not in strippedset:
            continue
        for pat, pkg in WIN_DLL_PKG:
            if pat.match(imp):
                needed.setdefault(pkg, set()).add(imp.lower())
                break
        else:
            sys.exit(f"stripped DLL {imp} is imported by torch but has no "
                     "package mapping; extend WIN_DLL_PKG")
    out = []
    for d in deps:
        pkg, _, spec = d.partition(" ")
        if pkg not in needed or pkg == "intel-openmp":
            out.append(d)
            continue
        want = needed[pkg]
        lo = next((v for op, v in pep_clauses(spec) if op == ">="), None)
        hi = next((v for op, v in pep_clauses(spec) if op == "<"), None)
        vers = sorted((v for v in cf_versions(pkg, "win-64")
                       if _spec_admits(v, spec)), key=V, reverse=True)
        if not vers:
            sys.exit(f"{pkg} {spec!r}: no win-64 conda-forge version in window at all")
        ok = [v for v in vers if want <= cf_dll_basenames(pkg, "win-64", v)]
        if not ok:
            sys.exit(f"{pkg}: NO conda-forge win-64 build in {spec!r} ships "
                     f"{sorted(want)} — unfixable basename gap, report as hole")
        # Admit exactly the versions PROVEN to ship the basename and nothing
        # newer. Two earlier heuristics were both unsound: keeping the wide
        # window when every version admitted TODAY ships the name (cu129 had
        # a single cuda-cupti build, so the wide <13.0a0 window survived and
        # the first future build with a different basename would win the
        # solve), and bounding at the next MINOR (<12.9) — CUPTI's basename
        # third component also moves on toolkit PATCH releases
        # (cupti64_2025.1.1 -> 2025.1.2), which a minor-level bound admits.
        # A future release is re-admitted only once a rebuild's audit proves
        # it ships the name; cupti is a hard-pinned profiler lib, so losing
        # automatic patch pickup is the intended trade.
        lo2, hi2 = min(ok, key=V), max(ok, key=V)
        newd = f"{pkg} >={max(lo or '0', lo2, key=V)},<={hi2}"
        if newd != d:
            log(f"basename audit: {d!r} -> {newd!r} ({ok} ship {sorted(want)})")
        out.append(newd)
    for pkg in needed:
        if not any(d.split(" ", 1)[0] == pkg for d in deps):
            sys.exit(f"torch imports {sorted(needed[pkg])} but {pkg} is not in deps")
    return out


# --------------------------------------------------------------------------
# per-platform splits: populate lt_stage/pt_stage from the extracted wheel
# --------------------------------------------------------------------------

def split_linux(sp: Path, pt_stage: Path, lt_stage: Path, subdir: str) -> dict[str, str]:
    """One split for both linux subdirs. The lib set is enumerated from the
    wheel (torch >=2.9 grew libtorch_nvshmem on x86 too; old versions have
    fewer libs), never from a static list."""
    if (sp / "torch.libs").exists():
        sys.exit("torch.libs/ present (auditwheel-mangled sonames): this is the "
                 "CPU-wheel layout, which this pipeline does not target. The CUDA "
                 "wheels vendor in torch/lib with stock sonames.")
    tlib = sp / "torch" / "lib"
    (lt_stage / "lib" / VENDOR_DIR).mkdir(parents=True)

    big, vendored, stripped_cuda = [], [], []
    # regular .so files only: 2.9.0 aarch64 ships stray DIRECTORIES
    # (libshm/, libshm_windows/) inside torch/lib — those stay put
    for p in sorted(tlib.iterdir()):
        so = p.name
        if not p.is_file() or ".so" not in so:
            continue
        if so == "libtorch_python.so" or so.startswith("libgomp"):
            continue
        if CUDA_VENDORED.match(so):
            stripped_cuda.append(so)
            p.unlink()
            continue
        (vendored if VENDOR_PRIVATE.match(so) else big).append(so)
    log(f"{subdir} split: {len(big)} big libs, {len(vendored)} vendored ({vendored}), "
        f"{len(stripped_cuda)} vendored-CUDA stripped ({stripped_cuda})")

    for so in big:
        shutil.move(tlib / so, lt_stage / "lib" / so)
        make_symlink(tlib / so, f"../../../../{so}")
    for so in vendored:
        shutil.move(tlib / so, lt_stage / "lib" / VENDOR_DIR / so)
        make_symlink(tlib / so, f"../../../../{VENDOR_DIR}/{so}")
    gomp = list(tlib.glob("libgomp*"))
    if gomp:
        for g in gomp:
            g.unlink()

    (pt_stage / "lib").mkdir(exist_ok=True)
    shutil.move(tlib / "libtorch_python.so", pt_stage / "lib" / "libtorch_python.so")
    make_symlink(tlib / "libtorch_python.so", "../../../../libtorch_python.so")

    move_headers_cmake(sp, lt_stage, ups=5)

    # Big libs may be opened directly ($ORIGIN = $PREFIX/lib) or through
    # the torch/lib symlink ($ORIGIN = torch/lib — glibc expands from the
    # opening path, not the realpath), so both spellings of both targets
    # are needed. Every big lib gets a wholesale --set-rpath with
    # --force-rpath (a RUNPATH would lose to LD_LIBRARY_PATH); patching
    # non-CUDA libs too is harmless and keeps the two linux subdirs
    # identical.
    full = ":".join([
        "$ORIGIN",
        ADDED_RPATH,
        f"$ORIGIN/{VENDOR_DIR}",
        f"$ORIGIN/../../../../{VENDOR_DIR}",
    ])
    log("setting clean RPATHs (all big libs + vendored set, --force-rpath)...")
    for so in big:
        patch_rpath(lt_stage / "lib" / so, full)
    patch_rpath(pt_stage / "lib" / "libtorch_python.so", full)
    for so in vendored:
        patch_rpath(lt_stage / "lib" / VENDOR_DIR / so, "$ORIGIN:$ORIGIN/..")
    scrub_site_elfs(pt_stage, sp)
    return {}, stripped_cuda


def vendored_cuda_dep_list(stripped: list[str], version: str, flavour: str,
                           py: str, subdir: str) -> list[str]:
    """conda deps replacing stripped vendored-CUDA libs (old aarch64 wheels).

    Bounds are borrowed from the x86 wheel of the same (version, flavour)
    where it declares pins; components without a borrowed pin ride the
    flavour's cuda-version window unversioned. Every mapped package is
    checked to exist on conda-forge for this subdir — a missing one means
    the cell must be reported as a hole, not published broken."""
    pkgs = set()
    for so in stripped:
        base = CUDA_VENDORED.match(so).group(1)
        if base.startswith("cudnn"):
            base = "cudnn"
        pkg = CUDA_SONAME_PKG.get(base) or ("libcudnn" if base == "cudnn" else None)
        if not pkg:
            sys.exit(f"stripped vendored CUDA lib {so} has no conda mapping; extend CUDA_SONAME_PKG")
        pkgs.add(pkg)
    log("borrowing CUDA bounds from the linux-64 wheel METADATA...")
    borrowed = cuda_dep_map(parse_requires(
        remote_metadata(version, flavour, py, "linux-64"), py, "linux-64"), flavour)
    deps = []
    for pkg in sorted(pkgs):
        bound = borrowed.get(pkg, "")
        if not cf_satisfiable(pkg, bound, subdir):
            sys.exit(f"conda-forge cannot satisfy {pkg} {bound!r} on {subdir}: "
                     "this cell is a grid hole, not publishable with honest metadata")
        deps.append(f"{pkg} {bound}".strip())
    return deps


def move_headers_cmake(sp: Path, lt_stage: Path, ups: int) -> None:
    """linux: real headers/cmake to $PREFIX, dir-symlinks back.

    pybind11 stays REAL in the package: moving it would clobber
    $PREFIX/include/pybind11 from the pybind11 package.
    """
    up = "../" * ups
    tinc = sp / "torch" / "include"
    (lt_stage / "include").mkdir()
    for entry in sorted(tinc.iterdir()):
        if entry.name not in TORCH_OWN_HEADERS:
            continue  # third-party trees stay real in-package (clobber hazard)
        shutil.move(entry, lt_stage / "include" / entry.name)
        make_symlink(tinc / entry.name, f"{up}include/{entry.name}")

    tcmake = sp / "torch" / "share" / "cmake"
    sed_cmake(tcmake)
    (lt_stage / "share").mkdir()
    shutil.move(tcmake, lt_stage / "share" / "cmake")
    # link lives at torch/share/cmake, resolved from torch/share: 5 ups to $PREFIX
    make_symlink(tcmake, f"{up}share/cmake")

    # torch/bin stays fully real inside pytorch: torch_shm_manager MUST
    # exist at $SP_DIR/torch/bin/ (import hard-fails otherwise) and protoc
    # must NOT go to $PREFIX/bin (it would clobber libprotobuf's).


def split_win64(sp: Path, pt_stage: Path, lt_stage: Path) -> tuple[dict[str, str], set[str], list[str]]:
    """Windows: no relocation at all. Lib/site-packages is python-version-
    independent, so libtorch owns the big python-independent files AT
    their wheel paths under torch/, and pytorch owns the rest. Strips:
    vendored CUDA DLLs (conda-forge supplies identical basenames in
    Library/bin, already on torch's search path), libiomp*/stubs
    (intel-openmp dep instead), and the dead .lib archives.

    Returns (prefix_files, union of PE-imported basenames of torch's own
    kept DLLs, stripped DLL names) for the basename audit.
    """
    tlib = sp / "torch" / "lib"
    # parse imports BEFORE stripping: the import table of the KEPT DLLs
    # names exactly the basenames conda-forge must supply
    imports: set[str] = set()
    for p in sorted(tlib.glob("*.dll")):
        if not WIN_STRIP_DLL.match(p.name):
            imports |= pe_imports(p)
    stripped_dll, stripped_lib, kept_dll = [], [], []
    for p in sorted(tlib.iterdir()):
        if p.suffix.lower() == ".dll":
            if WIN_STRIP_DLL.match(p.name):
                stripped_dll.append(p.name)
                p.unlink()
            else:
                kept_dll.append(p.name)
        elif p.suffix.lower() == ".lib":
            if p.name in WIN_STRIP_LIB and p.name not in WIN_KEEP_LIB:
                stripped_lib.append(p.name)
                p.unlink()
            elif p.name not in WIN_KEEP_LIB:
                kept_dll.append(p.name)  # unknown survivor: keep, python-independent
    log(f"win strip: {len(stripped_dll)} DLLs ({stripped_dll}); "
        f"{len(stripped_lib)} .lib ({stripped_lib}); kept {kept_dll}")
    if "cudart64_12.dll" in kept_dll or not any(d.startswith("torch_cuda") for d in kept_dll):
        sys.exit("win strip sanity check failed")

    sed_cmake(sp / "torch" / "share" / "cmake")

    # libtorch ownership: big DLLs + import libs (minus the per-python
    # pair) + headers (minus pybind11) + cmake, at wheel paths.
    sproot = "Lib/site-packages"
    for p in sorted(tlib.iterdir()):
        if p.name in ("torch_python.dll", "torch_python.lib", "_C.lib"):
            continue  # per-python, stays in pytorch
        rel = f"{sproot}/torch/lib/{p.name}"
        (lt_stage / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(p, lt_stage / rel)
    tinc = sp / "torch" / "include"
    for entry in sorted(tinc.iterdir()):
        if entry.name == "pybind11":
            continue
        rel = f"{sproot}/torch/include/{entry.name}"
        (lt_stage / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(entry, lt_stage / rel)
    rel = f"{sproot}/torch/share/cmake"
    (lt_stage / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(sp / "torch" / "share" / "cmake", lt_stage / rel)

    # entry point: launcher + registered text script (see WIN_MAIN comment)
    prefix_files: dict[str, str] = {}
    win_launcher_pair(pt_stage / "Scripts", prefix_files)
    return prefix_files, imports, stripped_dll


# --------------------------------------------------------------------------
# nvshmem repack (linux; torch >= 2.9 needs it on x86 too, and conda-forge
# has no nvshmem at all — verified 404)
# --------------------------------------------------------------------------

# sonames a repacked nvshmem lib may legitimately NEED without a mapping
NVSHMEM_OK_NEEDED = re.compile(
    r"^(libc|libm|libdl|librt|libpthread|ld-linux|libgcc_s|libstdc\+\+"
    r"|libcudart|libcuda|libnvidia-ml|libnvshmem)")


def pypi_wheel_url(pypi_name: str, version: str, want: list[str]) -> tuple[str, str]:
    """(url, sha256) of the first wheel for pypi_name==version matching all
    `want` substrings (falling back to a py3-none-any wheel). PyPI's JSON
    API publishes digests for every file — side-repacks are always
    hash-verified."""
    api = json.load(fetch(f"https://pypi.org/pypi/{pypi_name}/{version}/json", 60))
    wheels = [f for f in api["urls"] if f["filename"].endswith(".whl")]
    for f in wheels:
        if all(w in f["filename"] for w in want):
            return f["url"], f["digests"]["sha256"]
    for f in wheels:
        if "py3-none-any" in f["filename"]:
            return f["url"], f["digests"]["sha256"]
    sys.exit(f"no wheel for {pypi_name} {version} matching {want} on PyPI "
             f"(have: {[f['filename'] for f in wheels]})")


def build_nvshmem(pypi_name: str, pin_version: str, subdir: str, py: str,
                  outdir: Path, work: Path, build_number: int) -> tuple[str, Path | None]:
    """Repack nvidia-nvshmem-cuNN from PyPI into $PREFIX/lib.

    Returns (conda dep string, built .conda path).
    """
    fam = int(re.search(r"-cu(\d+)$", pypi_name).group(1))  # 12 or 13
    plat = "aarch64" if subdir == "linux-aarch64" else "x86_64"
    url, whl_sha = pypi_wheel_url(pypi_name, pin_version, [plat])
    wheel = work / url.rsplit("/", 1)[1]
    download(url, wheel, whl_sha)
    lics = license_gate(wheel, "nvidia-nvshmem")

    stage = work / "nvshmem_stage"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "lib").mkdir(parents=True)
    zf = zipfile.ZipFile(wheel)
    n_libs = 0
    for zi in zf.infolist():
        parts = PurePosixPath(zi.filename).parts
        # nvidia/nvshmem/lib/** -> $PREFIX/lib (torch's rpath finds it via $ORIGIN)
        if len(parts) >= 4 and parts[:3] == ("nvidia", "nvshmem", "lib"):
            dest = stage / "lib" / Path(*parts[3:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zi) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            n_libs += 1
    zf.close()
    if n_libs == 0:
        sys.exit("nvshmem wheel had no nvidia/nvshmem/lib payload")

    dropped = []
    for so in sorted((stage / "lib").rglob("*.so*")):
        with open(so, "rb") as fh:
            magic = fh.read(4)
        if not (so.is_file() and magic == b"\x7fELF"):
            continue
        run("patchelf", "--force-rpath", "--set-rpath", "$ORIGIN", str(so))
        needed = subprocess.run(["patchelf", "--print-needed", str(so)],
                                capture_output=True, text=True, check=True).stdout.split()
        outside = [d for d in needed if not NVSHMEM_OK_NEEDED.match(d)]
        if outside:
            # Optional dlopen'd transport/bootstrap plugins (UCX, ibverbs,
            # libfabric...) would be unloadable without dragging the whole
            # HPC network stack into every env. torch only hard-NEEDs
            # libnvshmem_host; nvshmem probes transports and tolerates
            # absent plugins. Core libs must resolve — hard-fail there.
            if re.match(r"^nvshmem_(transport|bootstrap)_", so.name):
                log(f"dropping optional plugin {so.name} (NEEDs {outside})")
                dropped.append(so.name)
                so.unlink()
            else:
                sys.exit(f"nvshmem CORE lib {so.name} NEEDs unexpected {outside}; "
                         "add a conda mapping before shipping")

    # license texts also ride the payload (redistribution condition)
    licdir = stage / "share" / "licenses" / "nvidia-nvshmem"
    licdir.mkdir(parents=True)
    for flat, blob in sorted(lics.items()):
        (licdir / flat).write_bytes(blob)

    hsh = hashlib.sha256(f"nvidia-nvshmem|{pin_version}|cuda{fam}".encode()).hexdigest()[:8]
    index = {
        "arch": PLATFORMS[subdir]["arch"], "platform": "linux", "subdir": subdir,
        "name": "nvidia-nvshmem", "version": pin_version,
        "build": f"cuda{fam}_repack_h{hsh}_{build_number}", "build_number": build_number,
        "depends": [f"cuda-version >={fam},<{fam + 1}",
                    f"cuda-cudart >={fam},<{fam + 1}"] + floor_deps([stage]),
        # defensive mutex: the day conda-forge publishes an `nvshmem` at
        # lib/libnvshmem_host.so.3, co-install becomes a loud UNSAT
        # instead of a silent clobber
        "constrains": ["nvshmem <0.0a0"],
        "license": "LicenseRef-NVIDIA-Proprietary", "timestamp": int(time.time() * 1000),
    }
    about = {"home": "https://developer.nvidia.com/nvshmem",
             "license": "LicenseRef-NVIDIA-Proprietary",
             "summary": "NVSHMEM runtime, repacked from the official PyPI wheel "
                        "(no conda-forge nvshmem exists)",
             "extra": {"repacked_from": wheel.name, "wheel_sha256": sha256_file(wheel),
                       "wheel_hash_source": "pypi-api",
                       "dropped_optional_plugins": dropped}}
    out = emit_conda(stage, outdir, index, about, {}, licenses=lics)
    dep = f"nvidia-nvshmem >={pin_version},<{int(pin_version.split('.')[0]) + 1}"
    return dep, out


# sonames a repacked cudnn lib may legitimately NEED
CUDNN_OK_NEEDED = re.compile(
    r"^(libc[.\d]|libm[.\d]|libdl|librt|libpthread|ld-linux|libgcc_s"
    r"|libstdc\+\+|libz\.so|libcudart|libcublas|libnvrtc|libcudnn)")


def build_nvidia_lib(pypi_name: str, pin: str, subdir: str,
                     outdir: Path, work: Path, conda_name: str) -> None:
    """Repack an nvidia-* payload wheel (cudnn today) into $PREFIX/lib
    under `conda_name` (the conda-forge name — see NVIDIA_FALLBACK). The
    artifact is shared across entries and immutable once published
    (emit_conda's skip-published check reuses it); SIDE_BUILD bumps it
    when the pipeline itself changes."""
    comp = re.sub(r"-cu\d+$", "", pypi_name)[len("nvidia-"):]
    fam = int(re.search(r"-cu(\d+)$", pypi_name).group(1))
    plat = "aarch64" if subdir == "linux-aarch64" else "x86_64"
    url, whl_sha = pypi_wheel_url(pypi_name, pin, [plat])
    wheel = work / url.rsplit("/", 1)[1]
    download(url, wheel, whl_sha)
    lics = license_gate(wheel, conda_name)

    stage = work / f"nvlib_{comp}"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "lib").mkdir(parents=True)
    zf = zipfile.ZipFile(wheel)
    n_libs = 0
    for zi in zf.infolist():
        parts = PurePosixPath(zi.filename).parts
        if len(parts) >= 4 and parts[:3] == ("nvidia", comp, "lib"):
            dest = stage / "lib" / Path(*parts[3:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zi) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            n_libs += 1
    zf.close()
    if n_libs == 0:
        sys.exit(f"{pypi_name} wheel had no nvidia/{comp}/lib payload")

    for so in sorted((stage / "lib").rglob("*.so*")):
        with open(so, "rb") as fh:
            magic = fh.read(4)
        if not (so.is_file() and magic == b"\x7fELF"):
            continue
        run("patchelf", "--force-rpath", "--set-rpath", "$ORIGIN", str(so))
        needed = subprocess.run(["patchelf", "--print-needed", str(so)],
                                capture_output=True, text=True, check=True).stdout.split()
        outside = [d for d in needed if not CUDNN_OK_NEEDED.match(d)]
        if outside:
            sys.exit(f"{comp} lib {so.name} NEEDs unexpected {outside}; "
                     "extend the whitelist and dependency set before shipping")

    licdir = stage / "share" / "licenses" / conda_name
    licdir.mkdir(parents=True)
    for flat, blob in sorted(lics.items()):
        (licdir / flat).write_bytes(blob)

    hsh = hashlib.sha256(f"{conda_name}|{pin}|cuda{fam}".encode()).hexdigest()[:8]
    index = {
        "arch": PLATFORMS[subdir]["arch"], "platform": "linux", "subdir": subdir,
        "name": conda_name, "version": pin,
        "build": f"cuda{fam}_repack_h{hsh}_{SIDE_BUILD}", "build_number": SIDE_BUILD,
        "depends": [f"cuda-version >={fam},<{fam + 1}",
                    f"cuda-cudart >={fam},<{fam + 1}", "libcublas", "cuda-nvrtc",
                    "libzlib"] + floor_deps([stage]),
        "license": "LicenseRef-NVIDIA-Proprietary", "timestamp": int(time.time() * 1000),
    }
    about = {"home": "https://developer.nvidia.com",
             "license": "LicenseRef-NVIDIA-Proprietary",
             "summary": f"{comp}, repacked from the official PyPI wheel "
                        "(no conda-forge build fits the entry's cuda-version window)",
             "extra": {"repacked_from": wheel.name, "wheel_sha256": sha256_file(wheel),
                       "wheel_hash_source": "pypi-api"}}
    emit_conda(stage, outdir, index, about, {}, licenses=lics)


def side_repack_pywheel(pypi_name: str, version: str, py: str, subdir: str,
                        outdir: Path, work: Path, build_number: int) -> None:
    """Repack a python-wheel dep (triton, cuda-bindings) whose translated
    range conda-forge cannot satisfy for this torch line — e.g. triton
    3.0.0 and 3.8.x have no conda-forge build at all. Same dist-info
    contract as pytorch. Console scripts are NOT generated (recorded in
    about.json; torch never shells out to them)."""
    pytag = "py" + py.replace(".", "")
    hsh = hashlib.sha256(f"{pypi_name}|{version}|{py}".encode()).hexdigest()[:8]
    stem = f"{pypi_name}-{version}-repack_{pytag}_h{hsh}_{build_number}"
    if SKIP_PUBLISHED and asset_published(subdir, f"{stem}.conda"):
        # deterministic filename: a sibling run (or an earlier wave) already
        # built and published this exact side artifact. Skip before ever
        # touching the network/license gate — never re-download the wheel
        # just to discover we'd throw the result away (this is what made
        # already-published triton 3.0.0 side-repacks fail the license
        # gate on a later PyPI reupload: work we didn't need to redo).
        if (Path("meta") / subdir / f"{stem}.conda.json").exists():
            log(f"SKIP {stem}: already published with fragment")
            return
        log(f"{stem}: published but fragment missing; fetching published bytes")
        download(f"{RELEASES}/{subdir}/{stem}.conda", outdir / f"{stem}.conda")
        return

    cp = "cp" + py.replace(".", "")
    plat = "aarch64" if subdir == "linux-aarch64" else "x86_64"
    url, whl_sha = pypi_wheel_url(pypi_name, version, [f"-{cp}-", plat])
    wheel = work / url.rsplit("/", 1)[1]
    download(url, wheel, whl_sha)
    lics = license_gate(wheel, pypi_name)

    stage = work / f"side_{pypi_name}"
    if stage.exists():
        shutil.rmtree(stage)
    sp = stage / f"lib/python{py}/site-packages"
    sp.mkdir(parents=True)
    extract_wheel(wheel, sp)

    di = next(sp.glob("*.dist-info"))
    (di / "RECORD").unlink(missing_ok=True)
    (di / "INSTALLER").write_bytes(b"conda")
    for junk in ("direct_url.json", "RECORD.jws", "REQUESTED"):
        (di / junk).unlink(missing_ok=True)
    entry_points = (di / "entry_points.txt").read_text() if (di / "entry_points.txt").exists() else ""
    if entry_points:
        log(f"{pypi_name}: console scripts NOT generated ({len(entry_points)} bytes of entry_points.txt)")

    metadata_text = (di / "METADATA").read_text(errors="replace")
    requires = parse_requires(metadata_text, py, subdir)
    sp_rel = f"lib/python{py}/site-packages"
    for top in sorted(p for p in sp.iterdir() if p.is_dir() and not p.name.endswith(".dist-info")):
        compileall.compile_dir(str(top), quiet=2, workers=0, ddir=f"{sp_rel}/{top.name}",
                               invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)

    has_elf = False
    # ALL ELF files, not *.so* — triton vendors ELF EXECUTABLES too
    # (ptxas, cuobjdump, FileCheck) whose upstream RUNPATH carries
    # build-machine paths and empty entries (load-from-CWD)
    for so in sorted(sp.rglob("*")):
        if not so.is_file() or so.is_symlink():
            continue
        with open(so, "rb") as fh:
            if fh.read(4) != b"\x7fELF":
                continue
        has_elf = True
        rel = os.path.relpath(stage / "lib", so.parent)
        patch_rpath(so, f"$ORIGIN:$ORIGIN/{rel}")

    deps = []
    for name, _, spec in requires:
        if name not in PY_DEP_MAP:
            sys.exit(f"{pypi_name} dependency {name!r} ({spec!r}) unmapped; add to PY_DEP_MAP")
        deps.append(f"{PY_DEP_MAP[name]} {conda_spec(spec)}".strip())
    nxt = f"{py.split('.')[0]}.{int(py.split('.')[1]) + 1}"
    deps += [f"python >={py},<{nxt}.0a0", f"python_abi {py}.* *_cp{py.replace('.', '')}"]
    if has_elf:
        deps += floor_deps([sp]) + ["libzlib"]

    lic = next((ln.split(":", 1)[1].strip() for ln in metadata_text.splitlines()
                if ln.lower().startswith(("license-expression:", "license:"))
                and 0 < len(ln.split(":", 1)[1].strip()) < 64),
               "LicenseRef-Wheel-License")
    pytag = "py" + py.replace(".", "")
    hsh = hashlib.sha256(f"{pypi_name}|{version}|{py}".encode()).hexdigest()[:8]
    index = {
        "arch": PLATFORMS[subdir]["arch"], "platform": PLATFORMS[subdir]["platform"],
        "subdir": subdir, "name": pypi_name, "version": version,
        "build": f"repack_{pytag}_h{hsh}_{build_number}", "build_number": build_number,
        "depends": sorted(set(deps)),
        "license": lic, "timestamp": int(time.time() * 1000),
    }
    about = {"license": lic,
             "summary": f"{pypi_name}, repacked from the official PyPI wheel "
                        "(conda-forge has no build satisfying torch's pin)",
             "extra": {"repacked_from": wheel.name, "wheel_sha256": sha256_file(wheel),
                       "wheel_hash_source": "pypi-api",
                       "skipped_entry_points": entry_points}}
    emit_conda(stage, outdir, index, about, {}, licenses=lics,
               run_exports=None)


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


RELEASES = "https://github.com/Comfy-Forge/conda-torch/releases/download"
SKIP_PUBLISHED = False  # set by --skip-published (CI): reuse published assets
SIDE_BUILD = 0  # build number for SHARED side artifacts (nvshmem/cudnn/triton):
                # one fixed value per wave so every cell converges on the same
                # filename and skip-published dedups instead of forking copies


def asset_published(subdir: str, filename: str) -> bool:
    req = urllib.request.Request(
        f"{RELEASES}/{subdir}/{filename}", method="HEAD",
        headers={"User-Agent": "conda-torch/0.1"})
    try:
        urllib.request.urlopen(req, timeout=30)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def emit_conda(stage: Path, outdir: Path, index: dict, about: dict,
               prefix_files: dict[str, str],
               licenses: dict[str, bytes] | None = None,
               run_exports: dict | None = None) -> Path | None:
    name, version, build = index["name"], index["version"], index["build"]
    stem = f"{name}-{version}-{build}"
    if SKIP_PUBLISHED and asset_published(index["subdir"], f"{stem}.conda"):
        # a sibling run already built and published this exact artifact
        # (shared libtorch across shims, shared nvshmem/triton across
        # cells). Published artifacts are immutable — never rebuild them.
        if (Path("meta") / index["subdir"] / f"{stem}.conda.json").exists():
            log(f"SKIP {stem}: already published with fragment")
            return None
        log(f"{stem}: published but fragment missing; fetching published bytes")
        out = outdir / f"{stem}.conda"
        download(f"{RELEASES}/{index['subdir']}/{stem}.conda", out)
        return out

    about = dict(about)
    about["extra"] = {**about.get("extra", {}), **provenance()}
    paths, files = collect_paths(stage, prefix_files)

    # very large payloads at zstd-19 can exhaust CI runner memory; drop to
    # a cheaper level past 2 GiB raw and record that in about.json
    raw_total = sum(p.get("size_in_bytes", 0) for p in paths)
    level = 19 if raw_total < (2 << 30) else 11
    about = dict(about)
    if level != 19:
        about.setdefault("extra", {})
        about["extra"] = dict(about["extra"], zstd_level=level,
                              zstd_note=f"level lowered from 19: {raw_total} raw bytes")
        log(f"payload {raw_total} bytes raw: compressing at zstd level {level}")

    infodir = outdir / f"_info_{name}"
    if infodir.exists():
        shutil.rmtree(infodir)
    (infodir / "info").mkdir(parents=True)
    w = lambda n, s: (infodir / "info" / n).write_text(s)
    w("index.json", json.dumps(index, indent=2, sort_keys=True))
    w("paths.json", json.dumps({"paths": paths, "paths_version": 1}, indent=2))
    w("about.json", json.dumps(about, indent=2, sort_keys=True))
    w("files", "".join(f + "\n" for f in files))
    if run_exports:
        w("run_exports.json", json.dumps(run_exports, indent=2, sort_keys=True))
    if licenses:
        (infodir / "info" / "licenses").mkdir()
        for flat, blob in sorted(licenses.items()):
            (infodir / "info" / "licenses" / flat).write_bytes(blob)
    if prefix_files:
        w("has_prefix", "".join(
            f"{PLACEHOLDER} {mode} {rel}\n" for rel, mode in sorted(prefix_files.items())))

    cctx = zstandard.ZstdCompressor(level=level, threads=-1)

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
    size = out.stat().st_size
    if size >= (2 << 30):
        sys.exit(f"{out.name} is {size} bytes — over the 2 GiB release-asset cap")
    log(f"built {out.name}: {size} bytes, sha256 {sha256_file(out)}")
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--flavour", required=True, help="e.g. cu128")
    ap.add_argument("--python", dest="py", required=True, help="e.g. 3.12")
    ap.add_argument("--subdir", default="linux-64", choices=sorted(PLATFORMS))
    ap.add_argument("--build-number", type=int, default=0)
    ap.add_argument("--side-build-number", type=int, default=0,
                    help="build number for shared side artifacts "
                         "(nvshmem/cudnn/triton); fixed per wave")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("dist"))
    ap.add_argument("--wheel", type=Path, default=None,
                    help="use a pre-downloaded wheel instead of fetching")
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--delete-wheel", action="store_true",
                    help="delete the wheel right after extraction (CI disk)")
    ap.add_argument("--skip-published", action="store_true",
                    help="skip building artifacts already on the release "
                         "(shared libtorch/nvshmem/triton across cells)")
    args = ap.parse_args()
    if args.skip_published:
        global SKIP_PUBLISHED
        SKIP_PUBLISHED = True
    global SIDE_BUILD
    SIDE_BUILD = args.side_build_number

    if f"{sys.version_info.major}.{sys.version_info.minor}" != args.py:
        sys.exit(f"must run under python {args.py} for .pyc magic "
                 f"(running {sys.version_info.major}.{sys.version_info.minor})")
    is_linux = args.subdir.startswith("linux")
    if is_linux and shutil.which("patchelf") is None:
        sys.exit("patchelf not on PATH")

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    wheel = args.wheel
    if wheel is None:
        url, idx_sha = wheel_url(args.version, args.flavour, args.py, args.subdir)
        gsha = grid_wheel_sha(args.version, args.flavour, args.subdir, args.py)
        if idx_sha and gsha and idx_sha != gsha:
            sys.exit(f"index sha256 {idx_sha} != grid.json sha256 {gsha}: "
                     "two sources disagree about this wheel — refusing")
        expect = idx_sha or gsha
        hash_source = ("index+grid" if idx_sha and gsha else
                       "index" if idx_sha else "grid" if gsha else "TOFU")
        if hash_source == "TOFU":
            log("WARNING: no published sha256 for this wheel anywhere (no index "
                "anchor, grid.json silent) — trust-on-first-use, recorded as such")
        wheel = args.work / PurePosixPath(url.split("#")[0]).name.replace("%2B", "+")
        download(url, wheel, expect)
    else:
        hash_source = "local-wheel"
    wheel_sha = sha256_file(wheel)
    torch_licenses = license_gate(wheel, "pytorch/libtorch")

    # ---- extract full wheel into the pytorch stage -------------------------
    pt_stage = args.work / "pytorch_stage"
    lt_stage = args.work / "libtorch_stage"
    for s in (pt_stage, lt_stage):
        if s.exists():
            shutil.rmtree(s)
    sp_rel = "Lib/site-packages" if args.subdir == "win-64" else f"lib/python{args.py}/site-packages"
    sp = pt_stage / sp_rel
    sp.mkdir(parents=True)
    lt_stage.mkdir(parents=True)

    extract_wheel(wheel, sp)
    if args.delete_wheel:
        wheel_name = wheel.name
        wheel.unlink()
    else:
        wheel_name = wheel.name

    dist_info = scrub_dist_info(sp)
    requires = parse_requires((dist_info / "METADATA").read_text(errors="replace"),
                              args.py, args.subdir)
    byte_compile(sp, sp_rel)

    # ---- CUDA-ness guard: the flavour directory can serve tag-less wheels;
    # version.py is the wheel's own statement of what it is --------------------
    vpy = (sp / "torch" / "version.py").read_text(errors="replace")
    m = re.search(r"^cuda[^=]*=\s*['\"]([0-9.]+)['\"]", vpy, re.M)
    want_cuda = f"{args.flavour[2:-1]}.{args.flavour[-1]}"
    if not m or m.group(1) != want_cuda:
        sys.exit(f"torch/version.py says cuda={m.group(1) if m else None!r}, "
                 f"flavour {args.flavour} wants {want_cuda!r}: refusing to publish "
                 "a mislabeled build (CPU wheel or wrong flavour — report as hole)")

    # ---- platform split ----------------------------------------------------
    extra_lt_deps: list[str] = []
    if is_linux:
        prefix_files, stripped_cuda = split_linux(sp, pt_stage, lt_stage, args.subdir)
        if stripped_cuda:
            extra_lt_deps += vendored_cuda_dep_list(
                stripped_cuda, args.version, args.flavour, args.py, args.subdir)
        nvshmem_req = next(((name, spec) for name, _, spec in requires
                            if nvidia_key(name) == "nvidia-nvshmem"), None)
        if nvshmem_req:
            pin = spec_floor(nvshmem_req[1])
            if not pin:
                sys.exit(f"cannot derive nvshmem version from {nvshmem_req[1]!r}")
            log(f"repacking {nvshmem_req[0]} {pin} (hard DT_NEEDED, no conda-forge pkg)...")
            # shared side artifact: SIDE_BUILD (immutable, reused across
            # entries; entry-level bumps must not fork new copies)
            build_nvshmem(nvshmem_req[0], pin, args.subdir, args.py,
                          args.outdir, args.work, SIDE_BUILD)
        # side-repack python-wheel deps conda-forge cannot satisfy (e.g.
        # triton 3.0.0 / 3.8.x never got a conda-forge build)
        for name, _, spec in requires:
            if name in SIDE_REPACK_OK and not cf_satisfiable(SIDE_REPACK_OK[name], spec, args.subdir):
                floor = spec_floor(spec)
                if not floor:
                    sys.exit(f"cannot derive {name} version from {spec!r}")
                log(f"side-repacking {name} {floor}: conda-forge cannot satisfy "
                    f"{spec!r} on {args.subdir}")
                side_repack_pywheel(name, floor, args.py, args.subdir,
                                    args.outdir, args.work, SIDE_BUILD)
    else:
        prefix_files, win_imports, win_stripped = split_win64(sp, pt_stage, lt_stage)

    # ---- entry point (POSIX; win handled inside split_win64) ---------------
    if is_linux:
        bindir = pt_stage / "bin"
        bindir.mkdir(exist_ok=True)
        (bindir / "torchrun").write_text(TORCHRUN)
        os.chmod(bindir / "torchrun", 0o755)
        prefix_files = dict(prefix_files)
        prefix_files["bin/torchrun"] = "text"
    # torchfrtrace deliberately NOT generated: its module (tools.flight_recorder)
    # is not in the wheel; the upstream entry point is broken.

    # ---- metadata ----------------------------------------------------------
    flavour, py = args.flavour, args.py
    plat = PLATFORMS[args.subdir]
    pytag = "py" + py.replace(".", "")
    now = int(time.time() * 1000)
    if args.subdir == "win-64":
        lt_deps = win_cuda_deps(args.version, flavour, py) + [
            "intel-openmp", "ucrt >=10.0.20348.0", "vc >=14.2,<15",
            "vc14_runtime >=14.44",
        ]
        # every stripped-but-imported DLL basename must provably ship
        # inside the pinned windows (cupti's is minor-versioned)
        lt_deps = win_basename_audit(win_imports, win_stripped, lt_deps, flavour)
    else:
        lt_deps = cuda_deps(requires, flavour) + ["libgomp"] + floor_deps([lt_stage])
        if args.subdir == "linux-aarch64":
            lt_deps.append("libzlib")  # vendored gfortran/cudnn_graph NEED libz.so.1
    # a version satisfying the bound is not enough — a conda-forge BUILD
    # must exist inside the flavour's cuda-version window (libcudnn 9.19+
    # is cuda13-only there); otherwise substitute a PyPI side-repack
    lt_deps = fix_window_unsat(lt_deps, requires, flavour, args.subdir,
                               args.outdir, args.work)
    hsh = lambda s: hashlib.sha256(s.encode()).hexdigest()[:8]
    lt_build = f"cuda{flavour[2:]}_repack_h{hsh('libtorch|' + args.version + '|' + flavour)}_{args.build_number}"
    pt_build = f"cuda{flavour[2:]}_repack_{pytag}_h{hsh('pytorch|' + args.version + '|' + flavour + '|' + py)}_{args.build_number}"

    lt_index = {
        "arch": plat["arch"], "platform": plat["platform"], "subdir": args.subdir,
        "name": "libtorch", "version": args.version,
        "build": lt_build, "build_number": args.build_number,
        "depends": sorted(set(lt_deps + extra_lt_deps)),
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
    if is_linux:
        pt_deps += floor_deps([pt_stage])
    # NOTE: no pytorch-cpu/pytorch-gpu constrains. The old
    # `pytorch-gpu <0.0a0` silently made every env using conda-forge's
    # common pytorch-gpu metapackage idiom resolve to mirrors only, never
    # repacks; the mutex intent is already served by the package NAME and
    # the exact libtorch build pin.
    pt_index = {
        "arch": plat["arch"], "platform": plat["platform"], "subdir": args.subdir,
        "name": "pytorch", "version": args.version,
        "build": pt_build, "build_number": args.build_number,
        "depends": sorted(set(pt_deps)),
        "license": "BSD-3-Clause", "license_family": "BSD",
        "timestamp": now,
    }
    about = {
        "home": "https://pytorch.org",
        "license": "BSD-3-Clause",
        "summary": "PyTorch, repacked byte-for-byte from the official PyPI wheel",
        "description": f"Repacked from {wheel_name}. See github.com/Comfy-Forge/conda-torch.",
        "extra": {"repacked_from": wheel_name, "wheel_sha256": wheel_sha,
                  "wheel_hash_source": hash_source},
    }

    # licenses ride the libtorch payload too (its stage has no dist-info)
    lt_licdir = lt_stage / "share" / "licenses" / "libtorch"
    lt_licdir.mkdir(parents=True, exist_ok=True)
    for flat, blob in sorted(torch_licenses.items()):
        (lt_licdir / flat).write_bytes(blob)

    mm = args.version.split(".")
    nxt_minor = f"{mm[0]}.{int(mm[1]) + 1}"
    lt_rex = {"weak": [f"libtorch >={args.version},<{nxt_minor}.0a0"]}
    pt_rex = {"weak": [f"pytorch >={args.version},<{nxt_minor}.0a0",
                       f"libtorch >={args.version},<{nxt_minor}.0a0"]}
    emit_conda(lt_stage, args.outdir, lt_index, about, {},
               licenses=torch_licenses, run_exports=lt_rex)
    emit_conda(pt_stage, args.outdir, pt_index, about, prefix_files,
               licenses=torch_licenses, run_exports=pt_rex)
    log("done")


if __name__ == "__main__":
    main()
