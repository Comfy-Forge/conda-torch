#!/usr/bin/env python3
"""Structural verification of a repacked torch .conda (no target hardware).

Asserts the hacker-report invariants that can be checked from bytes alone:
zip layout, paths.json completeness vs the payload tar, the no-RECORD /
INSTALLER=conda dist-info contract, platform strip lists, entry points,
ELF RPATH tags (readelf works cross-arch), and the 2 GiB asset cap.

Usage: verify_repack.py <pkg.conda> [<pkg.conda> ...]
Exits non-zero on the first violated invariant.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

FAILS = 0


def check(cond: bool, msg: str) -> None:
    global FAILS
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS += 1


def zstd_tar(blob: bytes) -> tarfile.TarFile:
    raw = subprocess.run(["zstd", "-d", "--stdout"], input=blob,
                         capture_output=True, check=True).stdout
    return tarfile.open(fileobj=io.BytesIO(raw))


def verify(conda_path: Path) -> None:
    global FAILS
    before = FAILS
    print(f"\n=== {conda_path.name} ===")
    size = conda_path.stat().st_size
    check(size < (2 << 30), f"size {size} under 2 GiB asset cap")

    z = zipfile.ZipFile(conda_path)
    names = z.namelist()
    check("metadata.json" in names, "metadata.json present")
    check(all(zi.compress_type == zipfile.ZIP_STORED for zi in z.infolist()),
          "all zip members STORED")
    pkg = [n for n in names if n.startswith("pkg-")]
    info = [n for n in names if n.startswith("info-")]
    check(len(pkg) == 1 and len(info) == 1, "exactly one pkg- and one info- tarball")

    with zstd_tar(z.read(info[0])) as tf:
        index = json.load(tf.extractfile("info/index.json"))
        paths = json.load(tf.extractfile("info/paths.json"))
    name, subdir = index["name"], index["subdir"]
    check("+" not in index["version"], "no '+' in version (conda ordering trap)")
    check("build_number" in index, "build_number present")
    pset = {p["_path"] for p in paths["paths"]}

    with zstd_tar(z.read(pkg[0])) as tf:
        members = {m.name: m for m in tf.getmembers() if not m.isdir()}
        # dist-info contract
        di = [n for n in members if ".dist-info/" in n]
        check(not any(n.endswith("/RECORD") for n in di),
              "no RECORD in dist-info (pip uninstall trap)")
        check(not any(n.endswith("direct_url.json") for n in di),
              "no direct_url.json")
        inst = [n for n in di if n.endswith("/INSTALLER")]
        if inst:
            check(tf.extractfile(inst[0]).read() == b"conda",
                  "INSTALLER is exactly b'conda'")
        elif name in ("pytorch",):
            check(False, "pytorch package missing dist-info INSTALLER")

        check(pset == set(members), "paths.json exactly matches payload tar "
              f"(paths {len(pset)}, tar {len(members)})")

        if name == "pytorch":
            if subdir.startswith("linux"):
                check(any(n.endswith("bin/torchrun") for n in members),
                      "bin/torchrun present")
                check(any("torch/bin/torch_shm_manager" in n for n in members),
                      "torch/bin/torch_shm_manager present (import hard-check)")
                check("lib/libtorch_python.so" in members,
                      "libtorch_python.so at $PREFIX/lib")
            else:
                check("Scripts/torchrun.exe" in members, "Scripts/torchrun.exe present")
                exe = tf.extractfile(members["Scripts/torchrun.exe"]).read()
                check(b"__main__.py" in exe[-4096:], "launcher has embedded __main__.py")
                check("Lib/site-packages/torch/lib/torch_python.dll" in members,
                      "torch_python.dll stays in pytorch")
            check(any(n.endswith(".pyc") for n in members), ".pyc shipped")

        if name == "libtorch" and subdir == "win-64":
            bad_dll = [n for n in members if n.lower().endswith(".dll")
                       and n.rsplit("/", 1)[-1].lower().startswith(
                ("cudart64", "cublas64", "cudnn", "cufft64", "cupti64", "curand64",
                 "cusolver64", "cusparse64", "nvjitlink", "nvrtc64", "libiomp5md"))]
            check(not bad_dll, f"no vendored CUDA/iomp DLLs {bad_dll[:3]}")
            bad_lib = [n for n in members if n.endswith((
                "dnnl.lib", "libprotobuf.lib", "XNNPACK.lib", "fbgemm.lib", "asmjit.lib"))]
            check(not bad_lib, f"dead .lib stripped {bad_lib[:3]}")
            for want in ("torch_cuda.dll", "torch_cpu.dll", "c10.lib", "torch_cuda.lib"):
                check(any(n.endswith(want) for n in members), f"{want} kept")

        if name in ("libtorch", "pytorch", "nvidia-nvshmem") and subdir.startswith("linux"):
            elfs = [n for n in members
                    if members[n].isfile() and (n.endswith(".so") or ".so." in n)
                    and tf.extractfile(members[n]).read(4) == b"\x7fELF"]
            gomp = [n for n in elfs if "libgomp" in n]
            check(not gomp, f"no vendored libgomp {gomp[:2]}")
            for n in elfs[:6] if name == "nvidia-nvshmem" else \
                    [e for e in elfs if e.rsplit("/", 1)[-1] in
                     ("libtorch_cuda.so", "libtorch_cpu.so", "libtorch_python.so",
                      "libtorch_nvshmem.so", "libc10_cuda.so")]:
                blob = tf.extractfile(members[n]).read()
                tmp = Path("/tmp/_verify_elf")
                tmp.write_bytes(blob)
                dyn = subprocess.run(["readelf", "-d", str(tmp)], capture_output=True,
                                     text=True).stdout
                check(dyn != "", f"readelf parsed {n}")
                check("(RPATH)" in dyn and "(RUNPATH)" not in dyn,
                      f"{n.rsplit('/', 1)[-1]}: DT_RPATH not RUNPATH")
                if name != "nvidia-nvshmem":
                    check("$ORIGIN" in dyn, f"{n.rsplit('/', 1)[-1]}: $ORIGIN rpath present")

    print(f"--- {conda_path.name}: {'PASS' if FAILS == before else 'FAILURES ABOVE'}")


def main() -> None:
    for arg in sys.argv[1:]:
        verify(Path(arg))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
