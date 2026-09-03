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
        infonames = tf.getnames()
    name, subdir = index["name"], index["subdir"]
    check("+" not in index["version"], "no '+' in version (conda ordering trap)")
    check("build_number" in index, "build_number present")
    if "repack" in index.get("build", ""):
        check(any(n.startswith("info/licenses/") for n in infonames),
              "info/licenses/ present (license gate)")
        if name in ("libtorch", "pytorch"):
            check("info/run_exports.json" in infonames, "run_exports.json present")
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
                check("Scripts/torchrun-script.py" in members,
                      "torchrun-script.py sibling present (launcher pair)")
                script = tf.extractfile(members["Scripts/torchrun-script.py"]).read()
                check(script.startswith(b"#!") and b"_placehold" in script[:300],
                      "script shebang carries the prefix placeholder")
                reg = next((p for p in paths["paths"]
                            if p["_path"] == "Scripts/torchrun-script.py"), {})
                check(reg.get("file_mode") == "text" and "prefix_placeholder" in reg,
                      "torchrun-script.py registered for text prefix rewrite")
                exe = tf.extractfile(members["Scripts/torchrun.exe"]).read()
                check(b"__main__.py" not in exe[-4096:],
                      "launcher is the plain stub (no dead appended archive)")
                check("Lib/site-packages/torch/lib/torch_python.dll" in members,
                      "torch_python.dll stays in pytorch")
            pycs = [n for n in members if n.endswith(".pyc")]
            check(bool(pycs), ".pyc shipped")
            if pycs:
                blob = tf.extractfile(members[pycs[0]]).read()
                check(b"pytorch_stage" not in blob and b"work/" not in blob[:2000],
                      ".pyc co_filename is env-relative (no staging path)")

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

        if subdir.startswith("linux"):
            # RPATH LINT, every ELF in the payload: DT_RPATH (not RUNPATH),
            # and every entry $ORIGIN-relative and non-empty. Absolute
            # entries shadow the prefix on Jetson/SBSA (/usr/local/cuda);
            # an EMPTY entry ('::') means load-from-CWD; leftover wheel
            # nvidia/* entries resurrect under pip-installed wheels.
            import shutil as _sh
            import re as _re
            n_elf = 0
            bad: list[str] = []
            for n, m in members.items():
                if not m.isfile():
                    continue
                src = tf.extractfile(m)
                head = src.read(4)
                if head != b"\x7fELF":
                    continue
                n_elf += 1
                tmp = Path("/tmp/_verify_elf")
                with open(tmp, "wb") as dst:
                    dst.write(head)
                    _sh.copyfileobj(src, dst, 1 << 20)
                dyn = subprocess.run(["readelf", "-d", str(tmp)],
                                     capture_output=True, text=True).stdout
                base = n.rsplit("/", 1)[-1]
                if "libgomp" in base:
                    bad.append(f"{base}: vendored libgomp")
                    continue
                if "(RUNPATH)" in dyn:
                    bad.append(f"{base}: RUNPATH (loses to LD_LIBRARY_PATH)")
                rp = _re.search(r"\((?:RPATH|RUNPATH)\)\s+Library r?u?n?path: \[([^\]]*)\]", dyn)
                if rp:
                    for entry in rp.group(1).split(":"):
                        if not entry or not entry.startswith("$ORIGIN"):
                            bad.append(f"{base}: rpath entry {entry!r}")
            check(n_elf > 0 or name not in ("libtorch", "pytorch"),
                  f"found {n_elf} ELFs to lint")
            check(not bad, f"RPATH lint clean over {n_elf} ELFs "
                  + ("" if not bad else f"— violations: {bad[:5]}"))

    print(f"--- {conda_path.name}: {'PASS' if FAILS == before else 'FAILURES ABOVE'}")


def main() -> None:
    for arg in sys.argv[1:]:
        verify(Path(arg))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
