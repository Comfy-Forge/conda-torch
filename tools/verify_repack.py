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
        # link.json is what tells conda to generate console scripts at install
        # time, and it is required exactly for `noarch: python` packages --
        # rattler-build emits it for those and (correctly) not for a
        # platform-specific package like ours, which ships bin/torchrun as a
        # real file with a prefix placeholder. Nothing here is noarch python
        # today, so this gate is dormant; it exists so that the day something
        # is, a missing link.json fails the build instead of silently shipping
        # a package whose entry points never get created.
        if index.get("noarch") == "python":
            check("info/link.json" in infonames,
                  "noarch: python package carries info/link.json "
                  "(without it conda never creates its console scripts)")
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
            # rattler-build owns this file: it overwrites whatever the build
            # staged with b"conda\n" and offers no knob to suppress the
            # trailing newline, so a byte-exact assert is unsatisfiable under
            # the rattler pipeline. Compare content, not trailing whitespace.
            val = tf.extractfile(inst[0]).read()
            check(val.strip() == b"conda",
                  f"INSTALLER names conda (got {val!r})")
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

        if name == "libtorch" and "repack" in index.get("build", ""):
            # conda-forge parity: the activation pair must ship and its
            # CF_TORCH_CUDA_ARCH_LIST must equal the gencode set baked into
            # this very artifact's ATen/cuda/CUDAConfig.h (self-contained).
            import re as _re2
            ext = "bat" if subdir == "win-64" else "sh"
            act = f"etc/conda/activate.d/libtorch_activate.{ext}"
            deact = f"etc/conda/deactivate.d/libtorch_deactivate.{ext}"
            check(act in members and deact in members,
                  f"activation pair present ({act}, {deact})")
            check(act in pset and deact in pset, "activation pair listed in paths.json")
            hdr = next((n for n in members if n.endswith("ATen/cuda/CUDAConfig.h")), None)
            check(hdr is not None, "ATen/cuda/CUDAConfig.h in payload")
            if act in members and hdr is not None:
                script = tf.extractfile(members[act]).read().decode(errors="replace")
                got = _re2.search(r'CF_TORCH_CUDA_ARCH_LIST=\"?([^\"\r\n]*)', script)
                got = got.group(1) if got else ""
                flags = _re2.search(r'NVCC_FLAGS_EXTRA\s+"([^"]*)"',
                                    tf.extractfile(members[hdr]).read().decode(errors="replace"))
                sm, ptx = set(), set()
                for kind, n_ in _re2.findall(r"code=(sm|compute)_(\d+)", flags.group(1) if flags else ""):
                    (sm if kind == "sm" else ptx).add(int(n_))
                want = ";".join(f"{n_ // 10}.{n_ % 10}" + ("+PTX" if n_ in ptx else "")
                                for n_ in sorted(sm | ptx))
                check(bool(got), "CF_TORCH_CUDA_ARCH_LIST non-empty")
                check(got == want, f"CF_TORCH_CUDA_ARCH_LIST matches CUDAConfig.h ({got!r} vs {want!r})")

        if name == "libtorch" and subdir == "win-64":
            bad_dll = [n for n in members if n.lower().endswith(".dll")
                       and n.rsplit("/", 1)[-1].lower().startswith(
                ("cudart64", "cublas64", "cudnn", "cufft64", "curand64",
                 "cusolver64", "cusparse64", "nvjitlink", "nvrtc64", "libiomp5md"))]
            check(not bad_dll, f"no vendored CUDA/iomp DLLs {bad_dll[:3]}")
            # PE import gate over the payload's own DLLs (the static half of
            # tools/win_dll_audit.py): every static import must be another
            # payload DLL, a stripped conda-forge-backed basename, or an
            # OS / vc14_runtime / ucrt DLL. CUPTI in particular MUST be
            # vendored: its basename carries the toolkit patch version
            # (cupti64_2025.1.1.dll), torch_cpu.dll imports it statically,
            # and conda-forge's cuda-cupti cannot be relied on to ship that
            # name across the flavour's cuda-version window (see
            # torch_repack.py WIN_STRIP_DLL). So a cuda-cupti dep must NOT
            # exist, and the imported cupti basename must be in the payload.
            import re as _re3
            import pefile as _pefile
            cf_backed = _re3.compile(
                r"^(cudart64|cublas64|cublasLt64|cudnn|cufft64|cufftw64|curand64"
                r"|cusolver64|cusolverMg64|cusparse64|nvJitLink|nvrtc64"
                r"|nvrtc-builtins64|libiomp)", _re3.I)
            os_dll = _re3.compile(
                r"^(api-ms-win-.*|ext-ms-.*|kernel32|kernelbase|ntdll|advapi32|user32|gdi32"
                r"|ws2_32|shell32|ole32|oleaut32|shlwapi|dbghelp|psapi|iphlpapi|userenv"
                r"|msvcrt|crypt32|bcrypt|secur32|version|winmm|rpcrt4|setupapi|cfgmgr32"
                r"|powrprof|netapi32|imagehlp|wintrust|synchronization|dxgi|d3d1[12]"
                r"|msvcp140(_\w+)?|vcruntime140(_\w+)?|vcomp140|concrt140|ucrtbase"
                r"|python3\d*)\.dll$", _re3.I)
            payload_dlls = {n.rsplit("/", 1)[-1].lower(): n for n in members
                            if "/torch/lib/" in n and n.lower().endswith(".dll")}
            unresolved: list[str] = []
            cupti_imported: set[str] = set()
            for base, n in sorted(payload_dlls.items()):
                blob = tf.extractfile(members[n]).read()
                pe = _pefile.PE(data=blob, fast_load=True)
                pe.parse_data_directories(directories=[
                    _pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
                for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", None) or []:
                    imp = e.dll.decode(errors="replace")
                    if imp.lower().startswith("cupti64"):
                        cupti_imported.add(imp.lower())
                    if (imp.lower() in payload_dlls or cf_backed.match(imp)
                            or os_dll.match(imp)):
                        continue
                    unresolved.append(f"{base} -> {imp}")
                pe.close()
                del blob
            check(bool(payload_dlls), f"PE-walked {len(payload_dlls)} payload DLLs")
            check(not unresolved, f"every static import has a provider (unresolved: {unresolved[:5]})")
            check(cupti_imported <= set(payload_dlls),
                  f"imported CUPTI {sorted(cupti_imported)} vendored in torch/lib")
            check(not any(d.split(" ", 1)[0] == "cuda-cupti" for d in index.get("depends", [])),
                  "no cuda-cupti dependency (CUPTI is vendored)")
            vendored_nv = [b for b in payload_dlls
                           if _re3.match(r"^(cupti64_[\d.]+|nvtoolsext64_1)\.dll$", b)]
            if vendored_nv:
                lic_names = [n for n in infonames if n.startswith("info/licenses/")]
                check(any("license" in n.lower() and "nvidia" in n.lower() for n in lic_names),
                      f"NVIDIA EULA in info/licenses for vendored {vendored_nv} ({lic_names})")
                check("LicenseRef-NVIDIA" in index.get("license", ""),
                      f"license expression declares vendored NVIDIA DLLs ({index.get('license')!r})")
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
            shim_needs_sleef: bool | None = None  # None = shim not in this package
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
                if base == "libtorch_global_deps.so":
                    shim_needs_sleef = "[libsleef.so.3]" in dyn
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
            # sleef redirect invariant, both directions: a libtorch that
            # depends on sleef must ship a shim that DT_NEEDs libsleef.so.3,
            # and a shim that NEEDs it must be backed by the dependency.
            if name == "libtorch" and "repack" in index.get("build", ""):
                dep_sleef = any(d.split(" ", 1)[0] == "sleef" for d in index.get("depends", []))
                check(shim_needs_sleef is not None, "libtorch_global_deps.so present in libtorch")
                if shim_needs_sleef is not None:
                    check(dep_sleef == shim_needs_sleef,
                          f"sleef redirect consistent (dep={dep_sleef}, shim NEEDs={shim_needs_sleef})")

    print(f"--- {conda_path.name}: {'PASS' if FAILS == before else 'FAILURES ABOVE'}")


def main() -> None:
    for arg in sys.argv[1:]:
        verify(Path(arg))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
