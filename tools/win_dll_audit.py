#!/usr/bin/env python3
"""Resolve every PE import of a win-64 torch install, transitively.

The Windows loader reports only the FIRST DLL it fails on ("Error loading
shm.dll or one of its dependencies") and never which dependency; a missing
DLL anywhere in the transitive import tree surfaces as that one useless
line. This tool walks the tree the way the loader would and prints every
imported basename that resolves NOWHERE, so the diagnosis is a list of
names instead of a guess.

Search order mirrors torch/__init__.py::_load_dll_libraries, which calls
LoadLibraryExW(dll, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR) after os.add_dll_directory on torch/lib
and <prefix>/Library/bin:

  1. the importing DLL's own directory
  2. <prefix>/Lib/site-packages/torch/lib
  3. <prefix>/Library/bin          (conda-forge CUDA/openmp DLLs)
  4. <prefix>                      (python3x.dll, vc14_runtime, ucrt: the
                                    application directory of python.exe)
  5. %SystemRoot%\\System32       (on Windows); off-Windows, a fixed list
                                    of OS DLL names + API-set prefixes

<prefix>/DLLs is python's extension-module directory, NOT a loader search
path; it is deliberately excluded so the audit cannot pass on a name the
loader would miss.

Delay-load imports are walked too but reported separately: an unresolved
delay import is fatal only when the code path fires, and nvcuda.dll (the
driver) is legitimately absent on a GPU-less machine.

Exit status: 0 when every static import resolves; 1 otherwise. Usable on a
real Windows prefix (CI gate) or on a linux box against an extracted tree
(artifact inspection); pass --offline-system for the latter.

Usage:
  win_dll_audit.py --prefix <env> [--offline-system] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# OS-provided DLLs that no conda package ships. Only consulted when the
# audit runs off-Windows; on Windows System32 itself is the authority.
OS_DLLS = {
    "kernel32.dll", "kernelbase.dll", "user32.dll", "advapi32.dll", "gdi32.dll",
    "ws2_32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll", "shlwapi.dll",
    "dbghelp.dll", "psapi.dll", "iphlpapi.dll", "userenv.dll", "msvcrt.dll",
    "ntdll.dll", "crypt32.dll", "bcrypt.dll", "secur32.dll", "version.dll",
    "winmm.dll", "comdlg32.dll", "comctl32.dll", "rpcrt4.dll", "setupapi.dll",
    "cfgmgr32.dll", "powrprof.dll", "netapi32.dll", "wtsapi32.dll",
    "imm32.dll", "dwmapi.dll", "uxtheme.dll", "d3d11.dll", "dxgi.dll",
    "ncrypt.dll", "wldap32.dll", "normaliz.dll", "winhttp.dll", "wininet.dll",
    "opengl32.dll", "d3d12.dll", "dxcore.dll", "synchronization.dll",
    "pdh.dll", "mpr.dll", "propsys.dll", "avrt.dll", "mfplat.dll",
    "ucrtbase.dll", "concrt140.dll", "imagehlp.dll", "wintrust.dll",
}
# API-set forwarders (api-ms-win-*, ext-ms-*) are virtual names the OS
# resolves through apisetschema; they never exist as files a package ships.
API_SET_PREFIXES = ("api-ms-win-", "ext-ms-")
# Driver-provided DLLs: present only where an NVIDIA driver is installed.
# Expected missing on GPU-less CI; only reachable via delay-load / dlopen.
DRIVER_DLLS = {"nvcuda.dll", "nvml.dll", "nvapi64.dll"}


def pe_imports(path: Path) -> tuple[set[str], set[str]]:
    """(static imports, delay-load imports) as lowercase basenames."""
    import pefile
    pe = pefile.PE(str(path), fast_load=True)
    pe.parse_data_directories(directories=[
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]])
    static = {e.dll.decode(errors="replace").lower()
              for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", None) or []}
    delay = {e.dll.decode(errors="replace").lower()
             for e in getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", None) or []}
    pe.close()
    return static, delay


class Resolver:
    def __init__(self, prefix: Path, offline_system: bool):
        self.prefix = prefix
        self.torch_lib = prefix / "Lib" / "site-packages" / "torch" / "lib"
        self.search = [self.torch_lib, prefix / "Library" / "bin", prefix]
        self.system32: Path | None = None
        if not offline_system:
            if os.name != "nt":
                sys.exit("not on Windows: pass --offline-system to use the built-in OS DLL list")
            self.system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        # case-insensitive index per directory (NTFS is; linux is not)
        self._index: dict[Path, dict[str, Path]] = {}

    def _listing(self, d: Path) -> dict[str, Path]:
        if d not in self._index:
            self._index[d] = ({p.name.lower(): p for p in d.iterdir() if p.is_file()}
                              if d.is_dir() else {})
        return self._index[d]

    def resolve(self, name: str, importer_dir: Path) -> tuple[str, Path | None]:
        """-> (kind, path). kind: package | os | apiset | driver | MISSING."""
        n = name.lower()
        for d in [importer_dir, *self.search]:
            hit = self._listing(d).get(n)
            if hit is not None:
                return "package", hit
        if n.startswith(API_SET_PREFIXES):
            return "apiset", None
        if self.system32 is not None:
            hit = self._listing(self.system32).get(n)
            if hit is not None:
                return "os", hit
        elif n in OS_DLLS:
            return "os", None
        if n in DRIVER_DLLS:
            return "driver", None
        return "MISSING", None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=Path, required=True)
    ap.add_argument("--offline-system", action="store_true",
                    help="no System32 available (linux inspection); use the built-in OS DLL list")
    ap.add_argument("--json", type=Path, help="write the full resolution map here")
    args = ap.parse_args()

    r = Resolver(args.prefix.resolve(), args.offline_system)
    roots = sorted(r.torch_lib.glob("*.dll"))
    if not roots:
        sys.exit(f"no DLLs under {r.torch_lib}")

    seen: dict[Path, dict] = {}
    queue = list(roots)
    missing_static: dict[str, set[str]] = {}   # name -> importers
    missing_delay: dict[str, set[str]] = {}
    driver_delay: dict[str, set[str]] = {}
    while queue:
        dll = queue.pop(0)
        if dll in seen:
            continue
        static, delay = pe_imports(dll)
        rec = {"static": {}, "delay": {}}
        for kind_name, names, sink in (("static", static, missing_static),
                                       ("delay", delay, missing_delay)):
            for n in sorted(names):
                kind, path = r.resolve(n, dll.parent)
                rec[kind_name][n] = {"kind": kind, "path": str(path) if path else None}
                if kind == "MISSING":
                    sink.setdefault(n, set()).add(dll.name)
                elif kind == "driver":
                    (driver_delay if kind_name == "delay" else missing_static
                     ).setdefault(n, set()).add(dll.name)
                elif kind == "package" and path is not None and path not in seen:
                    # walk into package DLLs only: OS DLLs are the OS's problem
                    queue.append(path)
        seen[dll] = rec

    rel = lambda p: os.path.relpath(p, r.prefix)
    print(f"win_dll_audit: prefix={r.prefix}")
    print(f"  roots: {len(roots)} DLLs in {rel(r.torch_lib)}")
    print(f"  walked: {len(seen)} PE files transitively")
    pkg_dlls = sorted({rel(p) for rec in seen.values() for kind in rec.values()
                       for v in kind.values() if v["kind"] == "package" for p in [v["path"]]})
    print(f"  resolved package DLLs ({len(pkg_dlls)}):")
    for p in pkg_dlls:
        print(f"    {p}")
    if driver_delay:
        print("  driver DLLs, delay-loaded (absent without an NVIDIA driver; not fatal at import):")
        for n, who in sorted(driver_delay.items()):
            print(f"    {n}  <- {', '.join(sorted(who))}")
    if missing_delay:
        print("  UNRESOLVED delay-load imports (fatal only when that code path runs):")
        for n, who in sorted(missing_delay.items()):
            print(f"    {n}  <- {', '.join(sorted(who))}")
    if args.json:
        args.json.write_text(json.dumps(
            {str(k): v for k, v in seen.items()}, indent=1, sort_keys=True))
    if missing_static:
        print("  UNRESOLVED static imports -- `import torch` WILL fail:")
        for n, who in sorted(missing_static.items()):
            print(f"    {n}  <- {', '.join(sorted(who))}")
        sys.exit(1)
    print("  every static import resolves")


if __name__ == "__main__":
    main()
