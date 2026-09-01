#!/usr/bin/env python3
"""wheel2conda: repack a PyPI wheel into a conda v2 (.conda) package.

Pure stdlib + zstandard. No conda-build, no rattler-build.

Layout produced (linux-64 / arch package, NOT noarch):
    lib/python3.X/site-packages/<...wheel contents...>
    bin/<console_script>            (with prefix placeholder, file_mode=text)
    info/{index,paths,about,files,has_prefix,...}

Usage:
  wheel2conda.py WHEEL --python 3.12 --subdir linux-64 --build-string ... [--omit FIELD]
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, re, sys, tarfile, time, zipfile
from pathlib import Path, PurePosixPath

import zstandard

# The classic conda "long dummy prefix": 255 chars. Installers rewrite this
# byte-range in-place, so the placeholder must be >= any real install prefix.
_P = "/opt/wheel2conda/_h_env" + "_placehold" * 40
PLACEHOLDER = _P[:255]
assert len(PLACEHOLDER) == 255, len(PLACEHOLDER)

# conda-forge's /bin/sh polyglot shim. Dodges the kernel's 127-byte shebang
# limit, which a 255-char placeholder prefix would blow through instantly.
SHIM = """#!/bin/sh
'''exec' {prefix}/bin/python "$0" "$@"
' '''
import sys
from {module} import {attr}
if __name__ == '__main__':
    sys.argv[0] = sys.argv[0].removesuffix('.exe')
    sys.exit({call})
"""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_entry_points(dist_info: Path) -> dict[str, str]:
    ep = dist_info / "entry_points.txt"
    if not ep.exists():
        return {}
    import configparser
    cp = configparser.ConfigParser()
    cp.optionxform = str
    cp.read_string(ep.read_text())
    if not cp.has_section("console_scripts"):
        return {}
    return dict(cp.items("console_scripts"))


def wheel_metadata(dist_info: Path) -> dict:
    """Pull the handful of METADATA fields we map into conda metadata."""
    out: dict[str, object] = {"requires_dist": []}
    text = (dist_info / "METADATA").read_text(errors="replace")
    for line in text.splitlines():
        if not line.strip():
            break  # headers end at first blank line
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip().lower(), v.strip()
        if k == "requires-dist":
            out["requires_dist"].append(v)
        elif k == "requires-python":
            out["requires_python"] = v
        elif k in ("name", "version", "summary", "license", "home-page"):
            out[k] = v
        elif k == "project-url" and v.lower().startswith(("code,", "homepage,")):
            out.setdefault("home-page", v.split(",", 1)[1].strip())
    return out


PEP508_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*(.*)$")


def pep508_to_conda(req: str) -> str | None:
    """Translate a PEP 508 Requires-Dist into a conda MatchSpec (best-effort).

    Drops anything with an environment marker (extras / platform conditionals),
    which conda has no direct equivalent for.
    """
    if ";" in req:
        head, marker = req.split(";", 1)
        if "extra" in marker:
            return None  # optional dependency -> not a hard conda depends
        req = head
    m = PEP508_RE.match(req)
    if not m:
        return None
    name, spec = m.group(1).lower().replace("_", "-"), m.group(2).strip()
    spec = spec.strip("()").strip()
    if not spec:
        return name
    # conda uses ',' the same way; '===' and '~=' need care. Keep it simple.
    spec = spec.replace("~=", ">=")
    return f"{name} {spec}"


def build(
    wheel: Path,
    outdir: Path,
    python_version: str,
    subdir: str,
    build_string: str | None,
    build_number: int,
    version_override: str | None,
    name_override: str | None,
    extra_depends: list[str],
    omit: set[str],
    include_deps: bool,
) -> Path:
    stage = outdir / "_stage"
    if stage.exists():
        import shutil
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # ---- 1. unpack wheel into the site-packages location -------------------
    if subdir.startswith("win-"):
        sp_rel = PurePosixPath("Lib/site-packages")           # Windows convention
    else:
        sp_rel = PurePosixPath(f"lib/python{python_version}/site-packages")
    sp = stage / sp_rel
    sp.mkdir(parents=True)

    zf = zipfile.ZipFile(wheel)
    data_dirs = []
    for zi in zf.infolist():
        if zi.is_dir():
            continue
        parts = PurePosixPath(zi.filename).parts
        # wheel .data/ subdirs map to prefix-relative locations, not site-packages
        if len(parts) >= 2 and parts[0].endswith(".data"):
            data_dirs.append(zi.filename)
            scheme = parts[1]  # scripts, data, purelib, platlib, headers
            rest = PurePosixPath(*parts[2:])
            if scheme in ("purelib", "platlib"):
                dest = stage / sp_rel / rest
            elif scheme == "scripts":
                dest = stage / "bin" / rest
            elif scheme == "data":
                dest = stage / rest
            else:
                dest = stage / sp_rel / rest
        else:
            dest = stage / sp_rel / zi.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(zi) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        # preserve the executable bit from the zip external attrs
        mode = (zi.external_attr >> 16) & 0o7777
        if mode:
            os.chmod(dest, mode)

    dist_info = next(sp.glob("*.dist-info"))
    meta = wheel_metadata(dist_info)
    name = name_override or meta["name"].lower().replace("_", "-").replace(".", "-")
    version = version_override or meta["version"]

    # ---- 2. version string trap -------------------------------------------
    # conda versions may not contain '+' (PEP440 local version segment).
    if "+" in version:
        raise SystemExit(
            f"REFUSING: conda version cannot contain '+': {version!r}\n"
            f"  Move the local segment into the build string instead."
        )

    # ---- 3. entry points -> real bin/ scripts (arch pkg, not noarch) -------
    eps = parse_entry_points(dist_info)
    prefix_files: dict[str, str] = {}          # relpath -> file_mode
    for script, target in eps.items():
        module, _, attr = target.partition(":")
        attr_root = attr.split(".")[0] if attr else "main"
        call = attr + "()" if attr else "main()"
        content = SHIM.format(prefix=PLACEHOLDER, module=module, attr=attr_root, call=call)
        bindir = stage / ("Scripts" if subdir.startswith("win-") else "bin")
        bindir.mkdir(parents=True, exist_ok=True)
        p = bindir / script
        p.write_text(content)
        os.chmod(p, 0o755)
        prefix_files[str(p.relative_to(stage))] = "text"

    # ---- 4. compute paths.json / info/files -------------------------------
    files: list[str] = []
    for root, dirs, names in os.walk(stage):
        dirs.sort()
        for n in sorted(names):
            files.append(str(Path(root, n).relative_to(stage)))
    files.sort()

    paths = []
    for rel in files:
        p = stage / rel
        entry = {
            "_path": rel,
            "path_type": "hardlink",
            "sha256": sha256_file(p),
            "size_in_bytes": p.stat().st_size,
        }
        if rel in prefix_files:
            entry["file_mode"] = prefix_files[rel]
            entry["prefix_placeholder"] = PLACEHOLDER
        paths.append(entry)
    paths.sort(key=lambda e: e["_path"])

    # ---- 5. index.json ----------------------------------------------------
    pytag = "py" + python_version.replace(".", "")
    if build_string is None:
        build_string = f"{pytag}_repack_{build_number}"

    depends = [f"python >={python_version},<{_next_minor(python_version)}.0a0",
               f"python_abi {python_version}.* *_cp{python_version.replace('.','')}"]
    if include_deps:
        for r in meta["requires_dist"]:
            c = pep508_to_conda(r)
            if c:
                depends.append(c)
    depends += extra_depends
    depends = sorted(set(depends))

    arch, platform = _arch_platform(subdir)
    index = {
        "arch": arch,
        "build": build_string,
        "build_number": build_number,
        "depends": depends,
        "license": meta.get("license", "") or "",
        "name": name,
        "platform": platform,
        "subdir": subdir,
        "timestamp": int(time.time() * 1000),
        "version": version,
    }
    for f in omit:
        index.pop(f, None)

    about = {
        "home": meta.get("home-page", ""),
        "license": meta.get("license", ""),
        "summary": meta.get("summary", ""),
        "description": f"Repacked from PyPI wheel {wheel.name}",
        "extra": {"repacked_from_wheel": wheel.name, "tool": "wheel2conda.py"},
    }

    # ---- 6. assemble the info/ tree ---------------------------------------
    infodir = outdir / "_info"
    if infodir.exists():
        import shutil
        shutil.rmtree(infodir)
    (infodir / "info").mkdir(parents=True)
    W = lambda n, s: (infodir / "info" / n).write_text(s)
    W("index.json", json.dumps(index, indent=2, sort_keys=True))
    W("paths.json", json.dumps({"paths": paths, "paths_version": 1}, indent=2))
    W("about.json", json.dumps(about, indent=2, sort_keys=True))
    W("files", "".join(f + "\n" for f in files))
    if prefix_files:
        W("has_prefix", "".join(
            f"{PLACEHOLDER} {mode} {rel}\n" for rel, mode in sorted(prefix_files.items())))

    # ---- 7. write the .conda (ZIP of two zstd tars) -----------------------
    stem = f"{name}-{version}-{build_string}"
    out = outdir / f"{stem}.conda"
    # torch's payload is ~1.7 GB uncompressed, so never hold a tar in memory:
    # stream files -> tar -> zstd -> a temp file on disk, then store into the zip.
    cctx = zstandard.ZstdCompressor(level=19, threads=-1)

    def make_tar_zst(base: Path, members: list[str], dest: Path) -> Path:
        with open(dest, "wb") as raw, cctx.stream_writer(raw) as zw, \
                tarfile.open(fileobj=zw, mode="w|", format=tarfile.GNU_FORMAT) as tf:
            for rel in members:
                p = base / rel
                ti = tf.gettarinfo(str(p), arcname=rel)
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                ti.mtime = 0
                with open(p, "rb") as fh:
                    tf.addfile(ti, fh)
        return dest

    for drop in os.environ.get("W2C_DROP_INFO", "").split(","):
        if drop.strip():
            f = infodir / "info" / drop.strip()
            if f.exists():
                f.unlink()
                print(f"  [ablation] dropped info/{drop.strip()}")

    info_members = sorted(
        str(p.relative_to(infodir)) for p in infodir.rglob("*") if p.is_file())
    tmp = outdir / "_tmp"
    tmp.mkdir(exist_ok=True)
    pkg_blob = make_tar_zst(stage, files, tmp / "pkg.tar.zst")
    info_blob = make_tar_zst(infodir, info_members, tmp / "info.tar.zst")

    # allowZip64 matters: a torch payload member can exceed the 4 GiB ZIP32 limit.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        z.writestr("metadata.json", json.dumps({"conda_pkg_format_version": 2}))
        z.write(pkg_blob, f"pkg-{stem}.tar.zst")
        z.write(info_blob, f"info-{stem}.tar.zst")
    import shutil
    shutil.rmtree(tmp)

    print(f"built {out}  ({out.stat().st_size} bytes, {len(files)} files)")
    print(f"  build string : {build_string}")
    print(f"  depends      : {depends}")
    print(f"  entry points : {list(eps)}")
    return out


def _next_minor(v: str) -> str:
    a, b = v.split(".")
    return f"{a}.{int(b)+1}"


def _arch_platform(subdir: str) -> tuple[str, str]:
    plat, _, a = subdir.partition("-")
    return {"64": "x86_64", "aarch64": "aarch64", "arm64": "arm64"}.get(a, a), \
           {"linux": "linux", "osx": "osx", "win": "win"}.get(plat, plat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", type=Path)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("."))
    ap.add_argument("--python", default="3.12")
    ap.add_argument("--subdir", default="linux-64")
    ap.add_argument("--build-string", default=None)
    ap.add_argument("--build-number", type=int, default=0)
    ap.add_argument("--version", dest="version_override", default=None)
    ap.add_argument("--name", dest="name_override", default=None)
    ap.add_argument("--depends", action="append", default=[])
    ap.add_argument("--omit", action="append", default=[],
                    help="omit a field from index.json (for ablation testing)")
    ap.add_argument("--no-deps", action="store_true")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    build(a.wheel, a.outdir, a.python, a.subdir, a.build_string, a.build_number,
          a.version_override, a.name_override, a.depends, set(a.omit),
          include_deps=not a.no_deps)


if __name__ == "__main__":
    main()
