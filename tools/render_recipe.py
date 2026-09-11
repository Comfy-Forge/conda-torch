#!/usr/bin/env python3
"""Render a rattler-build recipe for one (version, flavour, python, subdir) cell.

rattler-build needs `depends` to be static in the recipe, but ours are computed
from the extracted wheel: dependency translation from METADATA, the CUDA-window
satisfiability fixups, ELF-derived glibc/libstdc++ floors, whether the sleef
redirect applies, and which vendored CUDA libraries were stripped. So the recipe
is generated per cell rather than written by hand -- the same shape as
conda-cuda-packages' generate_recipes.py.

That costs one extra wheel download and one extra pass of the surgery: this
renderer runs the real thing to learn the metadata, then rattler-build runs it
again to produce the artifact from its own verified source. The alternative --
handing rattler-build the tree this pass already staged -- would make
`source: sha256:` decorative, since the artifact would no longer derive from the
bytes that were verified. The duplicated work is the price of an honest chain.

Usage:
  render_recipe.py --version 2.8.0 --flavour cu128 --python 3.12 \
                   --subdir linux-64 --build-number 5 --out recipes/<cell>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch_repack as tr  # noqa: E402  (path set above)


def py_spec(py: str) -> str:
    """Interpreter spec for the BUILD platform.

    `python 3.15.*` does not match 3.15.0rc2, and until a python lands
    conda-forge ships only release candidates -- so every py3.15 cell failed
    to resolve. .pyc magic is frozen at RC, which is the only property we need
    from the interpreter here, so spelling the floor as an rc admits the RC
    without loosening anything else. The old pipeline got this for free from
    setup-python's allow-prereleases; moving the interpreter into the build
    environment is what made it explicit.
    """
    major, minor = py.split(".")
    return f"python >={py}.0rc0,<{major}.{int(minor) + 1}.0a0"

# The two outputs partition one $PREFIX. Derived from split_linux()/main():
# the shim, the site-packages tree and the POSIX entry point are pytorch's;
# everything else the surgery writes -- big libs, the private vendored dir,
# headers, cmake, the activation scripts, licenses -- is libtorch's.
# The partition is derived from meta["lt_owned"], which the surgery records at
# each move: libtorch takes exactly what it was given, pytorch takes everything
# else the build created. A prefix glob cannot express this -- on win-64
# nothing is relocated, so libtorch's files sit interleaved inside
# Lib/site-packages/torch/ alongside pytorch's.


def render(meta: dict, *, wheel_url: str, wheel_sha: str, subdir: str,
           python: str, hash_source: str, tool_sha: str = "") -> str:
    lt, pt = meta["libtorch"], meta["pytorch"]
    lt_i, pt_i = lt["index"], pt["index"]
    version = lt_i["version"]

    def norm(spec: str) -> str:
        """rattler-build's parser rejects conda's bare `name VERSION BUILD`
        form (`pytorch 2.8.0 cuda128_repack_*`) and demands an explicit
        operator. Semantically identical, but it means the recipe cannot carry
        the exact string the hand-built artifacts do -- see the equivalence
        report."""
        parts = spec.split()
        # only a wildcard-free literal version needs the operator: a glob
        # like `3.12.*` is already a valid range and `==3.12.*` is rejected.
        if (len(parts) == 3 and parts[1][:1].isdigit()
                and "*" not in parts[1] and parts[1][0] not in "<>=!~"):
            return f"{parts[0]} =={parts[1]} {parts[2]}"
        return spec

    def dep_list(deps: list[str], indent: int) -> str:
        pad = " " * indent
        return "".join(f"\n{pad}- {json.dumps(norm(d))}" for d in deps) or " []"

    lt_owned = meta["lt_owned"]
    # pytorch's own libtorch dep is replaced by pin_subpackage(exact) so the
    # link is expressed in rattler's own terms rather than a string we format.
    pt_deps = [d for d in pt_i["depends"] if not d.startswith("libtorch ")]

    # about.extra: provenance the artifact must carry. The renderer records
    # how the wheel's hash was established, which `source: sha256:` alone
    # cannot say -- a pinned hash proves the bytes are unchanged, not that
    # upstream ever published one.
    # rattler-build 0.75 copies one top-level `extra:` into every output and
    # supports no per-output override, so only genuinely shared provenance
    # goes here. libtorch's vendored-component census ships as a payload file
    # (share/libtorch/vendored-sbom.json) rather than being duplicated onto
    # pytorch, where it would describe binaries that package does not contain.
    shared = pt["about"].get("extra", {})
    extra = {k: shared[k] for k in ("repacked_from", "wheel_sha256") if k in shared}
    extra.update(wheel_hash_source=hash_source, built_by="rattler-build",
                 source_url=wheel_url)
    # provenance() used to be stamped by emit_conda, which rattler-build
    # replaces; carry it explicitly or the artifact loses its build trail.
    extra.update(tr.provenance())

    owned = "\n".join(f"          - {json.dumps(g)}" for g in lt_owned)
    lic_files = "".join(f"\n        - {json.dumps(n)}"
                        for n in sorted(meta["licenses"])) or " []"
    # libtorch may carry texts the wheel does not (the NVIDIA EULA for the
    # DLLs vendored on win-64); pytorch lists only the wheel's own.
    lt_lic_files = "".join(f"\n        - {json.dumps(n)}"
                           for n in sorted(meta.get("lt_licenses", meta["licenses"]))) or " []"
    return f"""# GENERATED by tools/render_recipe.py -- do not edit.
# cell: torch {version} {lt_i['build'].split('_')[0]} py{python} {subdir}
recipe:
  name: torch-repack
  version: {json.dumps(version)}

source:
  # Structural verification: rattler-build hard-fails on a hash mismatch.
  # Never omit sha256 -- an absent hash downloads silently with no warning,
  # which is strictly worse than the check this replaces.
  url: {json.dumps(wheel_url)}
  sha256: {json.dumps(wheel_sha)}

outputs:
  - staging:
      name: torch-repack-staging
    build:
      script:
        content: |
          set -euo pipefail
          # The surgery runs against rattler-build's own verified download.
          # the BUILD platform's interpreter, not $PREFIX/bin/python: a
          # win-64 or aarch64 host env cannot execute here, and .pyc magic is
          # a property of the python VERSION, not the platform, so a linux
          # 3.12 produces the .pyc a win 3.12 will load. Keeping python out of
          # host also stops its run_export adding python_abi to libtorch.
          "$BUILD_PREFIX/bin/python" "$RECIPE_DIR/torch_repack.py" \\
            --version {version} --flavour {lt_i['build'].split('_')[0].replace('cuda', 'cu')} \\
            --python {python} --subdir {subdir} \\
            --build-number {lt_i['build_number']} \\
            --wheel "$SRC_DIR"/*.whl --self-sha256 {tool_sha} \\
            --stage-prefix "$PREFIX" --work "$SRC_DIR/_work"
    requirements:
      build:
        # everything the surgery needs runs on the BUILD platform: the
        # interpreter that byte-compiles, its zstandard (conda-forge artifacts
        # are read for the sleef gate and the win DLL audit), pefile for the
        # win PE import table, and patchelf, which edits aarch64 ELFs from an
        # x86 host perfectly well since it never executes them.
        - {py_spec(python)}
        - zstandard
        - pefile
        - patchelf

  - package:
      name: libtorch
      version: {json.dumps(version)}
    inherit: torch-repack-staging
    build:
      number: {lt_i['build_number']}
      string: {json.dumps(lt_i['build'])}
      # Byte-identity with the upstream wheel is the point of this channel:
      # rattler-build's default post-processing rewrites RPATHs with patchelf,
      # which would silently undo the deliberate --force-rpath layout.
      dynamic_linking:
        binary_relocation: false
      files:
        include:
{owned}
    requirements:
      run:{dep_list(lt_i['depends'], 8)}
      run_constraints:{dep_list(lt_i.get('constrains', []), 8)}
      run_exports:
        weak:{dep_list(lt["run_exports"]["weak"], 10)}
    about:
      homepage: https://pytorch.org
      license: {json.dumps(lt_i['license'])}
      license_family: {json.dumps(lt_i.get('license_family', 'BSD'))}
      summary: {json.dumps(lt["about"]["summary"])}
      license_file:{lt_lic_files}

  - package:
      name: pytorch
      version: {json.dumps(version)}
    inherit: torch-repack-staging
    build:
      number: {pt_i['build_number']}
      string: {json.dumps(pt_i['build'])}
      dynamic_linking:
        binary_relocation: false
      files:
        include:
          - "**"
        exclude:
{owned}
    requirements:
      run:{dep_list(pt_deps, 8)}
        - ${{{{ pin_subpackage('libtorch', exact=True) }}}}
      run_exports:
        weak:{dep_list(pt["run_exports"]["weak"], 10)}
    about:
      homepage: https://pytorch.org
      license: {json.dumps(pt_i['license'])}
      license_family: {json.dumps(pt_i.get('license_family', 'BSD'))}
      summary: {json.dumps(pt["about"]["summary"])}
      license_file:{lic_files}

extra:{_extra_yaml(extra, 2)}
"""


def _extra_yaml(extra: dict, indent: int) -> str:
    """about.extra as YAML. Values are JSON-encoded, so nested structures
    (the vendored SBOM) survive without a YAML emitter dependency."""
    pad = " " * indent
    out = []
    for k, v in sorted(extra.items()):
        if isinstance(v, (dict, list)):
            out.append(f"\n{pad}{k}: {json.dumps(v)}")
        else:
            out.append(f"\n{pad}{k}: {json.dumps(v)}")
    return "".join(out) or " {}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--flavour", required=True)
    ap.add_argument("--python", dest="py", required=True)
    ap.add_argument("--subdir", default="linux-64", choices=sorted(tr.PLATFORMS))
    ap.add_argument("--build-number", type=int, default=0)
    ap.add_argument("--side-build-number", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="recipe directory")
    ap.add_argument("--work", type=Path, default=Path("render_work"))
    ap.add_argument("--wheel", type=Path, default=None)
    ap.add_argument("--check", action="store_true",
                    help="fail if the rendered recipe or its torch_repack.py "
                         "snapshot differs from what is already in --out")
    args = ap.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    url, idx_sha = tr.wheel_url(args.version, args.flavour, args.py, args.subdir)
    gsha = tr.grid_wheel_sha(args.version, args.flavour, args.subdir, args.py)
    if idx_sha and gsha and idx_sha != gsha:
        sys.exit(f"index sha256 {idx_sha} != grid.json {gsha}: refusing")
    expect = idx_sha or gsha
    hash_source = ("index+grid" if idx_sha and gsha else "index" if idx_sha
                   else "grid" if gsha else "TOFU")

    wheel = args.wheel
    if wheel is None:
        wheel = args.work / url.split("#")[0].rsplit("/", 1)[-1].replace("%2B", "+")
        tr.download(url, wheel, expect)
    sha = tr.sha256_file(wheel)
    if expect and sha != expect:
        sys.exit(f"wheel sha256 {sha} != expected {expect}")
    if hash_source == "TOFU":
        # Pin the bytes we actually observed. rattler-build will then fail if
        # they ever change -- weaker than an upstream anchor, but far stronger
        # than omitting the hash, which downloads silently unverified.
        tr.log(f"no upstream sha256 for this wheel: pinning observed {sha} (TOFU)")

    meta_path = args.work / "meta.json"
    prefix = args.work / "prefix"
    if prefix.exists():
        shutil.rmtree(prefix)
    subprocess.run([sys.executable, str(HERE / "torch_repack.py"),
                    "--version", args.version, "--flavour", args.flavour,
                    "--python", args.py, "--subdir", args.subdir,
                    "--build-number", str(args.build_number),
                    "--side-build-number", str(args.side_build_number),
                    "--wheel", str(wheel), "--stage-prefix", str(prefix),
                    "--emit-metadata", str(meta_path),
                    "--work", str(args.work / "surgery")], check=True)

    meta = json.loads(meta_path.read_text())
    tool_src = (HERE / "torch_repack.py").read_bytes()
    # The script lives in the recipe DIR, which rattler-build does not fold
    # into its staging-cache key: a changed torch_repack.py with unchanged
    # deps restored a stale 13526-file tree and silently packaged it
    # (measured). Threading its hash through the build command puts it inside
    # the recipe text, so the key moves whenever the surgery does -- and the
    # script re-checks it at run time, so a swapped snapshot fails closed.
    tool_sha = hashlib.sha256(tool_src).hexdigest()
    text = render(meta, tool_sha=tool_sha, wheel_url=url.split("#")[0], wheel_sha=sha,
                  subdir=args.subdir, python=args.py, hash_source=hash_source)
    if args.check:
        # The recipe dir carries a SNAPSHOT of torch_repack.py, so editing the
        # tool without re-rendering silently builds stale code -- it already
        # cost one confusing debug cycle. CI runs this so it cannot reach a
        # fleet wave.
        drift = []
        cur = args.out / "recipe.yaml"
        if not cur.is_file() or cur.read_text() != text:
            drift.append("recipe.yaml")
        snap = args.out / "torch_repack.py"
        if not snap.is_file() or snap.read_bytes() != tool_src:
            drift.append("torch_repack.py")
        if drift:
            sys.exit(f"render drift in {args.out}: {', '.join(drift)} "
                     f"differ(s) from a fresh render -- re-run render_recipe.py")
        tr.log(f"no render drift in {args.out}")
        return

    (args.out / "recipe.yaml").write_text(text)
    (args.out / "torch_repack.py").write_bytes(tool_src)
    # The license gate substitutes canonical SPDX texts for wheels that
    # declare a license but ship no copy, and it resolves them next to
    # torch_repack.py -- so the recipe dir needs them beside the snapshot or
    # the gate fails inside the build (measured on triton, MIT).
    lic_dir = args.out / "licenses"
    lic_dir.mkdir(exist_ok=True)
    for src in sorted((HERE / "licenses").glob("*.txt")):
        shutil.copyfile(src, lic_dir / src.name)
    # info/licenses/ comes from about.license_file, which resolves against the
    # recipe directory -- so the wheel's license blobs are laid down beside it.
    for flat, blob in sorted(tr.wheel_license_files(wheel).items()):
        (args.out / flat).write_bytes(blob)
    # texts the surgery fetched from elsewhere (NVIDIA EULA for vendored DLLs)
    for flat, text in sorted(meta.get("extra_license_blobs", {}).items()):
        (args.out / flat).write_text(text)
    tr.log(f"recipe -> {args.out / 'recipe.yaml'}")


if __name__ == "__main__":
    main()
