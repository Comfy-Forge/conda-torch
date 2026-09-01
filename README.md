# conda-torch

A conda channel carrying **every PyPI-published torch flavour**, repackaged
from the official PyPI wheels with honest conda metadata. Exists because
conda-forge structurally cannot carry them: its policy builds only the latest
CUDA minor per release, leaving 56% of the (torch × CUDA-flavour × OS) grid
with no conda package (measured 2026-09; see the torch-conda pages in
andrea-personal-docs).

## Using the channel

```toml
[workspace]
channels = ["https://comfy-forge.github.io/conda-torch", "conda-forge"]
```

Flavour is encoded in the **build string**, never the version (conda orders
`2.8.0+cu128` *below* `2.8.0`, so local versions would break `>=` specs):

```
pytorch 2.8.0 cuda128_repack_py312_h<hash>_0
```

## Architecture

Static channel, zero servers, everything free-tier GitHub:

- **Packages** are GitHub **release assets**: one release per subdir
  (`linux-64`, `win-64`, `linux-aarch64`, `noarch`), 2 GB/file cap, no total
  cap, CDN-served.
- **repodata.json** is served by GitHub **Pages** and points at the release
  via CEP-15 `info.base_url` (supported by pixi/rattler; verified
  empirically before this repo was created).
- **Fragments, not re-indexing**: each publish job commits
  `meta/<subdir>/<filename>.conda.json` (the package's `index.json` +
  sha256/md5/size). `tools/make_repodata.py` assembles `site/` from those,
  so regenerating the channel never downloads a package.

```
repack job ──> .conda ──> gh release upload (tag = subdir)
          └──> tools/fragment.py ──> meta/<subdir>/*.json ──(commit)──> main
push to main ──> publish-channel.yml ──> make_repodata.py ──> Pages
```

## Package design (from the five-hacker investigation, 2026-09)

Split at conda-forge's line — a shared `libtorch-*` per (version, flavour,
platform) plus a thin (~17 MB) per-python `pytorch` shim; 74% smaller than
fat per-python packages at 4 pythons. CUDA comes from conda-forge's own
`cuda-*`/`lib*` packages (route (b)), which cover every needed soname on
every platform except `nvshmem` on aarch64 (repacked separately).

Non-negotiables, each one empirically earned:

- linux: `patchelf --force-rpath --add-rpath '$ORIGIN/../../../..'` on the
  seven CUDA-linked libs (`--force-rpath` or patchelf silently downgrades
  DT_RPATH→DT_RUNPATH and `LD_LIBRARY_PATH` can hijack CUDA resolution).
  Copy before patching — patchelf writes through hardlinks.
- Package name is `pytorch` (mutex with conda-forge's; `torch` would
  co-install and clobber `site-packages/torch/`).
- dist-info ships **without RECORD** (pip's uninstaller honours RECORD
  regardless of INSTALLER — shipping it lets `pip uninstall torch` delete
  conda-owned files) and without `direct_url.json`; `INSTALLER` = `conda`.
- `cuda-version >=MAJOR.MINOR,<MAJOR+1` (exact-minor pins are UNSAT:
  conda-forge triton only exists for one CUDA minor per version).
- `torch/bin` and `torch/lib` stay inside the package as real paths
  (`import torch` hard-fails if `torch/bin/torch_shm_manager` moves);
  symlink into the package, never out of it.
- Byte-compile `.pyc` with the target interpreter and record in
  `paths.json`; enumerate files from the wheel zip, never `top_level.txt`.
- rattler-build (if used) needs `binary_relocation: false` or it silently
  rewrites RPATHs in every binary.

## Tools

- `tools/wheel2conda.py`, `tools/wheel2conda_plus.py` — wheel → `.conda`
  assembler (proof-of-concept lineage; the torch-hardened pipeline lands on
  top of these).
- `tools/fragment.py` — `.conda` → repodata fragment under `meta/`.
- `tools/make_repodata.py` — fragments → `site/` (repodata + index page).
