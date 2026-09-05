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

### Selecting a flavour

Depend on the selector metapackage for your host's CUDA minor and nothing
else:

```toml
[dependencies]
pytorch-cuda128 = "*"        # any torch version, CUDA 12.8 builds only
```

Selectors exist per flavour (`pytorch-cuda124` … `pytorch-cuda132`) and pin
`pytorch * cuda<NNN>_*`, matching every build convention in the channel
(`_repack_`, `_mkl_`, `_generic_`). **Without a selector (or an explicit
build glob), the solver may legally pick a different CUDA minor** — conda's
`cuda-version` machinery only constrains the major, and build-number
tiebreaks decide the rest. That's conda semantics, not a channel bug; use
the selectors.

Two more rules that are load-bearing:

- **Channel order matters.** This channel first, conda-forge second. The
  reversed order shadows `pytorch` behind conda-forge's coverage holes
  under strict channel priority.
- **Drop `defaults`.** Anaconda's own `main` channel (present by default in
  most Miniconda/Anaconda installs) ships its *own* actively-maintained
  `pytorch`, `libtorch`, `triton`, and `libcudnn` builds under a third
  build-string convention (`gpu_cuda128_h…`). The selector metapackages
  don't match them, but a bare `pytorch = "*"` with `defaults` at equal or
  higher priority silently resolves to Anaconda's build instead of this
  channel's or conda-forge's. Use `nodefaults` (conda) or simply omit
  `defaults` from `channels` (pixi never adds it).
- **Metadata patches are forward-only.** A fix applied via `patches/`
  protects every *future* solve; it cannot reach a `pixi.lock` that already
  pins a bad build (`pixi install --locked` never re-reads repodata), nor a
  manifest that pins an exact build string. `known_bad.json` lists every
  such build and its fixed replacement; `tools/check_lock.py` scans a
  lockfile against it.
- **Client floor**: pixi ≥ 0.40, conda ≥ 24.5, mamba/micromamba ≥ 2.0.
  mamba/micromamba 1.x solves but then fails at fetch with a 404 (no
  CEP-15 `base_url` support); condas predating repodata_version 2 reject
  the channel outright.

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
- `tools/sweep_solve.py` — verification: one live `pixi lock` per grid entry,
  asserting the newest build resolves from this channel's release URL.
- `tools/check_lock.py` — scans a lockfile against `known_bad.json`.

## Channel status (2026-09-02)

The full grid is live: every PyPI torch flavour (cu124+, latest patch per
minor line, torch 2.4.1–2.14.0) on linux-64, linux-aarch64, and win-64 —
~950 packages, ~117 GiB, all on free GitHub infrastructure. Mirrored
conda-forge artifacts are byte-identical (their sha256s); repacked entries
are CI-built from PyPI wheels and structurally verified; linux-64 spot
checks ran real GPU compute through conda-forge CUDA libs.

Known caveats, named rather than hidden:

- **13 upstream holes**: wheel combos PyPI itself never published (listed in
  `grid/README.md`), e.g. no aarch64 cu128 wheel for 2.8.0.
- **2.7.1 cu129 linux-aarch64**: conda-forge's own build no longer solves
  (their libcudss→libcudss0 migration broke it); no PyPI wheel exists either.
  Mirrored for the record, effectively dead upstream-wide.
- **2.10.0/2.11.0/2.12.1 cu129 linux-aarch64**: same conda-forge bitrot, but
  PyPI wheels exist — `cuda129_repack` builds cover these cells.
- **py3.15 records** exist but cannot solve until conda-forge ships python
  3.15 (self-heals; final lands next month).
- **2.11.0 cu128 / 2.13.0 cu129**: use the `_1` builds (build-number
  tiebreaker picks them automatically); `_0` pinned a libcudnn that
  conda-forge only built for CUDA 13 and can never install. The channel
  carries `nvidia-cudnn` side-repacks for these.
- Strict-channel-priority rule for maintainers: any package *name* this
  channel carries must be carried completely (every version any dependent
  pins) — one partial name shadows all of conda-forge's copies.

## Metadata patches (patches/ overlay)

`patches/<subdir>/patches.json` maps an exact `.conda` filename to a
partial repodata override (allowed keys: `depends`, `constrains`,
`purls`). `make_repodata.py` applies the overlay after loading fragments
and before writing `site/`, so **published artifacts and their fragments
stay immutable while served metadata stays fixable** — the same mechanism
conda-forge uses (repodata patches). A patch naming a filename with no
fragment is a hard error, so typos cannot silently no-op.

When to patch vs republish: metadata-only fixes (a wrong bound, a missing
purl) get a patch; anything touching bytes gets a republished artifact
with a bumped build number.
