# The grid: full mirror manifest

Generated 2026-09-02 from fresh download.pytorch.org/whl/<flavour>/torch/ listings and conda-forge repodata (all three subdirs). Machine-readable form: [`grid.json`](grid.json).

Rules: torch >= 2.4, cu124+, latest patch per (flavour, minor line), CUDA minor must match exactly; freethreaded (t) ABIs excluded. Python axis: defined by PyPI wheels for PyPI entries; by conda-forge for conda-only.

## Scoreboard

| | count |
|---|---:|
| grid entries (version x flavour x platform, PyPI-defined) | **87** |
| ... of which fully mirrorable from conda-forge | 24 |
| ... needing repack (fully or mixed) | 63 |
| conda-only entries (no PyPI wheel; bonus mirrors) | 8 |
| holes (neither world has it; named below) | 13 |
| per-python records: mirror / repack | 151 / 339 |
| **.conda files, core channel** | **551** |
| = repacked shims 286 + repacked libtorch 63 + mirrored pytorch 164 + mirrored libtorch 38 (2026-09-02 basis; the aarch64 re-enumeration below moved 53 aarch64 records from mirror-only to repack, so the repack side is larger and the file total higher) | |
| .conda files incl. conda-only bonus | 599 |
| estimated bytes: mirrored (actual sizes) | 23.6 GiB |
| estimated bytes: repacked (508 MB linux anchor; cf means win/aarch64) | 36.2 GiB |
| estimated bytes: conda-only bonus | 5.4 GiB |
| **estimated channel total** | **65.1 GiB** |

Repacked-libtorch size assumptions: linux-64 485 MiB, linux-aarch64 546 MiB, win-64 418 MiB (linux-64 measured from the pilot; others = mean of conda-forge's cuda libtorch artifacts on that subdir).

## Per platform

| platform | entries | mirror recs | repack recs | repack libtorch | holes |
|---|---:|---:|---:|---:|---:|
| linux-64 | 34 | 68 | 109 | 24 | 2 |
| linux-aarch64 | 31 | 15 | 146 | 28 | 5 |
| win-64 | 30 | 68 | 84 | 18 | 6 |

Counts are recomputed from `grid.json` (2026-09-13); the hand-written 2026-09-02
figures under-counted mirror records on every platform.

## Drift vs the 2026-08 coverage table

One change only: **torch 2.14.0 landed on PyPI** (cu126/cu130/cu132, all three platforms, py3.10-3.15, conda-forge has none of it) — 9 new all-repack entries. No existing mark flipped.

Excluded from the grid, by name, not silently: 176 freethreaded (`cp31xt`) wheels across the indexes — a python-ABI axis to add later, not a flavour/platform gap.

## linux-aarch64: PyPI enumeration, 2026-09-13

Re-enumerated from the flavour indexes themselves (`download.pytorch.org/whl/cu124/torch/`
… `/whl/cu132/torch/`, plus the default `/whl/torch/`): every
`torch-<ver>[+<flavour>]-cp3XX-cp3XX-(manylinux_2_28|linux)_aarch64.whl` inside the policy
window (torch >= 2.4, cu124+, latest patch per (flavour, minor line), freethreaded ABIs
excluded), diffed against this channel's `linux-aarch64` repodata counting **`_repack_`
builds only** — a conda-forge mirror in the same cell does not count as coverage.

<!-- AARCH64-TABLE -->

Reading notes:

- The default index carries no aarch64 wheel a flavour index does not. The 2.4.x/2.5.x
  aarch64 files have no `+cuNN` local tag at all (the directory is the only statement of
  flavour, which is why `wheel_url()` falls back to the tag-less name and
  `torch/version.py` is checked after extraction).
- **Double coverage is now the aarch64 norm.** Where a row of the grid tables says
  "repack, cf mirror also hosted", both artifacts are served. conda-forge's build carries
  a far higher build number (`_2xx`) than any repack (`_0`…`_4`), so an *unpinned* solve
  still lands on the mirror; pin `pytorch * cuda1NN_repack_*` (or the flavour selector plus
  an explicit build glob) to get the repack. For **cu129** that preference is a trap — the
  conda-forge libcudss→libcudss0 migration left those mirrors unsatisfiable, and they are
  listed in `known_bad.json`. For cu126/cu130 the mirror is healthy and the repack is an
  alternative, not a rescue.
- `2.9.0 cu128 py3.12` exists on the channel as a repack (libtorch `_0`…`_4`, one shim) but
  is **outside** the window — 2.9.1 is the latest patch of that line. It was left alone
  rather than extended to the other pythons.
- aarch64 holes re-verified against the live indexes on 2026-09-13: PyPI still publishes no
  aarch64 wheel for cu124 2.6.0, cu126 2.8.0, or cu128 2.8.0 / 2.12.1 / 2.13.0. The five
  aarch64 rows of the hole table below stand.
- The sleef gate's three-way branch is load-bearing on this subdir and both branches are
  live: cu124 2.4.1/2.5.1 reference no `Sleef_*` symbol at all (NVPL/ACL math) and ship
  neither the redirect nor the dependency, while the cu130 2.11.0 build repacked in this
  pass does reference sleef and gets `sleef >=3.9.0,<4.0a0` plus the
  `libtorch_global_deps.so` `DT_NEEDED`. Nothing here was special-cased per flavour.

## Holes (no PyPI wheel AND no exact-minor conda-forge build)

| flavour | platform | version |
|---|---|---|
| cu124 | linux-aarch64 | 2.6.0 |
| cu126 | linux-aarch64 | 2.8.0 |
| cu128 | linux-aarch64 | 2.8.0 |
| cu128 | linux-64 | 2.12.1 |
| cu128 | linux-aarch64 | 2.12.1 |
| cu128 | linux-64 | 2.13.0 |
| cu128 | linux-aarch64 | 2.13.0 |
| cu129 | win-64 | 2.7.1 |
| cu129 | win-64 | 2.9.1 |
| cu129 | win-64 | 2.10.0 |
| cu129 | win-64 | 2.11.0 |
| cu129 | win-64 | 2.12.1 |
| cu129 | win-64 | 2.13.0 |

## The grid

Per python: `M` mirror from conda-forge, `R` repack from the PyPI wheel, `R*` repack + donates the shared libtorch payload, `P` already published on this channel.

### cu124

| version | platform | action | pythons |
|---|---|---|---|
| 2.4.1 | linux-64 | repack, libtorch pending | 3.8:R* 3.9:R 3.10:R 3.11:R 3.12:R |
| 2.4.1 | linux-aarch64 | repack, libtorch pending | 3.8:R* 3.9:R 3.10:R 3.11:R 3.12:R |
| 2.4.1 | win-64 | repack, libtorch pending | 3.8:R* 3.9:R 3.10:R 3.11:R 3.12:R |
| 2.5.1 | linux-64 | repack, libtorch pending | 3.9:R* 3.10:R 3.11:R 3.12:R 3.13:R |
| 2.5.1 | linux-aarch64 | repack, libtorch pending | 3.9:R* 3.10:R 3.11:R 3.12:R |
| 2.5.1 | win-64 | repack, libtorch pending | 3.9:R* 3.10:R 3.11:R 3.12:R |
| 2.6.0 | linux-64 | repack, libtorch pending | 3.9:R* 3.10:R 3.11:R 3.12:R 3.13:R |
| 2.6.0 | linux-aarch64 | HOLE | — |
| 2.6.0 | win-64 | repack, libtorch pending | 3.9:R* 3.10:R 3.11:R 3.12:R 3.13:R |

### cu126

| version | platform | action | pythons |
|---|---|---|---|
| 2.5.1 | linux-64 | mirror (conda-only) | 3.9:M 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.5.1 | linux-aarch64 | mirror (conda-only) | 3.9:M 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.5.1 | win-64 | mirror (conda-only) | 3.9:M 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.6.0 | linux-64 | mirror | 3.9:M 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.6.0 | linux-aarch64 | repack, cf mirror also hosted | 3.9:R* 3.10:R 3.11:R 3.12:R 3.13:R |
| 2.6.0 | win-64 | mirror | 3.9:M 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.7.1 | linux-64 | mirror | 3.9:M 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.7.1 | linux-aarch64 | mirror (conda-only) | 3.9:M 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.7.1 | win-64 | mirror | 3.9:M 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.8.0 | linux-64 | repack, libtorch pending | 3.9:R* 3.10:R 3.11:R 3.12:R 3.13:R |
| 2.8.0 | linux-aarch64 | HOLE | — |
| 2.8.0 | win-64 | repack, libtorch pending | 3.9:R* 3.10:R 3.11:R 3.12:R 3.13:R |
| 2.9.1 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.9.1 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.9.1 | win-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.10.0 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.10.0 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.10.0 | win-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.11.0 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.11.0 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.11.0 | win-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.12.1 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.12.1 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.12.1 | win-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.13.0 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R |
| 2.13.0 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R |
| 2.13.0 | win-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.14.0 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R |
| 2.14.0 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R |
| 2.14.0 | win-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R |

### cu128

| version | platform | action | pythons |
|---|---|---|---|
| 2.7.1 | linux-64 | repack, libtorch pending | 3.9:R* 3.10:R 3.11:R 3.12:R 3.13:R |
| 2.7.1 | linux-aarch64 | repack, libtorch pending | 3.9:R* 3.10:R 3.11:R 3.12:R 3.13:R |
| 2.7.1 | win-64 | mirror | 3.9:M 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.8.0 | linux-64 | repack, libtorch published | 3.9:R 3.10:R 3.11:R 3.12:P 3.13:R |
| 2.8.0 | linux-aarch64 | HOLE | — |
| 2.8.0 | win-64 | mixed, libtorch published | 3.9:R 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.9.1 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.9.1 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.9.1 | win-64 | mixed, libtorch pending | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:R* |
| 2.10.0 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.10.0 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.10.0 | win-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.11.0 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.11.0 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.11.0 | win-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.12.1 | linux-64 | HOLE | — |
| 2.12.1 | linux-aarch64 | HOLE | — |
| 2.12.1 | win-64 | mirror (conda-only) | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.13.0 | linux-64 | HOLE | — |
| 2.13.0 | linux-aarch64 | HOLE | — |
| 2.13.0 | win-64 | mirror (conda-only) | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |

### cu129

| version | platform | action | pythons |
|---|---|---|---|
| 2.7.1 | linux-64 | mirror (conda-only) | 3.9:M 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.7.1 | linux-aarch64 | mirror (conda-only) | 3.9:M 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.7.1 | win-64 | HOLE | — |
| 2.8.0 | linux-64 | mixed, libtorch pending | 3.9:R* 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.8.0 | linux-aarch64 | repack, cf mirror also hosted | 3.9:R* 3.10:R 3.11:R 3.12:R 3.13:R |
| 2.8.0 | win-64 | repack, libtorch pending | 3.9:R* 3.10:R 3.11:R 3.12:R 3.13:R |
| 2.9.1 | linux-64 | mixed, libtorch pending | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:R* |
| 2.9.1 | linux-aarch64 | repack, cf mirror also hosted | 3.10:R 3.11:R 3.12:R 3.13:R 3.14:R* |
| 2.9.1 | win-64 | HOLE | — |
| 2.10.0 | linux-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.10.0 | linux-aarch64 | repack, cf mirror also hosted | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.10.0 | win-64 | HOLE | — |
| 2.11.0 | linux-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.11.0 | linux-aarch64 | repack, cf mirror also hosted | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.11.0 | win-64 | HOLE | — |
| 2.12.1 | linux-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.12.1 | linux-aarch64 | repack, cf mirror also hosted | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.12.1 | win-64 | HOLE | — |
| 2.13.0 | linux-64 | mixed, libtorch pending | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M 3.15:R* |
| 2.13.0 | linux-aarch64 | repack, cf mirror also hosted | 3.10:R 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R* |
| 2.13.0 | win-64 | HOLE | — |

### cu130

| version | platform | action | pythons |
|---|---|---|---|
| 2.9.1 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.9.1 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.9.1 | win-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.10.0 | linux-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.10.0 | linux-aarch64 | repack, cf mirror also hosted | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.10.0 | win-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.11.0 | linux-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.11.0 | linux-aarch64 | repack, cf mirror also hosted | 3.10:R 3.11:R 3.12:R* 3.13:R 3.14:R |
| 2.11.0 | win-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.12.1 | linux-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.12.1 | linux-aarch64 | repack, cf mirror also hosted | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.12.1 | win-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.13.0 | linux-64 | mixed, libtorch pending | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M 3.15:R* |
| 2.13.0 | linux-aarch64 | repack, cf mirror also hosted | 3.10:R 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R* |
| 2.13.0 | win-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.14.0 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R |
| 2.14.0 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R |
| 2.14.0 | win-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R |

### cu132

| version | platform | action | pythons |
|---|---|---|---|
| 2.12.1 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.12.1 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.12.1 | win-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.13.0 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R |
| 2.13.0 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R |
| 2.13.0 | win-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.14.0 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R |
| 2.14.0 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R |
| 2.14.0 | win-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R 3.15:R |

