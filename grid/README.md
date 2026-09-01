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
| per-python records: mirror / repack | 164 / 286 |
| **.conda files, core channel** | **551** |
| = repacked shims 286 + repacked libtorch 63 + mirrored pytorch 164 + mirrored libtorch 38 | |
| .conda files incl. conda-only bonus | 599 |
| estimated bytes: mirrored (actual sizes) | 23.6 GiB |
| estimated bytes: repacked (508 MB linux anchor; cf means win/aarch64) | 36.2 GiB |
| estimated bytes: conda-only bonus | 5.4 GiB |
| **estimated channel total** | **65.1 GiB** |

Repacked-libtorch size assumptions: linux-64 485 MiB, linux-aarch64 546 MiB, win-64 418 MiB (linux-64 measured from the pilot; others = mean of conda-forge's cuda libtorch artifacts on that subdir).

## Per platform

| platform | entries | mirror recs | repack recs | repack libtorch | holes |
|---|---:|---:|---:|---:|---:|
| linux-64 | 32 | 58 | 109 | 24 | 2 |
| linux-aarch64 | 28 | 53 | 93 | 21 | 5 |
| win-64 | 27 | 53 | 84 | 18 | 6 |

## Drift vs the 2026-08 coverage table

One change only: **torch 2.14.0 landed on PyPI** (cu126/cu130/cu132, all three platforms, py3.10-3.15, conda-forge has none of it) — 9 new all-repack entries. No existing mark flipped.

Excluded from the grid, by name, not silently: 176 freethreaded (`cp31xt`) wheels across the indexes — a python-ABI axis to add later, not a flavour/platform gap.

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
| 2.6.0 | linux-aarch64 | mirror | 3.9:M 3.10:M 3.11:M 3.12:M 3.13:M |
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
| 2.8.0 | linux-aarch64 | mixed, libtorch pending | 3.9:R* 3.10:M 3.11:M 3.12:M 3.13:M |
| 2.8.0 | win-64 | repack, libtorch pending | 3.9:R* 3.10:R 3.11:R 3.12:R 3.13:R |
| 2.9.1 | linux-64 | mixed, libtorch pending | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:R* |
| 2.9.1 | linux-aarch64 | mixed, libtorch pending | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:R* |
| 2.9.1 | win-64 | HOLE | — |
| 2.10.0 | linux-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.10.0 | linux-aarch64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.10.0 | win-64 | HOLE | — |
| 2.11.0 | linux-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.11.0 | linux-aarch64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.11.0 | win-64 | HOLE | — |
| 2.12.1 | linux-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.12.1 | linux-aarch64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.12.1 | win-64 | HOLE | — |
| 2.13.0 | linux-64 | mixed, libtorch pending | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M 3.15:R* |
| 2.13.0 | linux-aarch64 | mixed, libtorch pending | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M 3.15:R* |
| 2.13.0 | win-64 | HOLE | — |

### cu130

| version | platform | action | pythons |
|---|---|---|---|
| 2.9.1 | linux-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.9.1 | linux-aarch64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.9.1 | win-64 | repack, libtorch pending | 3.10:R* 3.11:R 3.12:R 3.13:R 3.14:R |
| 2.10.0 | linux-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.10.0 | linux-aarch64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.10.0 | win-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.11.0 | linux-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.11.0 | linux-aarch64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.11.0 | win-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.12.1 | linux-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.12.1 | linux-aarch64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.12.1 | win-64 | mirror | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M |
| 2.13.0 | linux-64 | mixed, libtorch pending | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M 3.15:R* |
| 2.13.0 | linux-aarch64 | mixed, libtorch pending | 3.10:M 3.11:M 3.12:M 3.13:M 3.14:M 3.15:R* |
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

