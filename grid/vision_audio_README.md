# torchvision + torchaudio: design pass (action item 18)

Generated 2026-09-04. Machine-readable form: [`vision_audio_grid.json`](vision_audio_grid.json).

**Why this exists.** comfy-env's resolver pins the torch *family* — `_TORCH_PKGS = {"torch", "torchvision", "torchaudio"}` in `toml_generator.py`, driven by `TORCH_FAMILY_COMPAT` in `packages/cuda_wheels.py`. A channel carrying torch alone cannot let comfy-env delete its `[cuda]` resolver: any pack needing vision or audio ops still falls back to the old path. This is design only — no CI dispatched, no packages built.

## The compat table (ground truth, read from comfy-env source)

```
torch minor -> (torchvision minor, torchaudio minor)
2.4  -> (0.19, 2.4)      2.5  -> (0.20, 2.5)      2.6  -> (0.21, 2.6)
2.7  -> (0.22, 2.7)      2.8  -> (0.23, 2.8)      2.9  -> (0.24, 2.9)
2.10 -> (0.25, 2.10)     2.11 -> (0.26, 2.11)     2.12 -> (0.27, 2.11)  # torchaudio never shipped a matching 2.12
```

comfy-env's own table stops at 2.12. This grid's 2.13.0/2.14.0 rows **extrapolate** (`0.28`/`0.29`, torchaudio held at `2.11`, following the 2.12 precedent) — flag for comfy-env maintainers to confirm before relying on it.

## ABI coupling — the design-determining finding

Both packages declare an **exact** torch pin in their own METADATA, read from real wheels (torchvision 0.23.0+cu128 cp312, torchaudio 2.8.0+cu128 cp312):

```
torchvision: Requires-Dist: torch (==2.8.0)
torchaudio:  Requires-Dist: torch==2.8.0
```

Not a range. A torchvision/torchaudio repack must pair with the **exact already-published pytorch build** for that (version, flavour, platform, python) cell — same discipline the main grid already applies to pytorch↔libtorch. Practical consequence: torchvision/torchaudio repacks must dispatch **strictly after** their paired pytorch cell exists on the channel, never in parallel.

## Structure — much simpler than the torch repack

Neither package carries a CUDA payload of its own:

| | torchvision 0.23.0+cu128 | torchaudio 2.8.0+cu128 |
|---|---|---|
| wheel size | 8.6 MB | 3.9 MB |
| compiled `.so` | 2 (`_C.so` 10.3 MB, `image.so` 0.7 MB, uncompressed) | 12 |
| RPATH | one dead absolute build-machine path (`/__w/_temp/...`) | none |
| resolution mechanism | **already-loaded-by-soname**: torch's own `libc10`/`libtorch`/`libtorch_cpu`/`libtorch_cuda` are resident in the process before `torchvision._C` is `dlopen`'d, so the dead RPATH is never consulted — same mechanism already verified for `functorch/_C` in the main project | same |
| new CUDA deps needed | none — `libcudart.so.12` is already a `libtorch` dependency | none for the two core `.so`s |

**No libtorch-style split needed.** No shared-artifact donor logic, no per-python big-binary sharing. This is "download the exact-pinned wheel, scrub the dead RPATH, apply the item 1-14 gates (license/purls/provenance/`.pyc`/no-RECORD), depend on the exact paired pytorch build" — meaningfully lighter than the torch pipeline.

**torchaudio's one real complication**: it bundles three ffmpeg-major-specific backends (`_torio_ffmpeg{4,5,6}.so`), each hard-linking an exact ffmpeg soname line (`libavutil.so.58/57/56`, etc.) that is **not vendored**. Upstream's own design probes at runtime for whichever is loadable and degrades gracefully if none is present. Recommendation: leave `ffmpeg` **undeclared** in our repack too (matches upstream behavior) rather than forcing every torchaudio install to drag in a multi-hundred-MB ffmpeg — sox is fully vendored in-wheel, no equivalent question there.

## Coverage

| | records buildable | records with no wheel (hole) |
|---|---:|---:|
| torchvision | **436** | 14 |
| torchaudio | **382** | 68 |

("Buildable" = a PyPI wheel exists for that torch cell's (flavour, platform, python); does not yet distinguish mirror-vs-repack — see below.)

### Named holes

- **torchvision, 2.4.1/cu124/linux-aarch64, all pythons**: no torchvision wheel published at all for that early aarch64+cu124 combo (matches the main grid's own finding that early aarch64 CUDA coverage was thin).
- **torchvision, 2.5.1/cu124/linux-64, python 3.13**: that minor line's torchvision never got a 3.13 build.
- **torchaudio, every cu132 entry (2.12.1, 2.13.0, 2.14.0 × all platforms)**: **structural, not fixable by repacking** — torchaudio's last release is 2.11.0, which predates the cu132 flavour entirely. No wheel can ever exist here; this must ship as a documented permanent hole, the same discipline as the main grid's 13 named holes.
- **torchvision/torchaudio, python 3.15, torch 2.13.0 line**: 3.15 is RC-only — same caveat as the main channel's py3.15 records (self-heals once conda-forge/PyPI both ship stable 3.15).

## conda-forge coverage (mirror candidates)

| | linux-64 / linux-aarch64 | win-64 |
|---|---|---|
| torchvision | full `0.14.0`–`0.28.0` range | `0.20.1`–`0.28.0` (no 0.19 line) |
| torchaudio | full `2.4.1`–`2.11.0` range | **only `2.11.0`** |

Same minor+flavour-exact discipline as the main grid applies before anything can be classified mirror vs. repack — **this per-cell classification is the next concrete implementation step**, not done in this design pass (it needs the same two-pass build-string check `grid.json`'s builder already implements; reuse that code path). Directionally: torchvision mirrors well on linux, decently on win; torchaudio mirrors well on linux but is **almost entirely a repack job on win-64**.

## Byte/effort estimate

~818 buildable records (436 torchvision + 382 torchaudio) × ~10 MiB average observed wheel size (no CUDA payload inflates these the way torch's does) ≈ **8.6 GiB conservative estimate** — about 13% on top of the main channel's ~65 GiB, for the piece that's actually load-bearing for deleting `[cuda]`.

## Pipeline design recommendation

**New script, `tools/repack_family.py`, not an extension of `torch_repack.py`.** The two packages don't need libtorch's split-and-share design — no per-python shared-artifact donor logic, no CUDA payload to strip-and-depend. Shape:

1. Download the exact-pinned wheel for `(torch_version, flavour, platform, python)`.
2. Scrub the dead absolute RPATH to a clean `$ORIGIN`-relative list (reuse `verify_repack.py`'s RPATH lint).
3. Apply the same gates as items 1–14: license collection, no-RECORD dist-info, `INSTALLER=conda`, purls, provenance stamps, env-relative `.pyc`.
4. `depends: pytorch ==<version> <exact build string>` — hard-fail if that pytorch build isn't published yet (ordering dependency, not a parallel dispatch).
5. torchaudio: leave `ffmpeg` undeclared; document the probe-and-degrade behavior in `about.json`.

## Rollout order

1. **torchvision first** — zero external-dependency complexity, 436 buildable records, better conda-forge mirror overlap. Assumption (not verified against actual comfy-forge pack manifests in this pass): more packs use vision ops than audio ops — worth a quick grep of `comfy-forge-registry`/pack listings before committing resourcing order.
2. **torchaudio second** — same mechanics plus the ffmpeg-backend decision above; carries more repack-CI weight on win-64 since conda-forge covers almost nothing there.
