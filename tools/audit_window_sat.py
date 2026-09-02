import json, re, sys, urllib.request
from pathlib import Path
sys.path.insert(0, "tools")
from torch_repack import V, pep_clauses, fetch

CUDA_PKGS = {"cuda-cudart","libcublas","libcudnn","libcusparse","libcufft","libcurand",
             "libcusolver","nccl","cuda-cupti","cuda-nvrtc","libnvjitlink","libcufile",
             "cusparselt","cuda-nvtx","cuda-bindings","nvidia-nvshmem"}
OPS = {"==": V.__eq__, "!=": V.__ne__, "<": V.__lt__, "<=": V.__le__, ">": V.__gt__, ">=": V.__ge__}
clean = lambda ver: re.match(r"[0-9.]*", ver).group(0).rstrip(".")

def sat(ver_or_none, spec):
    if not spec: return True
    try: cl = pep_clauses(spec)
    except SystemExit: return True  # exotic spec: skip rather than false-alarm
    return all(OPS[op](V(ver_or_none), V(clean(v))) for op, v in cl)

def window_overlap(a, b):
    # both conda specs like ">=12.8,<13"; sample-based intersect: check each's bounds inside other
    def bounds(s):
        lo, hi = None, None
        for op, v in pep_clauses(s):
            if op == ">=": lo = clean(v)
            if op == "<": hi = clean(v)
        return lo, hi
    alo, ahi = bounds(a); blo, bhi = bounds(b)
    lo = max(V(alo or "0"), V(blo or "0"), key=lambda x: x.t or (0,))
    hi_candidates = [V(x) for x in (ahi, bhi) if x]
    if not hi_candidates: return True
    hi = min(hi_candidates, key=lambda x: x.t or (999,))
    return lo < hi

_cache = {}
def cf_files(pkg):
    if pkg not in _cache:
        try:
            _cache[pkg] = json.load(fetch(f"https://api.anaconda.org/package/conda-forge/{pkg}/files", 60))
        except Exception:
            _cache[pkg] = []
    return _cache[pkg]

def check(pkg, spec, subdir, window):
    if pkg == "nvidia-nvshmem":  # our own channel, not conda-forge
        return "OURS"
    ok = False
    for f in cf_files(pkg):
        a = f.get("attrs", {})
        if a.get("subdir") != subdir: continue
        if not sat(f["version"], spec): continue
        cv = next((d.split(None,1)[1] for d in a.get("depends",[]) if d.startswith("cuda-version ")), None)
        if cv is None or window_overlap(cv, window):
            ok = True; break
    return "ok" if ok else "UNSAT"

rows, bad = [], []
for frag in sorted(Path("meta").rglob("*.json")):
    j = json.loads(frag.read_text())
    if "repack" not in j.get("build",""): continue
    if j["name"] not in ("libtorch","pytorch"): continue
    subdir = j["subdir"]
    window = next((d.split(None,1)[1] for d in j["depends"] if d.startswith("cuda-version ")), None)
    for d in j["depends"]:
        parts = d.split(None, 1)
        pkg, spec = parts[0], (parts[1] if len(parts)>1 else "")
        if pkg not in CUDA_PKGS: continue
        r = check(pkg, spec, subdir, window or ">=0")
        rows.append((f"{j['name']}-{j['version']}-{j['build']}", subdir, f"{pkg} {spec}".strip(), r))
        if r == "UNSAT": bad.append(rows[-1])

byentry = {}
for e, s, d, r in rows: byentry.setdefault((e,s), []).append((d,r))
print(f"{len(byentry)} entries audited, {len(rows)} CUDA dep checks")
for (e,s), deps in sorted(byentry.items()):
    flags = [f"{d} [{r}]" for d,r in deps if r=="UNSAT"]
    print(("UNSAT " if flags else "pass  ") + f"{s:14} {e}" + ("  ->  " + "; ".join(flags) if flags else ""))
