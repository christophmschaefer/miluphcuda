#!/usr/bin/env python
"""
sph2blender.py -- convert miluphcuda HDF5 outputs into compact per-frame .npz
files for rendering with Blender (render_dimorphos.py).

Runs on the cluster next to the data. Needs only numpy + h5py.

What it does
  * reads x, v, material_type, damage (DIM_root_of_damage_*), alpha_jutzi
  * optional cut plane (cross-section view): keeps n.(x - p) < 0
  * culls the invisible interior: particles deeper than --depth below the
    current surface are dropped (voxel occupancy + erosion, numpy only).
    The kept set of frame i is the union of the visible sets of frames
    i-1, i, i+1, so Blender can interpolate between consecutive dumps
    without particles popping in and out.
  * writes frame_NNNN.npz (float32 x, float16 v, uint32 ids, uint8 mat,
    dmg, comp, float16 rk, sml) + index.json. NNNN is the dump number of the
    input file (impact.0050.h5 -> frame_0050.npz).
  * rk = distance to the k-th neighbour (--knn-k) among ALL target particles
    (before culling), for every kept particle faster than --knn-vmin; 0 for
    the others. The renderer uses it as the local smoothing length of the
    ejecta volume (miluphcuda's own sml is capped and freezes for isolated
    particles, so it is not usable for the thin plume). Needs scipy; without
    scipy rk is not written and the renderer falls back to its own search.
  * sml (miluphcuda smoothing length, full kernel support) is stored as well
    if the dump has it.

Incremental runs
  An existing index.json in the output directory is extended, not replaced:
  new dumps are added, dumps converted again are replaced, everything is
  sorted by time. Frames of earlier runs that are time-neighbours of new
  dumps are rewritten from their HDF5 source (if still there), so particles
  do not pop at the seam. Old-style files (frame_0000.npz for impact.0050.h5)
  are renamed to the dump number. --fresh starts a new index.

With --ref (impact.0000.h5) two more fields are written:
  disp = |x - x0 - shift|, displacement since t = 0 [m], with the translation
         of the whole body removed (shift = mass-weighted mean displacement of
         the target particles within --core m of the initial centre of mass);
         with --derotate the rotation of the body is removed as well (rigid
         Kabsch fit to the same core particles)
  rhor = rho / rho0 (loosening < 1, compaction > 1), if the dumps have rho

Damage  = max(tensile, porjutzi) with d = (DIM_root)^3
Compaction = (alpha_initial - alpha)/(alpha_initial - 1), clipped to [0,1]

Example
  python3 sph2blender.py "impact.*.h5" -o blender_frames
  # cross-section through the impact point, plane containing v_imp and z:
  python3 sph2blender.py "impact.*.h5" -o blender_cut \
      --clip 0.984 0.175 0 -8.06 -80.73 -12.62
"""
import argparse
import glob
import json
import os
import re
import sys
import time as _time

import numpy as np
import h5py

try:
    from scipy.spatial import cKDTree
except ImportError:          # rk is then not written, the renderer searches itself
    cKDTree = None

DIM = 3


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def read_time(f, fallback):
    for k in ("time", "Time", "t"):
        if k in f:
            return float(np.ravel(f[k][()])[0])
        if k in f.attrs:
            return float(np.ravel(f.attrs[k])[0])
    return float(fallback)


def read_damage(f, n):
    d = np.zeros(n, np.float32)
    for k in ("DIM_root_of_damage_tensile", "DIM_root_of_damage_porjutzi"):
        if k in f:
            d = np.maximum(d, np.asarray(f[k][()], np.float32).ravel() ** DIM)
    return np.clip(d, 0.0, 1.0)


def rigid_fit(x0, x, w, iters=5, clip=3.0):
    """Robust mass-weighted best rigid motion (Kabsch): x ~ Rm @ (x0 - c0) + c.
    Particles whose residual exceeds clip * median residual are dropped and the
    fit repeated, so locally deformed material inside the core does not bias it.
    Returns Rm, c0, c, rotation angle [rad], unit axis, rms residual before/after [m]
    (translation only / rigid, over the particles used in the final fit)."""
    use = np.ones(len(x0), bool)
    for _ in range(iters):
        res = _kabsch(x0[use], x[use], w[use])
        Rm, c0, c = res[:3]
        r = np.linalg.norm(x - ((x0 - c0) @ Rm.T + c), axis=1)
        new = r < max(clip * np.median(r), 0.02)
        if np.array_equal(new, use):
            break
        use = new
    return res


def _kabsch(x0, x, w):
    W = w.sum()
    c0 = (w[:, None] * x0).sum(0) / W
    c = (w[:, None] * x).sum(0) / W
    p, q = x0 - c0, x - c
    H = (w[:, None] * p).T @ q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    ang = float(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))
    ax = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    ax = ax / np.linalg.norm(ax) if np.linalg.norm(ax) > 0 else np.array([0.0, 0.0, 1.0])
    rms0 = float(np.sqrt((w * np.sum((q - p) ** 2, axis=1)).sum() / W))
    rms1 = float(np.sqrt((w * np.sum((q - p @ Rm.T) ** 2, axis=1)).sum() / W))
    return Rm, c0, c, ang, ax, rms0, rms1


def erode6(a):
    b = a.copy()
    b[1:, :, :] &= a[:-1, :, :]
    b[:-1, :, :] &= a[1:, :, :]
    b[:, 1:, :] &= a[:, :-1, :]
    b[:, :-1, :] &= a[:, 1:, :]
    b[:, :, 1:] &= a[:, :, :-1]
    b[:, :, :-1] &= a[:, :, 1:]
    b[0], b[-1], b[:, 0], b[:, -1], b[:, :, 0], b[:, :, -1] = (False,) * 6
    return b


def visible_mask(x, inplane, box_lo, box_hi, cell, nerode, never_cull):
    """True for particles that can be seen (within nerode cells of a surface)."""
    n = len(x)
    keep = inplane.copy()
    inbox = inplane & np.all((x >= box_lo) & (x < box_hi), axis=1)
    if nerode <= 0 or not inbox.any():
        return keep
    shape = np.ceil((box_hi - box_lo) / cell).astype(int) + 2
    idx = np.floor((x[inbox] - box_lo) / cell).astype(np.int64) + 1
    occ = np.zeros(shape, bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    inner = occ
    for _ in range(nerode):
        inner = erode6(inner)
    hidden = np.zeros(n, bool)
    hidden[np.nonzero(inbox)[0]] = inner[idx[:, 0], idx[:, 1], idx[:, 2]]
    hidden &= ~never_cull
    return keep & ~hidden


def knn_distance(xa, va, mat, ids, k, vmin, projmat):
    """Distance to the k-th neighbour among all target particles, for the kept
    particles (ids) faster than vmin rel. to the bulk; 0 for the others."""
    rk = np.zeros(len(ids), np.float16)
    tgt = mat != projmat
    spd = np.linalg.norm(va - np.median(va[tgt], axis=0), axis=1)
    cand = (tgt & (spd > vmin))[ids]
    if cand.any():
        tree = cKDTree(xa[tgt], balanced_tree=False, compact_nodes=False)
        d, _ = tree.query(xa[ids[cand]], k=k + 1, workers=-1)   # +1: the particle itself
        rk[cand] = np.minimum(d[:, -1], 6.0e4).astype(np.float16)
    return rk, int(cand.sum())


def dump_label(fn):
    """Dump number from the file name: last digit group (impact.0050.h5 -> '0050')."""
    stem = os.path.basename(fn)
    while True:
        root, ext = os.path.splitext(stem)
        if ext.lower() in (".h5", ".hdf5", ".npz") and root:
            stem = root
        else:
            break
    m = re.findall(r"\d+", stem)
    return m[-1].zfill(4) if m else None


def load_old_index(path, a):
    """Entries of an existing index.json (list of dicts) and its settings."""
    with open(path) as f:
        I = json.load(f)
    for key, val in (("stride", a.stride), ("clip", a.clip), ("depth", a.depth)):
        if key in I and I[key] != val:
            sys.exit(f"existing {path} was made with {key} = {I[key]}, this run uses {val}; "
                     "use the same options, another -o, or --fresh")
    want = (a.knn_k, a.knn_vmin) if cKDTree is not None and a.knn_k > 0 else None
    have = (I.get("knn_k"), I.get("knn_vmin")) if I.get("knn_k") else None
    if want != have:
        print(f"WARNING: frames in {path} were written with rk settings {have}, this run "
              f"uses {want}. Old frames are only updated at the seams; for a consistent "
              "series convert everything again with --fresh.")
    n = len(I["files"])
    src = I.get("source") or [None] * n
    out = [dict(time=float(I["times"][k]), file=I["files"][k], N=int(I["N"][k]),
                source=src[k], new=False) for k in range(n)]
    return out, I


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="HDF5 files or glob pattern(s)")
    ap.add_argument("-o", "--out", default="blender_frames")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing index.json in the output directory")
    ap.add_argument("--depth", type=float, default=3.0,
                    help="keep particles within this depth [m] of the surface; 0 = keep all")
    ap.add_argument("--cell", type=float, default=1.2,
                    help="voxel size [m] for surface detection (> particle spacing)")
    ap.add_argument("--box-pad", type=float, default=25.0,
                    help="padding [m] around the body for the culling grid; "
                         "particles outside are always kept")
    ap.add_argument("--max-box", type=float, default=400.0,
                    help="max. edge length [m] of the culling grid")
    ap.add_argument("--box", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                    help="explicit culling box [m] instead of the automatic one "
                         "(default: box of an existing index.json, else automatic)")
    ap.add_argument("--ref", help="initial dump (e.g. impact.0000.h5) as reference for "
                    "alpha_0 (compaction); default: per-material 99th percentile")
    ap.add_argument("--clip", type=float, nargs=6, metavar=("NX", "NY", "NZ", "PX", "PY", "PZ"),
                    help="cut plane: keep n.(x-p) < 0")
    ap.add_argument("--keep-sphere", type=float, nargs=4, metavar=("X", "Y", "Z", "R"),
                    default=[-8.06, -80.73, -12.62, 40.0],
                    help="never cull inside this sphere (impact site) [m]; R=0 disables")
    ap.add_argument("--keep-vmin", type=float, default=0.2,
                    help="never cull particles faster than this [m/s] rel. to the bulk")
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th particle (tests)")
    ap.add_argument("--projectile-mat", type=int, default=2)
    ap.add_argument("--core", type=float, default=30.0,
                    help="with --ref: particles within this radius [m] of the initial COM define "
                         "the body translation that is removed from disp")
    ap.add_argument("--derotate", action="store_true",
                    help="with --ref: remove the body rotation from disp as well (rigid fit to the core)")
    ap.add_argument("--knn-k", type=int, default=16,
                    help="neighbour number for the ejecta smoothing length rk; 0 = do not write rk")
    ap.add_argument("--knn-vmin", type=float, default=0.05,
                    help="rk is computed for particles faster than this [m/s] rel. to the bulk "
                         "(must be below the smallest --vcut used for rendering)")
    a = ap.parse_args()
    if a.knn_k > 0 and cKDTree is None:
        print("WARNING: scipy not found, rk is not written (renderer falls back to its own search)")

    files = []
    for pat in a.files:
        files += glob.glob(pat) if any(c in pat for c in "*?[") else [pat]
    files = sorted(set(os.path.abspath(f) for f in files), key=natural_key)
    if not files:
        sys.exit("no input files")
    os.makedirs(a.out, exist_ok=True)
    idx_path = os.path.join(a.out, "index.json")

    # ---- existing index ----------------------------------------------------
    old, old_meta = [], {}
    if os.path.exists(idx_path) and not a.fresh:
        old, old_meta = load_old_index(idx_path, a)
        print(f"existing index: {len(old)} frame(s) in {a.out}/ (extending; --fresh to restart)")
    newset = set(files)
    replaced = [e for e in old if e["source"] and os.path.abspath(e["source"]) in newset]
    old = [e for e in old if not (e["source"] and os.path.abspath(e["source"]) in newset)]
    if replaced:
        print(f"{len(replaced)} dump(s) already in the index are converted again")

    # ---- pass 1: times, visibility masks ---------------------------------
    t0 = _time.time()
    with h5py.File(files[0], "r") as f:
        print("datasets in", files[0], ":", ", ".join(sorted(f.keys())))
        x0 = np.asarray(f["x"][()], np.float64)[::a.stride]
        mat = np.asarray(f["material_type"][()]).ravel()[::a.stride].astype(np.int16)
        alpha0 = (np.asarray(f["alpha_jutzi"][()], np.float32).ravel()[::a.stride]
                  if "alpha_jutzi" in f else None)
    N0 = len(x0)
    target = mat != a.projectile_mat

    # reference distension alpha_0 for the compaction field
    xref = rhoref = core = mref = None
    if a.ref:
        with h5py.File(a.ref, "r") as f:
            alpha0 = (np.asarray(f["alpha_jutzi"][()], np.float32).ravel()[::a.stride]
                      if "alpha_jutzi" in f else alpha0)
            xref = np.asarray(f["x"][()], np.float64)[::a.stride]
            rhoref = (np.asarray(f["rho"][()], np.float32).ravel()[::a.stride]
                      if "rho" in f else None)
            mref = (np.asarray(f["m"][()], np.float64).ravel()[::a.stride]
                    if "m" in f else np.ones(len(xref)))
        if len(xref) != N0:
            print(f"WARNING: --ref has {len(xref)} particles, first file {N0}: no disp/rhor")
            xref = rhoref = None
        else:
            com0 = (mref[target, None] * xref[target]).sum(0) / mref[target].sum()
            core = target & (np.linalg.norm(xref - com0, axis=1) < a.core)
            print(f"--ref: displacement relative to the body; core {core.sum()} particles "
                  f"within {a.core:g} m of the initial COM {np.round(com0, 2)} m")
    elif alpha0 is not None:
        # first file may be a late dump: take the per-material maximum (pristine material)
        a0m = alpha0.copy()
        for m in np.unique(mat):
            sel = mat == m
            a0m[sel] = np.percentile(alpha0[sel], 99.0)
        alpha0 = a0m

    # culling grid: robust box around the *body* (quantiles, so far ejecta do not
    # blow up the grid), padded and capped. Particles outside are always kept.
    # An existing index keeps its box, so all frames are culled the same way.
    if a.box:
        box_lo, box_hi = np.array(a.box[:3]), np.array(a.box[3:])
    elif old and "box" in old_meta and old_meta.get("cell") == a.cell:
        box_lo, box_hi = np.array(old_meta["box"][:3]), np.array(old_meta["box"][3:])
    else:
        xt = x0[target]
        lo, hi = np.quantile(xt, 0.01, axis=0), np.quantile(xt, 0.99, axis=0)
        c, half = 0.5 * (lo + hi), np.minimum(0.5 * (hi - lo) + a.box_pad, 0.5 * a.max_box)
        box_lo, box_hi = c - half, c + half
    ncell = np.prod(np.ceil((box_hi - box_lo) / a.cell) + 2)
    print(f"culling grid: box {np.round(box_lo,1)} .. {np.round(box_hi,1)} m, "
          f"{ncell/1e6:.1f} M cells (~{3*ncell/1e9:.2f} GB peak)")
    nerode = int(np.ceil(a.depth / a.cell)) if a.depth > 0 else 0
    never_cull = mat == a.projectile_mat

    def visibility(fn, i):
        with h5py.File(fn, "r") as f:
            t = read_time(f, i)
            x = np.asarray(f["x"][()], np.float64)[::a.stride]
            vv = np.asarray(f["v"][()], np.float32)[::a.stride]
        n = len(x)
        if n != N0:
            print(f"WARNING: {fn} has {n} particles, first file {N0}; "
                  "interpolation across this frame will be disabled in Blender")
        nc = never_cull.copy() if n == N0 else np.zeros(n, bool)
        # never cull moving material (ejecta, excavation flow): it is rendered as volume
        spd = np.linalg.norm(vv - np.median(vv, axis=0), axis=1)
        nc |= spd > a.keep_vmin
        if a.keep_sphere and a.keep_sphere[3] > 0:
            # crater region: keep everything, so removing the ejecta later leaves a
            # crater floor instead of a hole into the (culled) hollow interior
            c, r = np.array(a.keep_sphere[:3]), a.keep_sphere[3]
            nc |= np.sum((x - c) ** 2, axis=1) < r * r
        inplane = np.ones(n, bool)
        if a.clip:
            nrm = np.array(a.clip[:3]); nrm /= np.linalg.norm(nrm)
            inplane = (x - np.array(a.clip[3:])) @ nrm < 0
        return t, n, visible_mask(x, inplane, box_lo, box_hi, a.cell, nerode, nc)

    new = []
    for i, fn in enumerate(files):
        t, n, m = visibility(fn, i)
        new.append(dict(time=t, file=None, N=n, source=fn, new=True, mask=np.packbits(m)))
        print(f"[1/2] {os.path.basename(fn)}  t = {t:9.4f} s  N = {n}  visible = {m.sum()}")

    # ---- merge with the existing index -----------------------------------
    ent = sorted(old + new, key=lambda e: e["time"])
    nE = len(ent)
    rewrite = [e["new"] for e in ent]
    for k in range(nE):
        if ent[k]["new"]:
            for j in (k - 1, k + 1):
                if 0 <= j < nE and not ent[j]["new"] and ent[j]["N"] == ent[k]["N"]:
                    src = ent[j]["source"]
                    if src and os.path.exists(src):
                        rewrite[j] = True
                    else:
                        print(f"WARNING: source of {ent[j]['file']} not found ({src}); "
                              "it is not updated, particles may pop at this seam")
    # masks of old frames = their kept ids (a superset of their visible set;
    # enough to keep the neighbours complete). Needed for rewritten frames and
    # for the neighbours of rewritten frames.
    for k in range(nE):
        if ent[k]["new"]:
            continue
        need = rewrite[k] or any(0 <= j < nE and rewrite[j] for j in (k - 1, k + 1))
        if need:
            p = os.path.join(a.out, ent[k]["file"])
            if os.path.exists(p):
                with np.load(p) as z:
                    mk = np.zeros(ent[k]["N"], bool)
                    mk[z["ids"]] = True
                ent[k]["mask"] = np.packbits(mk)
            elif rewrite[k]:
                print(f"WARNING: {p} missing, recomputing its visibility from the source")
                ent[k]["mask"] = np.packbits(visibility(ent[k]["source"], k)[2])

    # ---- file names: dump number of the source ----------------------------
    taken = set()
    for e in ent:
        lab = dump_label(e["source"]) if e["source"] else None
        if lab is None:
            name = e["file"] if e["file"] else f"frame_t{e['time']:.6f}.npz"
        else:
            name = f"frame_{lab}.npz"
        base, k = name[:-4], 1
        while name in taken:
            k += 1
            name = f"{base}_{k}.npz"
        taken.add(name)
        e["final"] = name
    # rename frames of earlier runs (two steps, so swapped names cannot collide)
    moves = [e for k, e in enumerate(ent) if not e["new"] and not rewrite[k]
             and e["file"] != e["final"] and os.path.exists(os.path.join(a.out, e["file"]))]
    for e in moves:
        os.replace(os.path.join(a.out, e["file"]), os.path.join(a.out, e["file"] + ".mv"))
    for e in replaced + [e for k, e in enumerate(ent) if not e["new"] and rewrite[k]]:
        p = os.path.join(a.out, e["file"])
        if e["file"] not in taken and os.path.exists(p):
            os.remove(p)
    for e in moves:
        os.replace(os.path.join(a.out, e["file"] + ".mv"), os.path.join(a.out, e["final"]))
        print(f"renamed {e['file']} -> {e['final']}")

    def unpack(k):
        return np.unpackbits(ent[k]["mask"], count=ent[k]["N"]).astype(bool)

    # ---- pass 2: write frames ---------------------------------------------
    nw = 0
    for i, e in enumerate(ent):
        if not rewrite[i]:
            continue
        fn = e["source"]
        keep = unpack(i)
        for j in (i - 1, i + 1):
            if 0 <= j < nE and ent[j]["N"] == e["N"] and "mask" in ent[j]:
                keep |= unpack(j)
        ids = np.nonzero(keep)[0].astype(np.uint32)
        extra, ncand = {}, 0
        with h5py.File(fn, "r") as f:
            xa = np.asarray(f["x"][()], np.float32)[::a.stride]
            va = np.asarray(f["v"][()], np.float32)[::a.stride]
            ma = np.asarray(f["material_type"][()]).ravel()[::a.stride]
            x, v, mt = xa[ids], va[ids], ma[ids]
            if xref is not None and e["N"] == N0:
                xa64 = xa.astype(np.float64)
                shift = (mref[core, None] * (xa64[core] - xref[core])).sum(0) / mref[core].sum()
                if a.derotate:
                    Rm, c0, c1, ang, ax, rms0, rms1 = rigid_fit(xref[core], xa64[core], mref[core])
                    xb = (xa64[ids] - c1) @ Rm + c0
                    extra["rot"] = Rm
                    extra["rot_angle"] = np.float64(ang)
                    print(f"      rigid fit: rotation {np.degrees(ang):.4f} deg about "
                          f"({ax[0]:+.3f}, {ax[1]:+.3f}, {ax[2]:+.3f}), core rms "
                          f"{rms0:.3f} -> {rms1:.3f} m")
                else:
                    xb = xa64[ids] - shift
                extra["disp"] = np.linalg.norm(xb - xref[ids], axis=1).astype(np.float16)
                extra["shift"] = shift
                if rhoref is not None and "rho" in f:
                    rr = np.asarray(f["rho"][()], np.float32).ravel()[::a.stride][ids]
                    extra["rhor"] = (rr / np.maximum(rhoref[ids], 1e-12)).astype(np.float16)
                del xa64
            if "sml" in f:
                extra["sml"] = np.asarray(f["sml"][()], np.float32).ravel()[::a.stride][ids] \
                    .astype(np.float16)
            if a.knn_k > 0 and cKDTree is not None:
                extra["rk"], ncand = knn_distance(xa, va, ma, ids, a.knn_k, a.knn_vmin,
                                                  a.projectile_mat)
                extra["knn_k"] = np.int64(a.knn_k)
            del xa, va, ma
            dmg = read_damage(f, f["x"].shape[0])[::a.stride][ids]
            if alpha0 is not None and "alpha_jutzi" in f and e["N"] == N0:
                al = np.asarray(f["alpha_jutzi"][()], np.float32).ravel()[::a.stride][ids]
                a0 = alpha0[ids]
                comp = np.where(a0 > 1.001, (a0 - al) / np.maximum(a0 - 1.0, 1e-6), 0.0)
            else:
                comp = np.zeros(len(ids), np.float32)
        out = os.path.join(a.out, e["final"])
        np.savez(out, time=np.float64(e["time"]), N=np.int64(e["N"]), ids=ids,
                 x=x, v=v.astype(np.float16),
                 mat=np.clip(mt, 0, 255).astype(np.uint8),
                 dmg=np.round(dmg * 255).astype(np.uint8),
                 comp=np.round(np.clip(comp, 0, 1) * 255).astype(np.uint8), **extra)
        nw += 1
        tag = "" if e["new"] else "  (seam, updated)"
        rkinfo = f"  rk for {ncand}" if "rk" in extra else ""
        if "shift" in extra:
            rkinfo += f"  body shift {np.round(extra['shift'], 2)} m"
        print(f"[2/2] {e['final']}  t = {e['time']:9.4f} s  "
              f"kept {len(ids)} / {e['N']}{rkinfo}  ({os.path.getsize(out)/1e6:.0f} MB){tag}")

    index = {"times": [e["time"] for e in ent], "files": [e["final"] for e in ent],
             "N": [e["N"] for e in ent], "stride": a.stride, "clip": a.clip,
             "depth": a.depth, "cell": a.cell,
             "box": [float(q) for q in np.concatenate([box_lo, box_hi])],
             "source": [e["source"] for e in ent]}
    if a.knn_k > 0 and cKDTree is not None:
        index["knn_k"], index["knn_vmin"] = a.knn_k, a.knn_vmin
    tmp = idx_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f, indent=1)
    os.replace(tmp, idx_path)
    print(f"done: {nw} frame(s) written, index has {nE} in {_time.time()-t0:.0f} s -> {a.out}/")


if __name__ == "__main__":
    main()
