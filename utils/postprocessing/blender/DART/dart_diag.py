#!/usr/bin/env python
"""
dart_diag.py -- physical diagnostics of a miluphcuda DART/Dimorphos dump.

Needs numpy, h5py, scipy. Runs on one HDF5 dump (5e6 particles: ~1-3 min,
a few GB RAM; use a compute node, not the login node).

What it reports
  1. bound / escaping mass (two-body energy criterion around the bound body,
     iterated), mass fractions in speed bins relative to the bound body
  2. momentum transfer beta = M_bound * dv_bound . n / (m_p v_p), n = impact
     direction (also the ejecta-momentum estimate 1 + p_ej . (-n) / (m_p v_p))
  3. material mix per speed bin (is the slow debris matrix or boulders?)
  4. negative pressure in fully damaged matrix (artificial cohesion?)
  5. clumps in the moving material: friends-of-friends on particles with
     v_rel > --vlo, per clump: size, N, matrix fraction, nearest-neighbour
     spacing and density relative to the initial state. Clumps much denser /
     tighter than the initial lattice point to SPH clumping (tensile
     instability, residual cohesion), not to physics.

Usage
  python3 dart_diag.py impact.0270.h5 --ref impact.0000.h5
  python3 dart_diag.py impact.0270.h5 --ref impact.0000.h5 --clumps-out clumps_0270.csv

Without --ref, density changes and the initial spacing fall back to the
material table below. All quantities SI (m, kg, s).
"""
import argparse
import sys
import time as _time

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

G = 6.674e-11
MAT_NAMES = {0: "matrix", 1: "boulders", 2: "projectile"}
# fallback particle masses [kg] (V_p = 0.3318 m^3, rho_s = 3200 / alpha_0)
MASS_FALLBACK = {0: 3200 / 1.8 * 0.3318, 1: 3200 / 1.1 * 0.3318, 2: 579.4 / 41}


def ds(f, *names):
    for n in names:
        if n in f:
            return np.asarray(f[n][()])
    return None


def read(fn, need_extra=True):
    with h5py.File(fn, "r") as f:
        keys = sorted(f.keys())
        d = dict(keys=keys)
        d["t"] = float(np.ravel(ds(f, "time", "Time", "t"))[0]) if ds(f, "time", "Time", "t") is not None else float("nan")
        d["x"] = np.asarray(f["x"][()], np.float64)
        d["v"] = np.asarray(f["v"][()], np.float64)
        d["mat"] = np.asarray(f["material_type"][()]).ravel().astype(np.int16)
        m = ds(f, "m", "mass")
        d["m_from_file"] = m is not None
        if m is None:
            m = np.array([MASS_FALLBACK.get(int(k), MASS_FALLBACK[0]) for k in d["mat"]])
        d["m"] = np.asarray(m, np.float64).ravel()
        if need_extra:
            for k, names in (("rho", ("rho", "density")), ("p", ("p", "pressure")),
                             ("alpha", ("alpha_jutzi",)),
                             ("dt", ("DIM_root_of_damage_tensile",)),
                             ("dp", ("DIM_root_of_damage_porjutzi",))):
                a = ds(f, *names)
                d[k] = None if a is None else np.asarray(a, np.float64).ravel()
    return d


def bound_set(x, v, m, sel, iters=6):
    """Iterate COM / COM velocity of the gravitationally bound part of `sel`."""
    b = sel.copy()
    for _ in range(iters):
        M = m[b].sum()
        xc = (m[b, None] * x[b]).sum(0) / M
        vc = (m[b, None] * v[b]).sum(0) / M
        r = np.linalg.norm(x - xc, axis=1)
        vr = np.linalg.norm(v - vc, axis=1)
        # point-mass potential; inside the body particles are slow -> bound anyway
        E = 0.5 * vr**2 - G * M / np.maximum(r, 1.0)
        nb = sel & (E < 0)
        if np.array_equal(nb, b):
            break
        b = nb
    return b, M, xc, vc, vr, r


def fmt_mass(x, M):
    return f"{x:10.3e} kg  ({100 * x / M:7.3f} %)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--ref", help="initial dump (impact.0000.h5): initial density/spacing, projectile momentum")
    ap.add_argument("--spacing", type=float, default=0.777, help="initial lattice spacing a [m]")
    ap.add_argument("--projectile-mat", type=int, default=2)
    ap.add_argument("--matrix-mat", type=int, default=0)
    ap.add_argument("--vimp", type=float, nargs=3, default=[-1063.0, 5963.7, 1032.0],
                    help="impact velocity [m/s] (used if no --ref)")
    ap.add_argument("--mimp", type=float, default=579.4, help="projectile mass [kg] (used if no --ref)")
    ap.add_argument("--track", type=float, nargs=3, default=[0.0, -1.0, 0.0],
                    help="orbital velocity direction of Dimorphos in the body frame (+x to Didymos, "
                         "prograde orbit -> -y; the leading hemisphere is the one DART hit)")
    ap.add_argument("--dvt-obs", type=float, default=2.70,
                    help="measured along-track dv [mm/s] (Cheng et al. 2023: 2.70 +- 0.10)")
    ap.add_argument("--vbins", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100])
    ap.add_argument("--vlo", type=float, default=0.03,
                    help="clump search: material moving faster than this [m/s] rel. to the body")
    ap.add_argument("--link", type=float, default=1.5, help="FoF linking length [units of a]")
    ap.add_argument("--nmin", type=int, default=20, help="min. particles per clump")
    ap.add_argument("--clumps-out", help="write clump table (CSV)")
    a = ap.parse_args()
    t0 = _time.time()
    A = a.spacing

    D = read(a.dump)
    x, v, m, mat = D["x"], D["v"], D["m"], D["mat"]
    n = len(x)
    print(f"{a.dump}: t = {D['t']:.3f} s, N = {n}")
    print("datasets:", ", ".join(D["keys"]))
    if not D["m_from_file"]:
        print("WARNING: no mass dataset, using fallback masses per material")
    proj = mat == a.projectile_mat
    tgt = ~proj

    R = read(a.ref) if a.ref else None
    if R is not None and len(R["x"]) != n:
        print("WARNING: --ref has a different particle number, ignoring it")
        R = None
    if R is not None and not proj.any():
        print(f"WARNING: no projectile particles (material {a.projectile_mat}), using --mimp/--vimp")
    if R is not None and proj.any():
        pp = (R["m"][proj, None] * R["v"][proj]).sum(0)
        mp = R["m"][proj].sum()
        p_t0 = (R["m"][tgt, None] * R["v"][tgt]).sum(0)
        M_t0 = R["m"][tgt].sum()
    else:
        mp, pp = a.mimp, a.mimp * np.array(a.vimp)
        p_t0, M_t0 = np.zeros(3), m[tgt].sum()
    vp = np.linalg.norm(pp) / mp
    nhat = pp / np.linalg.norm(pp)
    v_t0 = p_t0 / M_t0

    # ---- 1. bound / escaping ------------------------------------------------
    b, Mb, xc, vc, vr, r = bound_set(x, v, m, tgt)
    Mt = m[tgt].sum()
    esc = tgt & ~b
    vesc = np.sqrt(2 * G * Mb / (np.cbrt(3 * Mb / (4 * np.pi * 2117.0)) if Mb > 0 else 1))
    print("\n== 1. bound and escaping mass (Dimorphos alone, no Didymos) ==")
    print(f"target mass        {Mt:10.3e} kg")
    print(f"bound              {fmt_mass(Mb, Mt)}")
    print(f"escaping (E > 0)   {fmt_mass(m[esc].sum(), Mt)}")
    print(f"v_esc (surface, rho_bulk 2117) ~ {vesc*100:.1f} cm/s, bound body COM at {np.round(xc, 2)} m")
    edges = np.concatenate([[0.0], a.vbins, [np.inf]])
    print("\nspeed rel. to bound body        mass                     N    matrix%  boulder%  bound%")
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = tgt & (vr >= lo) & (vr < hi)
        if not s.any():
            continue
        ms = m[s].sum()
        fm = 100 * m[s & (mat == a.matrix_mat)].sum() / ms
        fb = 100 * m[s & (mat != a.matrix_mat)].sum() / ms
        fbd = 100 * m[s & b].sum() / ms
        print(f"  {lo:7.2f} .. {hi:<7.2f} m/s   {fmt_mass(ms, Mt)}  {s.sum():9d}  {fm:6.1f}   {fb:6.1f}   {fbd:6.1f}")

    # ---- 2. beta --------------------------------------------------------------
    print("\n== 2. momentum transfer (along the impact direction) ==")
    dv_b = vc - v_t0
    beta_b = Mb * np.dot(dv_b, nhat) / (mp * vp)
    p_ej = (m[esc, None] * (v[esc] - v_t0)).sum(0)
    beta_e = 1.0 + np.dot(p_ej, -nhat) / (mp * vp)
    p_proj = (m[proj, None] * v[proj]).sum(0)
    print(f"projectile         m = {mp:.1f} kg, v = {vp:.0f} m/s")
    print(f"dv of bound body   {np.round(dv_b * 1000, 4)} mm/s  (|dv| = {np.linalg.norm(dv_b)*1000:.4f} mm/s)")
    print(f"beta (bound body)  {beta_b:.3f}")
    print(f"beta (1 + ejecta)  {beta_e:.3f}   (projectile momentum now {np.dot(p_proj, nhat)/(mp*vp):+.3f} m_p v_p)")
    et = np.array(a.track) / np.linalg.norm(a.track)
    dvt = np.dot(dv_b, et) * 1000
    print(f"along-track dv     {dvt:+.3f} mm/s  (measured {-a.dvt_obs:+.2f} mm/s; ratio {abs(dvt)/a.dvt_obs:.2f}, "
          f"M_bound = {Mb:.3e} kg)")
    print(f"  without the projectile rebound: {np.dot(dv_b + p_proj/Mb, et)*1000:+.3f} mm/s")
    print("  note: beta here is along the impact direction, the DART value (~3.6) is along the")
    print("  orbit; both grow until the slow ejecta have decided between falling back and escaping")

    # ---- 3./4. pressure in damaged matrix -------------------------------------
    print("\n== 3. negative pressure in damaged matrix ==")
    if D["p"] is None:
        print("no pressure dataset in the dump (output p to check this)")
    else:
        d = np.zeros(n)
        for k in ("dt", "dp"):
            if D[k] is not None:
                d = np.maximum(d, np.clip(D[k], 0, 1) ** 3)
        sm = (mat == a.matrix_mat) & (d > 0.99)
        if not sm.any():
            sm = mat == a.matrix_mat
            print("(no damage dataset or no damaged matrix: using all matrix particles)")
        p = D["p"][sm]
        neg = p < 0
        q = np.percentile(p, [0.1, 1, 50, 99, 99.9])
        print(f"damaged matrix N = {sm.sum()}, p < 0: {neg.sum()} ({100*neg.mean():.3f} %), "
              f"min p = {p.min():.3e} Pa")
        print(f"p quantiles 0.1/1/50/99/99.9 %: " + " / ".join(f"{z:.3e}" for z in q) + " Pa")
        if neg.any():
            mv = vr[sm][neg] > a.vlo
            print(f"  of these {mv.sum()} move faster than {a.vlo} m/s (in the debris); "
                  f"median p there {np.median(p[neg]):.3e} Pa")
            print("  -> nonzero tension in d = 1 material acts as artificial cohesion")
        for m_ in sorted(set(np.unique(mat)) - {a.matrix_mat, a.projectile_mat}):
            pb = D["p"][mat == m_]
            print(f"{MAT_NAMES.get(int(m_), m_)}: p < 0 in {100*(pb<0).mean():.2f} %, min p = {pb.min():.3e} Pa")

    # ---- 5. clumps ----------------------------------------------------------
    print(f"\n== 4. clumps in moving material (v_rel > {a.vlo} m/s, FoF b = {a.link} a) ==")
    mov = np.nonzero(tgt & (vr > a.vlo))[0]
    if len(mov) < a.nmin:
        print("hardly any moving material")
        print(f"\ndone in {_time.time()-t0:.0f} s")
        return
    xm = x[mov]
    tree = cKDTree(xm)
    pairs = tree.query_pairs(a.link * A, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(len(mov), len(mov)))
    ncomp, lab = connected_components(g, directed=False)
    dnn = tree.query(xm, k=2)[0][:, 1]
    # initial nearest-neighbour spacing of the same particles
    if R is not None:
        dnn0 = cKDTree(R["x"]).query(R["x"][mov], k=2)[0][:, 1]
        rho0 = R["rho"][mov] if R["rho"] is not None else None
    else:
        dnn0, rho0 = np.full(len(mov), A), None
    rho = D["rho"][mov] if D["rho"] is not None else None
    ratio = dnn / dnn0
    print(f"moving particles {len(mov)}, {ncomp} FoF groups")
    print(f"nearest-neighbour distance / initial: 5/50/95 % = "
          + " / ".join(f"{z:.2f}" for z in np.percentile(ratio, [5, 50, 95])))
    tight = ratio < 0.6
    print(f"particles closer than 0.6x their initial spacing: {tight.sum()} ({100*tight.mean():.2f} %)"
          "  (pairing -> tensile instability)")

    # groups touching the resting body (a static particle within the linking length)
    # are the moving surface layer, not detached clumps
    stat = np.nonzero(tgt & (vr <= a.vlo))[0]
    touch = np.zeros(len(mov), bool)
    if len(stat):
        dd = cKDTree(x[stat]).query(xm, k=1, distance_upper_bound=a.link * A)[0]
        touch = np.isfinite(dd)
    cnt = np.bincount(lab)
    big = np.nonzero(cnt >= a.nmin)[0]
    rows = []
    for c in big:
        s = lab == c
        idx = mov[s]
        mc = m[idx].sum()
        com = (m[idx, None] * x[idx]).sum(0) / mc
        ext = np.ptp(x[idx], axis=0)
        row = dict(id=int(c), N=int(s.sum()), mass=mc, size=float(np.max(ext)),
                   dist=float(np.linalg.norm(com - xc)), v=float(np.median(vr[idx])),
                   matrix=float(m[idx][mat[idx] == a.matrix_mat].sum() / mc),
                   nn=float(np.median(ratio[s])),
                   rho=float(np.median(rho[s] / rho0[s])) if (rho is not None and rho0 is not None) else np.nan,
                   bound=float(m[idx][b[idx]].sum() / mc), attached=int(touch[s].any()),
                   x=com[0], y=com[1], z=com[2])
        rows.append(row)
    rows.sort(key=lambda r_: -r_["mass"])
    print(f"groups with >= {a.nmin} particles: {len(rows)} "
          f"({sum(r_['attached'] for r_ in rows)} still attached to the resting body)")
    if rows:
        print("\n  rank        N      mass[kg]  size[m]  dist[m]  v_med[m/s]  matrix%  nn/nn0  rho/rho0  bound%  att")
        for k, r_ in enumerate(rows[:25]):
            print(f"  {k:4d} {r_['N']:9d}  {r_['mass']:10.3e}  {r_['size']:7.1f}  {r_['dist']:7.1f}  "
                  f"{r_['v']:9.3f}   {100*r_['matrix']:6.1f}  {r_['nn']:6.2f}  {r_['rho']:7.2f}  {100*r_['bound']:6.1f}"
                  f"   {'y' if r_['attached'] else '-'}")
        det = [r_ for r_ in rows if not r_["attached"]]
        if det:
            w = np.array([r_["mass"] for r_ in det])
            mf = np.array([r_["matrix"] for r_ in det])
            nn = np.array([r_["nn"] for r_ in det])
            print(f"\n  detached groups: {len(det)}, mass {w.sum():.3e} kg, "
                  f"mass-weighted matrix fraction {100*np.average(mf, weights=w):.1f} %, "
                  f"median nn/nn0 {np.median(nn):.2f}")
            print("  reading: matrix-rich compact clumps (nn/nn0 <~ 1, rho/rho0 >~ 1) are suspicious for a")
            print("  cohesionless matrix; boulder-rich clumps or dilute groups (nn/nn0 > 1) are fine")
    if a.clumps_out and rows:
        with open(a.clumps_out, "w") as f:
            keys = list(rows[0].keys())
            f.write(",".join(keys) + "\n")
            for r_ in rows:
                f.write(",".join(f"{r_[k]:.6g}" if isinstance(r_[k], float) else str(r_[k]) for k in keys) + "\n")
        print(f"\n  clump table -> {a.clumps_out}")
    print(f"\ndone in {_time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
