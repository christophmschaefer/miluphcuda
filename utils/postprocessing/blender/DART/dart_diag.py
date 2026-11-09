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
  6. settling (printed as section 5): is the simulated time long enough?
     The resting body is the largest connected set of particles with
     v_rel < --vset; its surface radius is tabulated per direction. Everything
     else is split into material still moving on/in the body (crater flow) and
     material in flight (bound / escaping, inside / outside the Hill radius of
     Dimorphos in the Didymos system). For bound material in flight the time
     until it comes down is computed from a Kepler orbit around the point mass
     M_bound, together with the landing point (crater or elsewhere). The verdict
     gives the time after which less than --settle-frac of the target mass is
     still in the air. Material beyond r_H and escaping material is not
     followed: that is a job for the N-body integration with Didymos.

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


# ---------------------------------------------------------------------------
# settling: where is the material that still moves, and when does it come down
# ---------------------------------------------------------------------------
def body_component(xr, rest, cell, rmax):
    """Largest connected set of resting particles (grid of `cell`, 26-connectivity).

    xr: positions relative to the bound-body COM. Particles at rest but detached
    from the body (clumps near the apex of their arc) are not part of it.
    """
    from scipy import ndimage
    cand = np.nonzero(rest & (np.abs(xr).max(1) < rmax))[0]
    ijk = np.floor((xr[cand] + rmax) / cell).astype(np.int32)
    ng = int(np.ceil(2 * rmax / cell)) + 1
    occ = np.zeros((ng, ng, ng), bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    lab, nl = ndimage.label(occ, structure=np.ones((3, 3, 3), bool))
    if nl == 0:
        return np.zeros(len(xr), bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    big = np.argmax(sizes)
    body = np.zeros(len(xr), bool)
    body[cand[lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]] == big]] = True
    return body


class SurfaceMap:
    """Surface radius of the resting body per direction (equal-area bins in cos(theta), phi)."""

    def __init__(self, xr, body, nt=36, nphi=72):
        self.nt, self.np_ = nt, nphi
        r = np.linalg.norm(xr[body], axis=1)
        k = self._bin(xr[body])
        rs = np.full(nt * nphi, -1.0)
        np.maximum.at(rs, k, r)
        rs = rs.reshape(nt, nphi)
        # empty bins (should be rare): row median, else global median
        for i in range(nt):
            row = rs[i]
            if (row < 0).any():
                ok = row >= 0
                row[~ok] = np.median(row[ok]) if ok.any() else -1
        rs[rs < 0] = np.median(rs[rs >= 0])
        self.rs = rs

    def _bin(self, xr):
        rn = np.linalg.norm(xr, axis=1)
        rn = np.where(rn > 0, rn, 1.0)
        cz = np.clip(xr[:, 2] / rn, -1, 1)
        ph = np.arctan2(xr[:, 1], xr[:, 0])
        it = np.minimum(((cz + 1) / 2 * self.nt).astype(np.int64), self.nt - 1)
        ip = np.minimum(((ph + np.pi) / (2 * np.pi) * self.np_).astype(np.int64), self.np_ - 1)
        return it * self.np_ + ip

    def __call__(self, xr):
        return self.rs.ravel()[self._bin(xr)]


def kepler_landing(rr, vv, GM, surf, iters=2):
    """Time until a bound particle on a point-mass Kepler orbit comes down to the surface.

    rr, vv: position / velocity relative to the bound body (N, 3), E < 0.
    surf:   callable giving the surface radius for direction vectors.
    Returns dt [s] (inf if periapsis is above the surface) and the landing position.
    """
    r = np.linalg.norm(rr, axis=1)
    v2 = (vv**2).sum(1)
    E = 0.5 * v2 - GM / r
    a = -GM / (2 * E)
    h = np.cross(rr, vv)
    hn = np.linalg.norm(h, axis=1)
    ev = np.cross(vv, h) / GM - rr / r[:, None]
    e = np.minimum(np.linalg.norm(ev, axis=1), 1 - 1e-12)
    ok = e > 1e-9
    P = np.where(ok[:, None], ev / np.where(ok, e, 1)[:, None], rr / r[:, None])
    Q = np.cross(h, P) / np.where(hn > 0, hn, 1)[:, None]
    # radial orbit (h = 0): Q is irrelevant because sin-term vanishes with sqrt(1-e^2) -> 0
    n = np.sqrt(GM / a**3)
    out = (rr * vv).sum(1) > 0                        # outbound?
    E0 = np.arccos(np.clip((1 - r / a) / e, -1, 1))
    E0 = np.where(out, E0, 2 * np.pi - E0)
    M0 = E0 - e * np.sin(E0)
    Rl = surf(rr)                                    # first guess: surface below the particle
    for _ in range(iters):
        c1 = (1 - Rl / a) / e
        hit = ok & (a * (1 - e) < Rl) & (c1 <= 1)
        E1 = 2 * np.pi - np.arccos(np.clip(c1, -1, 1))   # inbound branch
        E1 = np.where(E1 < E0, E0, E1)                   # already at/below the surface
        xl = (a * (np.cos(E1) - e))[:, None] * P + (a * np.sqrt(1 - e**2) * np.sin(E1))[:, None] * Q
        Rl = np.where(hit, surf(xl), Rl)                 # surface where it actually lands
    dt = np.where(hit, (E1 - e * np.sin(E1) - M0) / n, np.inf)
    return dt, xl, hit


def settle_report(a, D, x, v, m, tgt, b, Mb, xc, vc, vr, r, R, A):
    """Section 5: is the simulated time long enough?"""
    print(f"\n== 5. settling: is the simulated time long enough? ==")
    t = D["t"]
    Mt = m[tgt].sum()
    xr = x - xc
    vv = v - vc
    GM = G * Mb
    Rvol = np.cbrt(3 * Mb / (4 * np.pi * 2117.0))
    rH = a.r_hill if a.r_hill else a.a_orbit * np.cbrt(Mb / (3 * a.m_didymos))

    rest = tgt & b & (vr <= a.vset)
    body = body_component(xr, rest, a.link * A, max(2.5 * Rvol, 1.2 * rH))
    surf = SurfaceMap(xr, body)
    Rs = surf(xr)
    hgt = r - Rs
    hs = a.hsurf * A
    print(f"resting body (connected, v < {100*a.vset:.0f} cm/s): {fmt_mass(m[body].sum(), Mt)}, "
          f"surface radius min/median/max {surf.rs.min():.1f} / {np.median(surf.rs):.1f} / {surf.rs.max():.1f} m")
    print(f"Hill radius in the Didymos system r_H = {rH:.0f} m"
          + ("" if a.r_hill else f"  (a = {a.a_orbit:.0f} m, M_Didymos = {a.m_didymos:.2e} kg)"))

    # crater centre: impact point moved with the body COM (no rotation in the run)
    if R is not None:
        xc0 = (R["m"][tgt, None] * R["x"][tgt]).sum(0) / R["m"][tgt].sum()
    else:
        xc0 = np.zeros(3)
    xcr = np.array(a.impact_point) - xc0           # relative to the COM

    free = tgt & ~body
    on = free & (hgt < hs)
    flow = on & (vr > a.vset)                       # moving on / in the body
    above = free & (hgt >= hs)
    ab_b, ab_e = above & b, above & ~b
    inH = r < rH
    cats = [("moving on/in the body (h < %.1f m)" % hs, flow),
            ("above surface, bound,    r < r_H", ab_b & inH),
            ("above surface, bound,    r > r_H", ab_b & ~inH),
            ("above surface, escaping, r < r_H", ab_e & inH),
            ("above surface, escaping, r > r_H", ab_e & ~inH)]
    print("\nunsettled material                         mass                     N     v_med[m/s]")
    for name, s in cats:
        ms = m[s].sum()
        vm = np.median(vr[s]) if s.any() else np.nan
        print(f"  {name:36s} {fmt_mass(ms, Mt)}  {s.sum():9d}  {vm:9.4f}")
    if flow.any():
        dcr = np.linalg.norm(xr[flow] - xcr, axis=1)
        fin = m[flow][dcr < a.crater_radius].sum() / m[flow].sum()
        print(f"  ({100*fin:.0f} % of the material moving on the body is within {a.crater_radius:.0f} m "
              f"of the crater centre)")

    # where does the slow material of section 1 sit?
    print("\nslow speed bins by location (% of the bin mass)")
    print("  speed [m/s]        resting   on body   bound<rH  bound>rH  escaping")
    for lo, hi in ((0.0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 0.3)):
        s = tgt & (vr >= lo) & (vr < hi)
        ms = m[s].sum()
        if ms == 0:
            continue
        f = [m[s & body].sum() + m[s & on & ~flow].sum(), m[s & flow].sum(),
             m[s & ab_b & inH].sum(), m[s & ab_b & ~inH].sum(), m[s & ab_e].sum()]
        print(f"  {lo:5.2f} .. {hi:<5.2f}    " + "  ".join(f"{100*z/ms:8.2f}" for z in f))

    # ballistic fall-back of bound material above the surface (point mass M_bound)
    sel = np.nonzero(ab_b)[0]
    print(f"\nballistic fall-back of bound material above the surface (point mass M_bound, from t = {t:.0f} s)")
    if len(sel) == 0:
        print("  none")
        dt = np.zeros(0)
    else:
        dt, xl, hit = kepler_landing(xr[sel], vv[sel], GM, surf)
        ms = m[sel]
        inside = inH[sel]
        incr = hit & (np.linalg.norm(xl - xcr, axis=1) < a.crater_radius)
        print("  lands within        all bound     r < r_H     into crater (< %.0f m)" % a.crater_radius)
        for lim in (100, 300, 1000, 3000, 10000, 30000):
            w = dt <= lim
            print(f"  {lim:7.0f} s       {100*ms[w].sum()/Mt:9.4f} %  {100*ms[w & inside].sum()/Mt:9.4f} %"
                  f"  {100*ms[w & incr].sum()/Mt:9.4f} %")
        w = np.isfinite(dt) & (dt > 30000)
        print(f"  later             {100*ms[w].sum()/Mt:9.4f} %  {100*ms[w & inside].sum()/Mt:9.4f} %"
              f"  {100*ms[w & incr].sum()/Mt:9.4f} %")
        w = ~np.isfinite(dt)
        print(f"  never (q > surf.) {100*ms[w].sum()/Mt:9.4f} %  {100*ms[w & inside].sum()/Mt:9.4f} %"
              "        (point mass; Didymos decides)")

    # verdict
    print("\nverdict (fractions of the target mass)")
    f_flow = m[flow].sum() / Mt
    print(f"  still flowing on the body:                   {100*f_flow:8.4f} %")
    tgt_frac = a.settle_frac
    if len(sel):
        use = inH[sel] & np.isfinite(dt)
        f_ret = ms[use].sum() / Mt
        f_cr = ms[use & incr].sum() / Mt
        print(f"  bound, r < r_H, still to come down:          {100*f_ret:8.4f} %"
              f"   (into the crater: {100*f_cr:.4f} %)")
        if use.any():
            o = np.argsort(dt[use])
            cm = np.cumsum(ms[use][o])
            q = [dt[use][o][min(np.searchsorted(cm, z * cm[-1]), len(o) - 1)] for z in (0.5, 0.9, 0.99)]
            print(f"  ... 50 / 90 / 99 % of it landed by t = " + " / ".join(f"{t+z:.0f}" for z in q) + " s")
            # time after which less than settle_frac of the target mass is still in the air
            rem = cm[-1] - cm
            k = np.searchsorted(-rem, -tgt_frac * Mt)
            t_need = t + (dt[use][o][min(k, len(o) - 1)] if f_ret > tgt_frac else 0.0)
        else:
            t_need = t
    else:
        f_ret, t_need = 0.0, t
    if f_flow < tgt_frac and f_ret < tgt_frac:
        print(f"  -> settled: both below {100*tgt_frac:.2g} % of the target mass. t = {t:.0f} s is long enough.")
    else:
        if f_ret >= tgt_frac:
            print(f"  -> fall-back drops below {100*tgt_frac:.2g} % of the target mass at t ~ {t_need:.0f} s "
                  "(point mass, ignores Didymos)")
        if f_flow >= tgt_frac:
            print(f"  -> {100*f_flow:.3f} % of the target mass still moves on the body (> {100*tgt_frac:.2g} %); "
                  "compare with an earlier dump to see whether it is dying out")
    print("  material beyond r_H and escaping material are left to the N-body integration with Didymos")


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
    g5 = ap.add_argument_group("5. settling")
    g5.add_argument("--no-settle", action="store_true", help="skip section 5")
    g5.add_argument("--vset", type=float, default=0.01, help="'at rest' below this speed [m/s]")
    g5.add_argument("--hsurf", type=float, default=2.0,
                    help="surface layer thickness [units of a]; below = on the body, above = in flight")
    g5.add_argument("--r-hill", type=float, help="Hill radius [m] (default: from --a-orbit, --m-didymos)")
    g5.add_argument("--a-orbit", type=float, default=1152.0, help="Dimorphos semi-major axis [m] (post impact)")
    g5.add_argument("--m-didymos", type=float, default=5.4e11, help="Didymos mass [kg]")
    g5.add_argument("--impact-point", type=float, nargs=3, default=[-8.06, -80.73, -12.62],
                    help="impact point in the initial frame [m]")
    g5.add_argument("--crater-radius", type=float, default=40.0,
                    help="radius around the impact point counted as crater [m]")
    g5.add_argument("--settle-frac", type=float, default=1e-3,
                    help="'settled' when less than this fraction of the target mass is unsettled")
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
        if not a.no_settle:
            settle_report(a, D, x, v, m, tgt, b, Mb, xc, vc, vr, r, R, A)
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
    if not a.no_settle:
        settle_report(a, D, x, v, m, tgt, b, Mb, xc, vc, vr, r, R, A)
    print(f"\ndone in {_time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
