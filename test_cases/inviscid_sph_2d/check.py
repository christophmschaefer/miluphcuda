#!/usr/bin/env python3
"""
Checks of test_cases/inviscid_sph_2d (INVISCID_SPH in 2D, with TENSORIAL_CORRECTION).

    python3 check.py [DIR]        (DIR: directory with run_inviscid_{gresho,sod2d} and, optionally,
                                   run_standard_{gresho,sod2d}; default: this directory)

Uses the last snapshot of each run, without the frozen outer layers (materials 2 and 3); prints PASS or FAIL per
check, exit status 1 if one fails; plots gresho.png and sod2d.png.

parameter.h sets INVISCID_SPH_START_ALPHA_MIN: alpha starts at alpha_min = 0, so every nonzero av_alpha comes from
the shock indicator.

Gresho vortex (t = 1): a stationary vortex, a shear flow without shocks; the viscosity must stay off there and the
vortex must keep its velocity profile much better than with the constant viscosity (alpha = 1, beta = 2):
  G1. alpha in the vortex (r < 0.4): mass-weighted mean below 0.1 and maximum below 0.3 (alpha_max = 1)
  G2. velocity profile: L1 error of the azimuthal velocity (mean in rings of width 0.02) over r < 0.5 at most
      0.5 times that of the standard run (only if run_standard_gresho exists)
  G3. peak of the azimuthal velocity profile at least 0.7 (exact 1 at r = 0.2)
Planar Sod shock tube in 2D (t = 0.6), evaluated in the strip |y| < 0.15 (away from the free edges), against the
exact Riemann solution:
  S1. shock position within 2 h of the exact one
  S2. post-shock density plateau within 3 % of the exact value
  S3. alpha in the undisturbed left gas: mean below 0.01
  S4. alpha at the shock: maximum within 3 h of it at least 0.15
  S5. alpha in the rarefaction (head + 3 h to tail - 3 h): maximum below 0.05; in 2D the limiter xi must suppress
      the indicator in the expansion (R ~ 1), unlike in 1D
  S6. L1 error of the density over [-0.3, 1.7] not larger than 1.25 times that of the standard run (if it exists)
  S7. decay behind the shock: mean alpha between contact + 3 h and shock - 5 h at most 0.7 times the value of S4
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

ALPHA_MAX, DECAY_L = 1.0, 0.05
SOD_GAMMA = 1.4
SOD_LEFT, SOD_RIGHT, SOD_X0 = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.5


def load(run_dir, name):
    fns = sorted(glob.glob(os.path.join(run_dir, f"{name}.*.h5")))
    if not fns:
        return None
    with h5py.File(fns[-1], "r") as f:
        d = {k: f[k][...] for k in ("x", "v", "rho", "p", "m", "sml", "material_type") if k in f}
        d["av_alpha"] = f["av_alpha"][...] if "av_alpha" in f else None
        d["time"] = float(np.ravel(f["time"][...])[0])
    if "material_type" in d:   # without the frozen outer layers (materials 2, 3)
        gas = d.pop("material_type") < 2
        for k in ("x", "v", "rho", "p", "m", "sml", "av_alpha"):
            if d.get(k) is not None:
                d[k] = d[k][gas]
    d["file"] = fns[-1]
    return d


# ---- Gresho vortex ----
def gresho_vphi(r):
    return np.where(r < 0.2, 5 * r, np.where(r < 0.4, 2 - 5 * r, 0.0))


def gresho_profile(d, dr=0.02, rmax=0.5):
    x, y = d["x"][:, 0], d["x"][:, 1]
    r = np.hypot(x, y)
    vphi = (x * d["v"][:, 1] - y * d["v"][:, 0]) / np.where(r > 0, r, 1.0)
    edges = np.arange(0.0, rmax + 0.5 * dr, dr)
    k = np.digitize(r, edges) - 1
    m = (k >= 0) & (k < len(edges) - 1)
    cnt = np.bincount(k[m], minlength=len(edges) - 1)
    prof = np.bincount(k[m], vphi[m], minlength=len(edges) - 1) / np.maximum(cnt, 1)
    rc = 0.5 * (edges[1:] + edges[:-1])
    return rc, prof, r


def check_gresho(runs, checks, plots):
    d = runs.get("inviscid")
    if d is None:
        checks.append(("G: run_inviscid_gresho", "no output", False))
        return
    rc, prof, r = gresho_profile(d)
    exact = gresho_vphi(rc)
    l1 = np.mean(np.abs(prof - exact))
    print(f"Gresho, t = {d['time']:g}: L1(v_phi) over r < 0.5: INVISCID_SPH {l1:.4f}", end="")
    s = runs.get("standard")
    if s is not None:
        rcs, profs, _ = gresho_profile(s)
        l1s = np.mean(np.abs(profs - exact))
        print(f", standard {l1s:.4f}")
    else:
        print()
    m = r < 0.4
    am = np.average(d["av_alpha"][m], weights=d["m"][m]) if "m" in d else np.mean(d["av_alpha"][m])
    ax = np.max(d["av_alpha"][m])
    checks.append(("G1 alpha in the vortex", f"mass-weighted mean {am:.4f} < {0.1 * ALPHA_MAX:g}, "
                   f"max {ax:.4f} < {0.3 * ALPHA_MAX:g}", am < 0.1 * ALPHA_MAX and ax < 0.3 * ALPHA_MAX))
    if s is not None:
        checks.append(("G2 velocity profile vs standard viscosity", f"L1 {l1:.4f} <= 0.5 x {l1s:.4f}",
                       l1 <= 0.5 * l1s))
    checks.append(("G3 peak of v_phi", f"{prof.max():.3f} >= 0.7", prof.max() >= 0.7))
    plots["gresho"] = (rc, exact, {k: gresho_profile(v)[1] for k, v in runs.items() if v is not None}, d)


# ---- Sod shock tube ----
def sound(rho, p):
    return np.sqrt(SOD_GAMMA * p / rho)


def pressure_function(p, rho_k, p_k):
    c_k = sound(rho_k, p_k)
    if p > p_k:
        a = 2.0 / ((SOD_GAMMA + 1.0) * rho_k)
        b = (SOD_GAMMA - 1.0) / (SOD_GAMMA + 1.0) * p_k
        q = np.sqrt(a / (p + b))
        return (p - p_k) * q, q * (1.0 - 0.5 * (p - p_k) / (b + p))
    pr = p / p_k
    f = 2.0 * c_k / (SOD_GAMMA - 1.0) * (pr ** ((SOD_GAMMA - 1.0) / (2.0 * SOD_GAMMA)) - 1.0)
    return f, pr ** (-(SOD_GAMMA + 1.0) / (2.0 * SOD_GAMMA)) / (rho_k * c_k)


def sod_exact(x, t):
    (rl, vl, pl), (rr, vr, pr) = SOD_LEFT, SOD_RIGHT
    ps = 0.5 * (pl + pr)
    for _ in range(100):
        fl, dl = pressure_function(ps, rl, pl)
        fr, dr = pressure_function(ps, rr, pr)
        dp = (fl + fr + vr - vl) / (dl + dr)
        ps = max(ps - dp, 1e-12)
        if abs(dp) < 1e-14 * ps:
            break
    fl, _ = pressure_function(ps, rl, pl)
    fr, _ = pressure_function(ps, rr, pr)
    vs = 0.5 * (vl + vr) + 0.5 * (fr - fl)
    g = SOD_GAMMA
    cl = sound(rl, pl)
    rho_ls = rl * (ps / pl) ** (1.0 / g)
    cls = sound(rho_ls, ps)
    rho_rs = rr * (ps / pr + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pr + 1.0)
    s_shock = vr + sound(rr, pr) * np.sqrt((g + 1) / (2 * g) * ps / pr + (g - 1) / (2 * g))
    head, tail = vl - cl, vs - cls
    xi = (np.asarray(x, dtype=float) - SOD_X0) / t
    rho, v, p = np.empty_like(xi), np.empty_like(xi), np.empty_like(xi)
    m = xi < head
    rho[m], v[m], p[m] = rl, vl, pl
    m = (xi >= head) & (xi < tail)
    v[m] = 2 / (g + 1) * (cl + 0.5 * (g - 1) * vl + xi[m])
    c = 2 / (g + 1) * (cl + 0.5 * (g - 1) * (vl - xi[m]))
    rho[m] = rl * (c / cl) ** (2 / (g - 1))
    p[m] = pl * (c / cl) ** (2 * g / (g - 1))
    m = (xi >= tail) & (xi < vs)
    rho[m], v[m], p[m] = rho_ls, vs, ps
    m = (xi >= vs) & (xi < s_shock)
    rho[m], v[m], p[m] = rho_rs, vs, ps
    m = xi >= s_shock
    rho[m], v[m], p[m] = rr, vr, pr
    return rho, v, p, dict(head=SOD_X0 + head * t, tail=SOD_X0 + tail * t, contact=SOD_X0 + vs * t,
                           shock=SOD_X0 + s_shock * t, rho_rs=rho_rs)


def strip(d, ymax=0.15):
    m = np.abs(d["x"][:, 1]) < ymax
    o = np.argsort(d["x"][m, 0])
    out = {"x": d["x"][m, 0][o], "v": d["v"][m, 0][o], "rho": d["rho"][m][o], "p": d["p"][m][o],
           "time": d["time"], "h": float(np.median(d["sml"]))}
    out["av_alpha"] = d["av_alpha"][m][o] if d["av_alpha"] is not None else None
    return out


def shock_position(s, w):
    level = 0.5 * (w["rho_rs"] + SOD_RIGHT[0])
    sel = (s["x"] > w["contact"]) & (s["x"] < w["shock"] + 0.3)
    xs, rs = s["x"][sel], s["rho"][sel]
    # the strip has several particles per x: average in bins of the lattice spacing
    xb = np.round(xs / 0.01) * 0.01
    ux, inv = np.unique(xb, return_inverse=True)
    rb = np.bincount(inv, rs) / np.bincount(inv)
    above = np.nonzero(rb > level)[0]
    if len(above) == 0:
        return np.nan
    k = above[-1]
    if k + 1 >= len(ux):
        return ux[k]
    return ux[k] + (level - rb[k]) * (ux[k + 1] - ux[k]) / (rb[k + 1] - rb[k])


def check_sod(runs, checks, plots):
    d = runs.get("inviscid")
    if d is None:
        checks.append(("S: run_inviscid_sod2d", "no output", False))
        return
    s = strip(d)
    t, h, x, al = s["time"], s["h"], s["x"], s["av_alpha"]
    _, _, _, w = sod_exact(x, t)
    errs = {}
    for k, r in runs.items():
        if r is None:
            continue
        st = strip(r)
        re_, ve, pe, _ = sod_exact(st["x"], st["time"])
        sel = (st["x"] >= -0.3) & (st["x"] <= 1.7)
        errs[k] = np.mean(np.abs(st["rho"][sel] - re_[sel]))
    print(f"Sod 2D, t = {t:g}, h = {h:g}: L1(rho) over [-0.3, 1.7], |y| < 0.15: "
          + ", ".join(f"{k} {v:.5f}" for k, v in errs.items()))
    xs = shock_position(s, w)
    checks.append(("S1 shock position", f"{xs:.4f}, exact {w['shock']:.4f}, |diff| <= 2 h = {2 * h:.4f}",
                   abs(xs - w["shock"]) <= 2 * h))
    m = (x > w["contact"] + 3 * h) & (x < w["shock"] - 3 * h)
    plateau = np.mean(s["rho"][m])
    checks.append(("S2 post-shock density", f"{plateau:.4f}, exact {w['rho_rs']:.4f}, within 3 %",
                   abs(plateau / w["rho_rs"] - 1) <= 0.03))
    m = (x < w["head"] - 3 * h) & (x > -0.85 + 3 * h)
    v = np.mean(al[m])
    checks.append(("S3 alpha in the undisturbed gas", f"mean {v:.4f} < {0.01 * ALPHA_MAX:g}", v < 0.01 * ALPHA_MAX))
    m = np.abs(x - w["shock"]) <= 3 * h
    a_shock = np.max(al[m])
    checks.append(("S4 alpha at the shock", f"max {a_shock:.3f} >= {0.15 * ALPHA_MAX:g}", a_shock >= 0.15 * ALPHA_MAX))
    m = (x > w["head"] + 3 * h) & (x < w["tail"] - 3 * h)
    v = np.max(al[m])
    checks.append(("S5 alpha in the rarefaction", f"max {v:.4f} < {0.05 * ALPHA_MAX:g}", v < 0.05 * ALPHA_MAX))
    if "standard" in errs:
        checks.append(("S6 L1(rho) vs standard viscosity",
                       f"{errs['inviscid']:.5f} <= 1.25 x {errs['standard']:.5f}",
                       errs["inviscid"] <= 1.25 * errs["standard"]))
    m = (x > w["contact"] + 3 * h) & (x < w["shock"] - 5 * h)
    v = np.mean(al[m])
    checks.append(("S7 decay behind the shock", f"mean {v:.3f} <= 0.7 x {a_shock:.3f}", v <= 0.7 * a_shock))
    plots["sod2d"] = (w, {k: strip(r) for k, r in runs.items() if r is not None})


def plot(plots, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if "gresho" in plots:
        rc, exact, profs, d = plots["gresho"]
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
        rr = np.linspace(0, 0.5, 500)
        axs[0].plot(rr, gresho_vphi(rr), "k-", lw=1, label="exact")
        for k, pr in profs.items():
            axs[0].plot(rc, pr, "o-", ms=3, label=k)
        axs[0].set(xlabel="r", ylabel="v_phi (ring mean)", title=f"Gresho vortex, t = {d['time']:g}")
        axs[0].legend()
        r = np.hypot(d["x"][:, 0], d["x"][:, 1])
        m = r < 0.6
        sc = axs[1].scatter(d["x"][m, 0], d["x"][m, 1], c=d["av_alpha"][m], s=4, cmap="viridis", vmin=0,
                            vmax=max(0.05, d["av_alpha"][m].max()))
        fig.colorbar(sc, ax=axs[1], label="av_alpha")
        axs[1].set(aspect="equal", title="av_alpha (INVISCID_SPH)")
        fig.savefig(os.path.join(outdir, "gresho.png"), dpi=130)
        print(f"wrote {os.path.join(outdir, 'gresho.png')}")
    if "sod2d" in plots:
        w, strips = plots["sod2d"]
        t = next(iter(strips.values()))["time"]
        xx = np.linspace(-0.85, 2.35, 2000)
        re_, ve, pe, _ = sod_exact(xx, t)
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, constrained_layout=True)
        for ax, k, ex in zip(axs.flat[:3], ("rho", "v", "p"), (re_, ve, pe)):
            ax.plot(xx, ex, "k-", lw=1, label="exact")
            for name, s in strips.items():
                ax.plot(s["x"], s[k], ".", ms=1.5, label=name)
            ax.set_ylabel(k)
        axs[0, 0].legend(fontsize=8, markerscale=5)
        s = strips["inviscid"]
        axs[1, 1].plot(s["x"], s["av_alpha"], ".", ms=1.5)
        for k in ("head", "tail", "contact", "shock"):
            axs[1, 1].axvline(w[k], color="0.7", lw=0.8)
        axs[1, 1].set(ylabel="av_alpha (INVISCID_SPH)", ylim=(-0.05, 1.1 * ALPHA_MAX))
        axs[0, 0].set_xlim(-0.85, 2.35)
        fig.suptitle(f"planar Sod shock tube in 2D, |y| < 0.15, t = {t:g}")
        fig.savefig(os.path.join(outdir, "sod2d.png"), dpi=130)
        print(f"wrote {os.path.join(outdir, 'sod2d.png')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    checks, plots = [], {}
    for name, fn in (("gresho", check_gresho), ("sod2d", check_sod)):
        runs = {v: load(os.path.join(a.dir, f"run_{v}_{name}"), name) for v in ("inviscid", "standard")}
        if runs["inviscid"] is not None and runs["inviscid"]["av_alpha"] is None:
            sys.exit(f"no av_alpha in {runs['inviscid']['file']}: built with INVISCID_SPH 0?")
        fn(runs, checks, plots)
    ok = True
    for name, txt, res in checks:
        ok &= bool(res)
        print(f"  {'PASS' if res else 'FAIL'}  {name}: {txt}")
    plot(plots, a.dir)
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
