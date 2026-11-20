#!/usr/bin/env python3
"""
Checks the Sod shock tube of test_cases/inviscid_sph against the exact solution.

    python3 check.py RUN_DIR [RUN_DIR_STANDARD] [--out sod.png]

RUN_DIR: output of the executable with INVISCID_SPH 1 (sod.*.h5 with av_alpha); RUN_DIR_STANDARD (optional): the
same with INVISCID_SPH 0 (constant alpha = 1, beta = 2), for comparison. Uses the last snapshot of each run.

Exact solution: Riemann solver for the ideal gas (Toro 2009, ch. 4). Checks of the INVISCID_SPH run, each printed
with PASS or FAIL (exit status 1 if one fails):
  1. shock position: within 2 h of the exact one (h: smoothing length)
  2. post-shock density plateau: mean between contact + 3 h and shock - 3 h within 3 % of the exact value
  3. decay of alpha: in the undisturbed left gas (x < rarefaction head - 3 h, away from the free end), alpha starts
     at alpha_max and decays without a source, d alpha/dt = -alpha/tau, tau = h/(2 l c_L) (alpha_min = 0); away
     from the free end of the tube (x > -1 + c_L t + 3 h):
     mean av_alpha within 10 % of alpha_max exp(-t/tau)
  4. alpha at the shock: max av_alpha within 3 h of the shock >= 2 alpha_max exp(-t/tau_R), twice the value the
     undisturbed gas ahead of the shock has after the decay (tau_R = h/(2 l c_R)): the indicator switched it on
  5. alpha in the rarefaction: max av_alpha between rarefaction head + 3 h and tail - 3 h below 2 times the value
     of 3. (a smooth expansion must not switch on the viscosity)
  6. L1 error of the density over [0, 1]: not larger than 1.25 times that of the standard run (if given)
Printed for information: L1 errors of density, velocity and pressure over [0, 1] of both runs.
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

GAMMA = 1.4
LEFT = (1.0, 0.0, 1.0)      # rho, v, p
RIGHT = (0.125, 0.0, 0.1)
X0 = 0.5


def sound(rho, p):
    return np.sqrt(GAMMA * p / rho)


def pressure_function(p, rho_k, p_k):
    """f_K(p) and its derivative (Toro eq. 4.6, 4.7)"""
    c_k = sound(rho_k, p_k)
    if p > p_k:   # shock
        a = 2.0 / ((GAMMA + 1.0) * rho_k)
        b = (GAMMA - 1.0) / (GAMMA + 1.0) * p_k
        q = np.sqrt(a / (p + b))
        return (p - p_k) * q, q * (1.0 - 0.5 * (p - p_k) / (b + p))
    pr = p / p_k   # rarefaction
    f = 2.0 * c_k / (GAMMA - 1.0) * (pr ** ((GAMMA - 1.0) / (2.0 * GAMMA)) - 1.0)
    return f, pr ** (-(GAMMA + 1.0) / (2.0 * GAMMA)) / (rho_k * c_k)


def star_state():
    (rl, vl, pl), (rr, vr, pr) = LEFT, RIGHT
    p = 0.5 * (pl + pr)
    for _ in range(100):
        fl, dl = pressure_function(p, rl, pl)
        fr, dr = pressure_function(p, rr, pr)
        dp = (fl + fr + vr - vl) / (dl + dr)
        p = max(p - dp, 1e-12)
        if abs(dp) < 1e-14 * p:
            break
    fl, _ = pressure_function(p, rl, pl)
    fr, _ = pressure_function(p, rr, pr)
    return p, 0.5 * (vl + vr) + 0.5 * (fr - fl)


def exact(x, t):
    """rho, v, p at positions x and time t, and the wave positions (rarefaction head, tail, contact, shock);
    for the Sod set-up (left rarefaction, right shock)"""
    (rl, vl, pl), (rr, vr, pr) = LEFT, RIGHT
    ps, vs = star_state()
    cl = sound(rl, pl)
    rho_ls = rl * (ps / pl) ** (1.0 / GAMMA)
    cls = sound(rho_ls, ps)
    g = (GAMMA - 1.0) / (GAMMA + 1.0)
    rho_rs = rr * (ps / pr + g) / (g * ps / pr + 1.0)
    s_shock = vr + sound(rr, pr) * np.sqrt((GAMMA + 1.0) / (2.0 * GAMMA) * ps / pr + (GAMMA - 1.0) / (2.0 * GAMMA))
    head, tail = vl - cl, vs - cls
    xi = (np.asarray(x, dtype=float) - X0) / t
    rho, v, p = np.empty_like(xi), np.empty_like(xi), np.empty_like(xi)
    m = xi < head
    rho[m], v[m], p[m] = rl, vl, pl
    m = (xi >= head) & (xi < tail)
    v[m] = 2.0 / (GAMMA + 1.0) * (cl + 0.5 * (GAMMA - 1.0) * vl + xi[m])
    c = 2.0 / (GAMMA + 1.0) * (cl + 0.5 * (GAMMA - 1.0) * (vl - xi[m]))
    rho[m] = rl * (c / cl) ** (2.0 / (GAMMA - 1.0))
    p[m] = pl * (c / cl) ** (2.0 * GAMMA / (GAMMA - 1.0))
    m = (xi >= tail) & (xi < vs)
    rho[m], v[m], p[m] = rho_ls, vs, ps
    m = (xi >= vs) & (xi < s_shock)
    rho[m], v[m], p[m] = rho_rs, vs, ps
    m = xi >= s_shock
    rho[m], v[m], p[m] = rr, vr, pr
    waves = dict(head=X0 + head * t, tail=X0 + tail * t, contact=X0 + vs * t, shock=X0 + s_shock * t,
                 rho_ls=rho_ls, rho_rs=rho_rs, p_star=ps, v_star=vs)
    return rho, v, p, waves


def load(run_dir):
    fns = sorted(glob.glob(os.path.join(run_dir, "sod.*.h5")))
    if not fns:
        sys.exit(f"no sod.*.h5 in {run_dir}")
    with h5py.File(fns[-1], "r") as f:
        d = {k: f[k][...] for k in ("x", "v", "rho", "p", "sml") if k in f}
        d["av_alpha"] = f["av_alpha"][...] if "av_alpha" in f else None
        d["time"] = float(np.ravel(f["time"][...])[0])
    o = np.argsort(d["x"].reshape(-1))
    for k in ("x", "v", "rho", "p", "sml", "av_alpha"):
        if d.get(k) is not None:
            d[k] = d[k].reshape(len(o), -1)[:, 0][o]
    d["file"] = fns[-1]
    return d


def l1(d, rho_e, v_e, p_e):
    m = (d["x"] >= 0.0) & (d["x"] <= 1.0)
    return tuple(np.mean(np.abs(a[m] - b[m])) for a, b in ((d["rho"], rho_e), (d["v"], v_e), (d["p"], p_e)))


def shock_position(d, w):
    """first crossing of rho = (rho_rs + rho_r)/2 from the right, interpolated"""
    level = 0.5 * (w["rho_rs"] + RIGHT[0])
    x, rho = d["x"], d["rho"]
    m = (x > w["contact"]) & (x < w["shock"] + 0.3)
    xs, rs = x[m], rho[m]
    above = np.nonzero(rs > level)[0]
    if len(above) == 0:
        return np.nan
    k = above[-1]
    if k + 1 >= len(xs):
        return xs[k]
    return xs[k] + (level - rs[k]) * (xs[k + 1] - xs[k]) / (rs[k + 1] - rs[k])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("standard", nargs="?")
    ap.add_argument("--alpha-max", type=float, default=1.0, help="alpha of the material (material.cfg)")
    ap.add_argument("--decay-l", type=float, default=0.05, help="INVISCID_SPH_DECAY_L of parameter.h")
    ap.add_argument("--out", default="sod.png")
    a = ap.parse_args()

    runs = {"INVISCID_SPH": load(a.run)}
    if a.standard:
        runs["standard"] = load(a.standard)
    d = runs["INVISCID_SPH"]
    if d["av_alpha"] is None:
        sys.exit(f"no av_alpha in {d['file']}: built with INVISCID_SPH 0?")
    t = d["time"]
    h = float(np.median(d["sml"]))
    rho_e, v_e, p_e, w = exact(d["x"], t)
    print(f"t = {t:g}, h = {h:g}; exact: rarefaction {w['head']:.4f} .. {w['tail']:.4f}, contact {w['contact']:.4f}, "
          f"shock {w['shock']:.4f}, p* = {w['p_star']:.5f}, v* = {w['v_star']:.5f}, rho*L = {w['rho_ls']:.5f}, "
          f"rho*R = {w['rho_rs']:.5f}")
    errs = {}
    for name, r in runs.items():
        re_, ve, pe, _ = exact(r["x"], r["time"])
        errs[name] = l1(r, re_, ve, pe)
        print(f"{name:>13}: {os.path.basename(r['file'])}  L1 over [0, 1]: rho {errs[name][0]:.5f}  "
              f"v {errs[name][1]:.5f}  p {errs[name][2]:.5f}  shock at {shock_position(r, w):.4f}")

    x, al = d["x"], d["av_alpha"]
    amax = a.alpha_max
    checks = []
    xs = shock_position(d, w)
    checks.append(("shock position", f"{xs:.4f}, exact {w['shock']:.4f}, |diff| <= 2 h = {2 * h:.4f}",
                   abs(xs - w["shock"]) <= 2.0 * h))
    m = (x > w["contact"] + 3 * h) & (x < w["shock"] - 3 * h)
    plateau = np.mean(d["rho"][m]) if m.any() else np.nan
    checks.append(("post-shock density", f"{plateau:.4f}, exact {w['rho_rs']:.4f}, within 3 %",
                   abs(plateau / w["rho_rs"] - 1.0) <= 0.03))
    c_l = np.sqrt(GAMMA * LEFT[2] / LEFT[0])
    tau = h / (2.0 * a.decay_l * c_l)
    a_decay = amax * np.exp(-t / tau)
    # the free end of the tube expands as well, its rarefaction moves in with c_L
    m = (x < w["head"] - 3 * h) & (x > x.min() + c_l * t + 3 * h)
    v = np.mean(al[m]) if m.any() else np.nan
    checks.append(("decay of alpha in the undisturbed gas", f"mean {v:.4f}, expected {a_decay:.4f} (tau = {tau:.4f})",
                   abs(v / a_decay - 1.0) <= 0.1))
    a_ahead = amax * np.exp(-t * 2.0 * a.decay_l * np.sqrt(GAMMA * RIGHT[2] / RIGHT[0]) / h)
    m = np.abs(x - w["shock"]) <= 3 * h
    v = np.max(al[m]) if m.any() else np.nan
    checks.append(("alpha at the shock", f"max {v:.3f} >= 2 x {a_ahead:.3f} (decayed value ahead of the shock)",
                   v >= 2.0 * a_ahead))
    m = (x > w["head"] + 3 * h) & (x < w["tail"] - 3 * h)
    v = np.max(al[m]) if m.any() else np.nan
    checks.append(("alpha in the rarefaction", f"max {v:.4f} < {2 * a_decay:.4f}", v < 2.0 * a_decay))
    if "standard" in errs:
        e1, e0 = errs["INVISCID_SPH"][0], errs["standard"][0]
        checks.append(("L1(rho) vs standard viscosity", f"{e1:.5f} <= 1.25 x {e0:.5f}", e1 <= 1.25 * e0))
    ok = True
    for name, txt, res in checks:
        ok &= bool(res)
        print(f"  {'PASS' if res else 'FAIL'}  {name}: {txt}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xx = np.linspace(-0.2, 1.2, 2000)
        re_, ve, pe, _ = exact(xx, t)
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, constrained_layout=True)
        for ax, k, ex, lab in zip(axs.flat[:3], ("rho", "v", "p"), (re_, ve, pe), ("density", "velocity", "pressure")):
            ax.plot(xx, ex, "k-", lw=1, label="exact")
            for name, r in runs.items():
                ax.plot(r["x"], r[k], ".", ms=2, label=name)
            ax.set_ylabel(lab)
        axs[0, 0].legend(fontsize=8, markerscale=4)
        ax = axs[1, 1]
        ax.plot(x, al, ".", ms=2, color="C0")
        for k in ("head", "tail", "contact", "shock"):
            ax.axvline(w[k], color="0.7", lw=0.8)
        ax.set_ylabel("av_alpha (INVISCID_SPH)")
        ax.set_ylim(-0.05, 1.1 * amax)
        for ax in axs[1]:
            ax.set_xlabel("x")
        axs[0, 0].set_xlim(-0.2, 1.2)
        fig.suptitle(f"Sod shock tube, t = {t:g}")
        fig.savefig(a.out, dpi=130)
        print(f"wrote {a.out}")
    except ImportError:
        pass
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
