#!/usr/bin/env python3
"""
Initial conditions of test_cases/inviscid_sph_2d, ascii for DIM 2, INTEGRATE_DENSITY 0 (density by kernel
summation), INTEGRATE_ENERGY 1 (columns x y vx vy m e material_type), square lattices. The outer layers are frozen
particles (materials 2 and 3, see boundary.cu of this test): they keep their initial state and confine the gas, so
that there are no free edges expanding into vacuum.

gresho.0000  Gresho-Chan vortex (Gresho & Chan 1990), gamma = 5/3, rho = 1, stationary solution
             v_phi = 5 r (r < 0.2), 2 - 5 r (0.2 < r < 0.4), 0 outside,
             p = 5 + 12.5 r^2, 9 + 12.5 r^2 - 20 r + 4 ln(5 r), 3 + 4 ln 2;
             dx = 0.02 on [-1, 1]^2 (10000 particles), material 0, frozen (material 2) where max(|x|, |y|) > 0.76
sod2d.0000   planar Sod shock tube, gamma = 1.4: rho = 1, p = 1 for x < 0.5, rho = 0.125, p = 0.1 for x > 0.5,
             v = 0; dx = 0.01 on [-1, 2.5] x [-1.1, 1.1] (77000 particles), masses rho dx^2, material 1, frozen
             (material 3) where x < -0.85, x > 2.35 or |y| > 0.95. Until t = 0.6 the waves do not reach the frozen
             ends, and disturbances from the frozen sides do not reach the strip |y| < 0.15 that check.py evaluates.
The smoothing lengths (4 dx) are in material.cfg.
"""
import numpy as np


def lattice(x0, x1, y0, y1, dx):
    x = np.arange(x0 + 0.5 * dx, x1, dx)
    y = np.arange(y0 + 0.5 * dx, y1, dx)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X.ravel(), Y.ravel()


def write(fn, x, y, vx, vy, m, e, mt):
    np.savetxt(fn, np.c_[x, y, vx, vy, m, e, mt], fmt=["%.10e"] * 6 + ["%d"])
    print(f"wrote {fn} with {len(x)} particles, {np.count_nonzero(mt >= 2)} of them frozen")


def gresho():
    gamma, dx = 5.0 / 3.0, 0.02
    x, y = lattice(-1.0, 1.0, -1.0, 1.0, dx)
    r = np.hypot(x, y)
    vphi = np.where(r < 0.2, 5 * r, np.where(r < 0.4, 2 - 5 * r, 0.0))
    p = np.where(r < 0.2, 5 + 12.5 * r**2,
                 np.where(r < 0.4, 9 + 12.5 * r**2 - 20 * r + 4 * np.log(np.maximum(5 * r, 1e-300)),
                          3 + 4 * np.log(2)))
    rs = np.where(r > 0, r, 1.0)
    vx, vy = -vphi * y / rs, vphi * x / rs
    rho = np.ones_like(x)
    mt = np.where(np.maximum(np.abs(x), np.abs(y)) > 0.76, 2, 0)
    write("gresho.0000", x, y, vx, vy, rho * dx * dx, p / ((gamma - 1) * rho), mt)


def sod2d():
    gamma, dx = 1.4, 0.01
    x, y = lattice(-1.0, 2.5, -1.1, 1.1, dx)
    left = x < 0.5
    rho = np.where(left, 1.0, 0.125)
    p = np.where(left, 1.0, 0.1)
    z = np.zeros_like(x)
    mt = np.where((x < -0.85) | (x > 2.35) | (np.abs(y) > 0.95), 3, 1)
    write("sod2d.0000", x, y, z, z, rho * dx * dx, p / ((gamma - 1) * rho), mt)


if __name__ == "__main__":
    gresho()
    sod2d()
