#!/usr/bin/env python3
"""
Initial conditions of test_cases/inviscid_sph_2d, ascii for DIM 2, INTEGRATE_DENSITY 0 (density by
kernel summation), INTEGRATE_ENERGY 1 (columns x y vx vy m e material_type), square lattices:

gresho.0000  Gresho-Chan vortex (Gresho & Chan 1990), material 0 (gamma = 5/3), rho = 1, stationary solution
             v_phi = 5 r (r < 0.2), 2 - 5 r (0.2 < r < 0.4), 0 outside,
             p = 5 + 12.5 r^2, 9 + 12.5 r^2 - 20 r + 4 ln(5 r), 3 + 4 ln 2;
             dx = 0.02 on [-3.8, 3.8]^2 (144400 particles), large enough that the rarefaction from the free edges
             does not reach r = 0.5 before t = 1 (smoothing length 4 dx in material.cfg)
sod2d.0000   planar Sod shock tube, material 1 (gamma = 1.4): rho = 1, p = 1 for x < 0.5, rho = 0.125, p = 0.1
             for x > 0.5, v = 0; dx = 0.01 on [-1.5, 2.5] x [-1, 1] (80000 particles), masses rho dx^2, large
             enough that the rarefactions from the free ends and edges do not reach the strip |y| < 0.15 between the
             rarefaction head and the shock before t = 0.6 (smoothing length 4 dx in material.cfg)
"""
import numpy as np


def lattice(x0, x1, y0, y1, dx):
    x = np.arange(x0 + 0.5 * dx, x1, dx)
    y = np.arange(y0 + 0.5 * dx, y1, dx)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X.ravel(), Y.ravel()


def write(fn, x, y, vx, vy, m, e, mt):
    np.savetxt(fn, np.c_[x, y, vx, vy, m, e, np.full(len(x), mt)], fmt=["%.10e"] * 6 + ["%d"])
    print(f"wrote {fn} with {len(x)} particles")


def gresho():
    gamma, dx, L = 5.0 / 3.0, 0.02, 3.8
    x, y = lattice(-L, L, -L, L, dx)
    r = np.hypot(x, y)
    vphi = np.where(r < 0.2, 5 * r, np.where(r < 0.4, 2 - 5 * r, 0.0))
    with np.errstate(divide="ignore"):
        p = np.where(r < 0.2, 5 + 12.5 * r**2,
                     np.where(r < 0.4, 9 + 12.5 * r**2 - 20 * r + 4 * np.log(np.maximum(5 * r, 1e-300)),
                              3 + 4 * np.log(2)))
    rs = np.where(r > 0, r, 1.0)
    vx, vy = -vphi * y / rs, vphi * x / rs
    rho = np.ones_like(x)
    write("gresho.0000", x, y, vx, vy, rho * dx * dx, p / ((gamma - 1) * rho), 0)


def sod2d():
    gamma, dx = 1.4, 0.01
    x, y = lattice(-1.5, 2.5, -1.0, 1.0, dx)
    left = x < 0.5
    rho = np.where(left, 1.0, 0.125)
    p = np.where(left, 1.0, 0.1)
    z = np.zeros_like(x)
    write("sod2d.0000", x, y, z, z, rho * dx * dx, p / ((gamma - 1) * rho), 1)


if __name__ == "__main__":
    gresho()
    sod2d()
