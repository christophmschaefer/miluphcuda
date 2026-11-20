#!/usr/bin/env python3
"""
Initial particle distribution for the gravity_quadrupole test case.

Two bodies at rest:
  A: sphere (disc for --dim 2), R = 1 km, centrally concentrated mass distribution
     (particle mass ~ 1 + 4 (1 - r^2/R^2))
  B: prolate ellipsoid (ellipse), semi-axes 600 x 300 x 300 m, homogeneous, rotated by 30 deg about z,
     centre at (2200, 300, 0) m
on a cubic (square) lattice with 10% random jitter, so that the tree nodes have non-vanishing quadrupole moments.
Lattice spacing: R/18 in 3D (25674 particles), R/40 in 2D (5902 particles); the sml in material.cfg and
material_2d.cfg is 2.1 lattice spacings.

All particles get rho = rho_0 and e = 0 (zero pressure for Tillotson), so that they hardly move during the short
run, and the gravitational accelerations in the output can be compared with a direct summation over the output
positions (check_gravity.py).

Columns (HYDRO, INTEGRATE_DENSITY, INTEGRATE_ENERGY, see parameter.h of this test case):
    3D: x y z vx vy vz m rho e material_type
    2D: x y vx vy m rho e material_type
"""

import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--dim', type=int, choices=(2, 3), default=3)
args = ap.parse_args()
dim = args.dim

R = 1000.0
dx = R / 18 if dim == 3 else R / 40
rho0 = 2680.0
jitter = 0.1
mp = rho0 * dx**dim

rng = np.random.default_rng(42)
g = np.arange(-3000.0, 3000.0 + dx, dx)
pos = np.column_stack([c.ravel() for c in np.meshgrid(*([g] * dim), indexing='ij')])

# body A
r2 = np.sum(pos**2, axis=1)
a = pos[r2 < R**2]
ma = mp * (1.0 + 4.0 * (1.0 - np.sum(a**2, axis=1) / R**2))

# body B, in its body frame
semi = np.array([600.0, 300.0, 300.0])[:dim]
q = pos[np.sum((pos / semi)**2, axis=1) < 1.0]
phi = np.radians(30.0)
rot = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])[:dim, :dim]
b = q @ rot.T + np.array([2200.0, 300.0, 0.0])[:dim]
mb = np.full(len(b), mp)

x = np.vstack((a, b))
x += jitter * dx * (2.0 * rng.random(x.shape) - 1.0)
m = np.concatenate((ma, mb))

out = sys.stdout
fmt = " ".join(["%.10e"] * dim) + " " + " ".join(["0.0"] * dim) + " %.10e %.10e 0.0 0\n"
for xi, mi in zip(x, m):
    out.write(fmt % (*xi, mi, rho0))
sys.stderr.write("DIM %d: %d particles (A: %d, B: %d), sml = %g\n" % (dim, len(m), len(a), len(b), 2.1 * dx))
