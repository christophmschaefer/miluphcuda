#!/usr/bin/env python3
"""
Sod shock tube (Sod 1978) for test_cases/inviscid_sph: ideal gas, gamma = 1.4,
    rho = 1,     p = 1,   v = 0   for x < 0.5,
    rho = 0.125, p = 0.1, v = 0   for x > 0.5,
in 1D on [-1, 2] with particles of equal mass (spacing 5e-4 left, 4e-3 right, at the centres of
cells of these widths), as in test_cases/shocktube.
Writes sod.0000 (ascii, columns for DIM 1, INTEGRATE_DENSITY 0, INTEGRATE_ENERGY 1: x v m e material_type).
"""
gamma = 1.4
rho_l, p_l = 1.0, 1.0
rho_r, p_r = 0.125, 0.1
x0, xmin, xmax = 0.5, -1.0, 2.0
dx = 5e-4
m = rho_l * dx

dx_r = dx * rho_l / rho_r          # equal masses
cells = [(xmin, x0, dx, rho_l, p_l), (x0, xmax, dx_r, rho_r, p_r)]

n = 0
with open("sod.0000", "w") as f:
    for a, b, d, rho, p in cells:          # particles at the centres of cells of width d
        for k in range(int(round((b - a) / d))):
            x = a + (k + 0.5) * d
            print(f"{x:.10e} {0.0:e} {m:.10e} {p / ((gamma - 1.0) * rho):.10e} 0", file=f)
            n += 1
print(f"wrote sod.0000 with {n} particles")
