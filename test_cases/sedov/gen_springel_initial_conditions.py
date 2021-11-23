#!/usr/bin/env python

"""
generate initial conditions for 3D Sedov-Taylor blast wave simulation
according to Springel & Hernquist 2002 (MNRAS), see sect. 4 "Point-like energy injection"
"""

import numpy as np


def cubicSpline(dx_vec, sml):
    r = 0
    for d in range(3):
        r += dx_vec[d] * dx_vec[d]

    r = np.sqrt(r)
    W = 0
    q = r/sml

    f = 8./np.pi * 1./(sml * sml * sml);

    if q > 1:
        W = 0
    elif q > 0.5:
        W = 2. * f * (1.-q) * (1.-q) * (1-q)
    elif q <= 0.5:
        W = f * (6. * q * q * q - 6. * q * q + 1.)

    return W


# settings: Energy disposal of E=1 at single particle in the origin (0,0,0). cubic lattice with
# unit length and unit density

min = -0.5
max = 0.5
explosion_energy = 1.0
efloor = 1e-6

N = 65
cond = N//2

x, dx = np.linspace(min, max, N, retstep=True)
y = x
z = x
material_type = 0
sml = 2.51*dx
r_smooth = 2*sml
print("set smoothing length to %.17lf" % (2.51 * dx))
rho = 1.0
m = rho * dx**3
ethermal = explosion_energy/m
output = open("springel_sedov.0000", 'w')
verify = 0.0
# print(r_smooth)
for i, xi in enumerate(x):
    for j, yi in enumerate(y):
        for k, zi in enumerate(z):
            ri = np.sqrt(xi**2 + yi**2 + zi**2)
            W = cubicSpline(np.array([xi, yi, zi]), r_smooth)
            e = W
            verify += W * m
            if e < 1e-6:
                e = 1e-6
            print("%.17lf %.17lf %.17lf 0.0 0.0 0.0 %.17lf %.17lf %d" % (xi, yi, zi, m, e, material_type), file=output)

print(verify)
output.close()





