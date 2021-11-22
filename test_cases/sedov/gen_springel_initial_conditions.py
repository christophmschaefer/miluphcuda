#!/usr/bin/env python

"""
generate initial conditions for 3D Sedov-Taylor blast wave simulation
according to Springel & Hernquist 2002 (MNRAS), see sect. 4 "Point-like energy injection"
"""

import numpy as np

# settings: Energy disposal of E=1 at single particle in the origin (0,0,0). cubic lattice with
# unit length and unit density

min = -0.5
max = 0.5
explosion_energy = 1.0
efloor = 1e-6

N = 64
cond = N//2

x, dx = np.linspace(min, max, N, retstep=True)
y = x
z = x
material_type = 0

print("set smoothing length to %.17lf" % (2.51 * dx))
rho = 1.0
m = rho * dx**3
ethermal = explosion_energy/m
output = open("springel_sedov.0000", 'w')
for i, xi in enumerate(x):
    for j, yi in enumerate(y):
        for k, zi in enumerate(z):
            if i == cond and j == cond and k == cond:
                e = ethermal
            else:
                e = efloor
            print("%.17lf %.17lf %.17lf 0.0 0.0 0.0 %.17lf %.17lf %d" % (xi, yi, zi, m, e, material_type), file=output)

output.close()





