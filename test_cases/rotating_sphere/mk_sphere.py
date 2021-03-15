#!/usr/bin/env python3


# create initial particle distribution for a rigidly rotating sphere

# the values are taken from Keisuke Sugiura's phd thesis p.26
# units are
# density rho_0
# sound speed cs
# Rsphere 
# omega_z = 2\pi/100 * (cs/Rsphere)
# shear modulus = cs**2 * rho_0

import sys
import numpy as np
import random

dx = 0.2
x = np.arange(-1, 1+dx, dx)
y = np.arange(-1, 1+dx, dx)
z = np.arange(-1, 1+dx, dx)

Rmax = 1.0
cs = 1.0

r = Rmax+dx
Rin = 0.0

rho1 = 1.0
rho0 = 1.0
mass0 = dx**3 * rho0
mass1 = dx**3 * rho1


be_random = 0

omega = 2*np.pi*1e-2 * cs / Rmax

for i in x:
    for j in y:
        for k in z:
            Rsq = i**2 + j**2 + k**2
        #     x y z vx vy vz m rho e mt S00 S01 S02 S10 S11 S12 S20 S21 S22
            if Rsq < Rmax**2:
                rho = rho0
                mass = mass0
                material_type = 0
            elif  Rsq < Rin**2:
                rho = rho1
                mass = mass1
                material_type = 1
            else:
                continue

            # add random noise of 10% of dx to each position
            rnd = 0.2 * np.random.random_sample(3) - 0.1
            if be_random:
                xi = i+rnd[0]
                xj = j+rnd[1]
                xk = k+rnd[2]
            else:
                xi = i
                xj = j
                xk = k

            print("%g %g %g 0.0 %g %g %g %g 0.0 %d 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" % (xi, xj, xk, xi*omega, xi*omega, mass, rho, material_type))

