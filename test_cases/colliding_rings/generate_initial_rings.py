#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

"""
generate initial input sph particle distribution for colliding rings testcase
see Monaghan SPH without a tensile instability, journal of computational physics 2000
and James Gray's PhD thesis


Christoph, 2019-10-02
"""

speed = 0.059
rmin = 3.0
rmax = 4.0
offset = 5.0

dx = 0.075
density = 1.0


mass = dx**2 * density
outputf = open("rings.0000", 'w')

x = []
y = []
i = -offset
while i < offset:
    j = -offset
    while j < offset:
        r = np.sqrt(i**2 + j**2)
        if r >= rmin and r <= rmax:
            x.append(i-offset)
            y.append(j)
            x.append(i+offset)
            y.append(j)
            print("%e %e %e 0.0 %e %e 0 0.0 0.0 0.0 0.0" % (i-offset, j, speed, mass, density), file=outputf)
            print("%e %e %e 0.0 %e %e 0 0.0 0.0 0.0 0.0" % (i+offset, j, -speed, mass, density), file=outputf)
        j += dx
    i += dx


outputf.close()

fig, ax = plt.subplots()

ax.scatter(x, y, s=1, c='r')
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_xlabel("x-coordinate")
ax.set_ylabel("y-coordinate")
plt.grid()
ax.set_aspect('equal')
fig.savefig("rings_initial.png")
