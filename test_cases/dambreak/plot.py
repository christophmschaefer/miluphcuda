#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys


if len(sys.argv) != 2:
    print("usage: %s dam.XXXX where XXXX denotes the output file number of the simulation" % sys.argv[0])
    sys.exit(1)



x, y, rho = np.loadtxt(sys.argv[1], usecols=(0,1,5), unpack=True)

fig, ax = plt.subplots()

ax.scatter(x, y, c=rho, s=1.0)
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 0.3)
#ax.set_aspect('equal')

fig.savefig(sys.argv[1]+".png")
