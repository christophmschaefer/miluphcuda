#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

if len(sys.argv) != 2:
    print("Usage: %s <hdf5 output file>" % sys.argv[0])
    sys.exit(1)

filename = sys.argv[1]


h5f = h5py.File(filename, 'r')
coordinates = h5f['x']
rho = h5f['rho']
pressure = h5f['p']

x = coordinates[:,0]
y = coordinates[:,1]
z = coordinates[:,2]
r = np.sqrt(x**2 + y**2 + z**2)
time = float(h5f['time'][0])

fig, ax = plt.subplots()
ax.scatter(r, rho, c='r', s=0.1, alpha=0.3)
ax.set_title("Density at time t = %.2e" % time)
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$\varrho$')
ax.set_xlim(0,0.5)
ax.set_ylim(0,4.0)

fig.savefig(filename+"_density"+".png")


fig, ax = plt.subplots()
ax.scatter(r, pressure, c='b', s=0.1, alpha=0.3)
ax.set_title("Pressure at time t = %.2e" % time)
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$p$')
ax.set_xlim(0,0.5)
ax.set_ylim(0,20)

fig.savefig(filename+"_pressure"+".png")


h5f.close()







