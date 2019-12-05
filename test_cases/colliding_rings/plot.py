#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys


if len(sys.argv) is not 2:
    print("usage is ./plot.py outputfile")
    sys.exit(1)

filename = sys.argv[1]
my_dpi = 100

x, y, vx, vy, rho = np.loadtxt(filename,  usecols=(0, 1, 2, 3, 5), unpack=True)


plt.figure(figsize=(720/my_dpi, 720/my_dpi), dpi=my_dpi)

plt.scatter(x,y, c=rho, s=3, edgecolor='')
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.clim(0.8, 1.2)

plt.xlabel("x")
plt.ylabel("y")

cb = plt.colorbar()
cb.set_label(r'$\varrho$')

fileout = filename+'.png'
plt.savefig(fileout, bbox_inches='tight', dpi=my_dpi)
