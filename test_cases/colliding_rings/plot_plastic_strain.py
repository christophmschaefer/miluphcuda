#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py


if len(sys.argv) != 2:
    print("usage is ./plot_plastic_strain.py hdf5_outputfile")
    sys.exit(1)

filename = sys.argv[1]
my_dpi = 100

h5f = h5py.File(filename)
coordinates = h5f['x']
ep = h5f['total_plastic_strain']
total_strain = ep[:]
x = coordinates[:,0]
y = coordinates[:,1]
h5f.close()

plt.figure(figsize=(720/my_dpi, 720/my_dpi), dpi=my_dpi)

plt.scatter(x, y, c=total_strain, s=1, edgecolor='')
plt.xlim(-10,10)
plt.ylim(-10,10)
# plt.clim(0.8, 1.2)

plt.xlabel("x")
plt.ylabel("y")

cb = plt.colorbar()
cb.set_label(r'$\varepsilon_{\mathrm{plastic}}$')

fileout = filename+'.png'
plt.savefig(fileout, bbox_inches='tight', dpi=my_dpi)
