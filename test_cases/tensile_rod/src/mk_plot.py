#!/usr/bin/env python3


import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import h5py
except:
    print("You need numpy, h5py and matplotlib to use this script.")
    sys.exit(1)


if len(sys.argv) != 2:
    print("Usage: %s <filename.h5>" % sys.argv[0])
    sys.exit(1)


try:
    f = h5py.File(sys.argv[1], 'r')
except:
    print("Cannot open %s." % sys.argv[1])
    sys.exit(1)


x = f['x'][:,0]
y = f['x'][:,1]
strain = f['local_strain'][:]
d = f['DIM_root_of_damage'][:]
f['DIM_root_of_damage'][:] =  0.0
d = d**2
time = float(f['time'][0])

fig, ax = plt.subplots(2, sharex=True)
ax[0].set_xlim(-1.5, 1.5)
ax[1].set_xlabel('x')
ax[0].set_ylabel('y')
ax[1].set_ylabel('y')
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[0].set_ylim(-0.5, 0.5)
ax[1].set_ylim(-0.5, 0.5)
s0 = ax[0].scatter(x, y, c=strain, s=1, cmap='Oranges')
s1 = ax[1].scatter(x, y, c=d, s=1, vmin=0, vmax=1)
#fig.colorbar(s0, ax=ax[0], orientation='horizontal', label=r'strain at time %s $t_0$' % time)
fig.colorbar(s1, ax=ax[1], orientation='horizontal', label=r'damage at time %.2f $t_0$' % time)
#boundaries=np.linspace(0,1.0,10000))

fig.tight_layout()
fig.savefig(sys.argv[1]+".png", dpi=300)
