#!/usr/bin/env python3

import numpy as np
import h5py 
import matplotlib.pyplot as plt
import sys


"""
small python3 script to plot alpha_epspor(epsilon_v) from miluph hdf5 output file 
and the theoretical crush curve

all particles should lie on or below the crush curve


the input file should be a hdf5 file with the following data sets:

/epsilon_v
/alpha_epspor
/time

Oliver Wandel, Nov 2018
comments to: oliver.wandel@uni-tuebingen.de
"""


# you'll find these values in material.cfg of your simulation as 
# porepsilon_kappa, porepsilon_alpha_0, porepsilon_epsilon_e, porepsilon_epsilon_x, porepsilon_epsilon_c - while
# epsilon_c needs to be calculated by hand at the time being

kappa = 0.9
alpha_0 = 2.0
epsilon_e = 0.0
epsilon_x = -0.4
epsilon_c = 2.0 * (1.0 - alpha_0 * np.exp(kappa * (epsilon_x - epsilon_e))) / (kappa * alpha_0 * np.exp(kappa * (epsilon_x - epsilon_e))) + epsilon_x

print("Caluculated epsilon_c: %e\n" % epsilon_c)



if len(sys.argv) != 2:
    print("Usage: %s <sph_outputfile.h5> " % sys.argv[0])
    print("Please make sure, that you use the correct values for the theoretical crush curve in the python script")
    sys.exit(0)


try:
    inputf = h5py.File(sys.argv[1], 'r')
except: 
    print("Cannot open file %s " % sys.argv[1])
    sys.exit(1)





time = inputf['time'][0]
epsilon_v = inputf['epsilon_v'][:]
alpha_epspor = inputf['alpha_epspor'][:]
nop = len(epsilon_v)
x = np.linspace(epsilon_c, np.max(epsilon_v), nop)
y1 = alpha_0 * np.exp(kappa * (x - epsilon_e))
y2 = 1.0 + (alpha_0 * np.exp(kappa * (epsilon_x - epsilon_e)) - 1.0) * ((epsilon_c - x) / (epsilon_c - epsilon_x))**2.0


fig, ax = plt.subplots()


ax.plot(x, y1, '--', c='r', label='crush curve')
ax.plot(x, y2, '--', c='b', label='crush curve')
ax.scatter(epsilon_v, alpha_epspor, c='y', s=2, label='distention from simulation')
ax.set_xlim(epsilon_c - 0.5, 0.0)
ax.set_ylim(0.9, alpha_0*1.1)
ax.grid(True, color='gray', linestyle=':')
ax.legend(loc='upper right')

ax.set_xlabel(r'Volumetric Strain [1]')
ax.set_ylabel(r'Distention [1]')



fig.savefig(sys.argv[1]+".png", dpi=100)

inputf.close()

