#!/usr/bin/env python3

import numpy as np
import h5py 
import matplotlib.pyplot as plt
import sys


"""
small python3 script to plot p(alpha) from miluph hdf5 output file 
and the theoretical crush curve

all particles should lie on or below the crush curve


the input file should be a hdf5 file with the following data sets:

/p
/alpha_jutzi
/time

Christoph Schaefer, May 2017
comments to: ch.schaefer@uni-tuebingen.de
"""


# you'll find these values in material.cfg of your simulation as 
# porjutzi_p_elastic porjutzi_p_transition porjutzi_p_compacted porjutzi_alpha_0 porjutzi_alpha_e porjutzi_alpha_t porjutzi_n1 porjutzi_n2

alpha_0 = 2.0
alpha_e = 4.64
alpha_t = 1.9
p_e = 1e6
p_t = 6.8e7
p_c = 2.13e8
n1 = 12.0
n2 = 3.0





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
p = inputf['p'][:]
alpha = inputf['alpha_jutzi'][:]
nop = len(p)
x = np.linspace(np.min(p), np.max(p), nop)
y1 = (alpha_0-1)/(alpha_e-1) * (alpha_e-alpha_t) *  (p_t - x)**n1 / (p_t - p_e)**n1 + (alpha_0-1)/(alpha_e-1) * (alpha_t - 1) * (p_c - x)**n2 / (p_c - p_e)**n2 + 1
y2 = (alpha_0-1)/(alpha_e-1) * (alpha_t - 1) * (p_c - x)**n2 / (p_c - p_e)**n2 + 1


fig, ax = plt.subplots()


ax.plot(x, y1, '--', c='r', label='crush curve')
ax.plot(x, y2, '--', c='b', label='crush curve')
ax.scatter(p, alpha, c='y', s=2, label='distention from simulation')
ax.set_xlim(0, np.max(p))
ax.set_ylim(0, alpha_0*1.1)
ax.grid(True, color='gray', linestyle=':')
ax.legend(loc='upper right')

ax.set_xlabel(r'Pressure [Pa]')
ax.set_ylabel(r'Distention [1]')



fig.savefig(sys.argv[1]+".png", dpi=100)

inputf.close()

