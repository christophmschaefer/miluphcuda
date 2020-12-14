#!/usr/bin/env python 


# simple matplotlib script to draw contour lines of density
# of 2D grid data
# input file has to be an output file of map_sph_to_grid
#
# author: Christoph Schaefer, cm.schaefer@gmail.com
# last changes: 2015-08-17


import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import sys


try:
    filename = sys.argv[1]
except:
    print "usage is density_contour.py filename"
    sys.exit(0)

if len(sys.argv) > 2:
    print "usage is density_contour.py filename"
    sys.exit(0)



# load x, y and rho from grid file
# format x y rho
x, y, rhotmp = np.loadtxt(filename,  usecols=(0, 1, 2), unpack=True)

levels = np.linspace(np.min(rhotmp), np.max(rhotmp), 20)

# create matplotlib conforming format: vector x, vector y, 2D-array rho with size len(x), len(y)
xu = np.unique(x)
yu = np.unique(y)

rho = np.zeros((len(xu), len(yu)))


fig = plt.figure()

plt.xlabel('x [m]')
plt.ylabel('y [m]')

plt.title('Contour lines of density')

for i in range(len(xu)):
    for j in range(len(yu)):
            rho[j,i] = rhotmp[len(xu)*i+j]
                


plt.contourf(xu,yu,rho,levels=levels)
#plt.contour(xu,yu,rho,levels=levels)
plt.colorbar()
plt.savefig("density_contour.pdf", bbox_inches='tight')               
