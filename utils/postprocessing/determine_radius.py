#!/usr/bin/env python 


# simple numpy script to determine crater radius and crater depth of 2D simulation
# input file has to be an output file of map_sph_to_grid
#
# author: Christoph Schaefer, cm.schaefer@gmail.com
# last changes: 2015-08-18


import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import sys
import math


try:
    filename = sys.argv[1]
except:
    print "usage is determine_radius.py filename"
    sys.exit(0)

if len(sys.argv) > 2:
    print "usage is determine_radius.py filename"
    sys.exit(0)



# find array index that is nearest to value
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
        return idx-1
    else:
        return idx

# height where we want to find the radius of the crater
yvalue = 0.1
# crater center
xvalue = 0.0

# load x, y and rho from grid file
# format x y rho
x, y, rhotmp = np.loadtxt(filename,  usecols=(0, 1, 2), unpack=True)

levels = np.linspace(np.min(rhotmp), np.max(rhotmp), 20)

xu = np.unique(x)
yu = np.unique(y)

yu_idx = find_nearest(yu, yvalue)
xu_idx = find_nearest(xu, xvalue)

rho = np.zeros((len(xu), len(yu)))


for i in range(len(yu)):
    for j in range(len(xu)):
        rho[i,j] = rhotmp[len(xu)*i+j]
    

radius = np.zeros(0)
# now return the xvalue where rho first gets zero
for i in range(len(xu)):
    if rho[i,yu_idx] < 1e-8:
        radius = np.append(radius,xu[i])


depth = np.array([-1])
# now return the yvalue where rho first gets zero
for i in range(10,len(yu)):
    if rho[xu_idx,i] < 1e-8:
        depth = np.append(depth, yu[i])



print "Radius:",  (np.max(radius)-np.min(radius))*0.5,
print "    Depth:", (yvalue - np.min(depth[1:]))
        




