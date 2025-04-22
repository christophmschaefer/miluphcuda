#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script calculates the total angular momentum from a sph particles distribution in a miluphcuda hdf5 output file and
includes additionally the angular momentum from existing point masses

Note: Only one input file has to be specificed. 
"""

import numpy as np
import h5py
import sys

if len(sys.argv) != 2:
    print("Usage: ./angular_momentum_particles_and_point_masses.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = input_file.replace('.h5', '_total_angular_momentum.txt')

try:
    h5data = h5py.File(input_file, 'r')
except:
    print("Error: Could not open file %s" % input_file)
    sys.exit(1)


# read time from file
time = h5data['time'][0]
print("Time in input file %g" % time)

# read particle data
x_p = np.array(h5data['x'])
v_p = np.array(h5data['v'])  
m_p = np.array(h5data['m'])  
material_type_p = np.array(h5data['material_type'])  # Read the material type of the particles

# try to open point mass file if available
try:
    input_file_pm = input_file.replace('.h5', '.mass.h5')
    h5data_pm = h5py.File(input_file_pm, 'r')
    x_pm = np.array(h5data_pm['x'])
    v_pm = np.array(h5data_pm['v'])  # Read the point mass velocities
    m_pm = np.array(h5data_pm['m'])  # Read the point mass masses
except:
    x_pm = np.zeros((0, 3))    
    v_pm = np.zeros((0, 3))    
    m_pm = np.zeros(0)


if len(x_pm) > 0:
    print("Found particles file with %d particles and point masses file with %d point masses" % (x_p.shape[0], x_pm.shape[0]))
else:   
    print("Found particles file with %d particles" % (x_p.shape[0]))


# filter for active particles that were not accreted or disabled or removed somehow
x_p = x_p[material_type_p > -1]
v_p = v_p[material_type_p > -1]
m_p = m_p[material_type_p > -1]

print("Found %d active particles" % (x_p.shape[0]))
# calculate momentum
momentum_p = np.expand_dims(m_p, axis=1)*v_p
momentum_pm = np.expand_dims(m_pm, axis=1)*v_pm

# calculate angular momentum
angular_momentum_p = np.cross(x_p, momentum_p)
angular_momentum_pm = np.cross(x_pm, momentum_pm)

print("Angular momentum at time %g from particles in (x,y,z): %.17e %.17e %.17e" % (time, np.sum(angular_momentum_p[:,0]), np.sum(angular_momentum_p[:,1]), np.sum(angular_momentum_p[:,2])))
print("Angular momentum at time %g from point masses in (x,y,z): %.17e %.17e %.17e" % (time, np.sum(angular_momentum_pm[:,0]), np.sum(angular_momentum_pm[:,1]), np.sum(angular_momentum_pm[:,2])))


L_tot_x = np.sum(angular_momentum_p[:,0]) + np.sum(angular_momentum_pm[:,0])
L_tot_y = np.sum(angular_momentum_p[:,1]) + np.sum(angular_momentum_pm[:,1])
L_tot_z = np.sum(angular_momentum_p[:,2]) + np.sum(angular_momentum_pm[:,2])

print("Total angular momentum at time %g in (x,y,z): %.17e %.17e %.17e" % (time, L_tot_x, L_tot_y, L_tot_z))
# write to file
np.savetxt(output_file, np.array([[time, L_tot_x, L_tot_y, L_tot_z]]), header="time Lx Ly Lz", fmt="%.17g", delimiter="\t")
