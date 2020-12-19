#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
small python script to determine the fragments of a SPH simulation outcome particle
distribution.

author: Christoph Schaefer, ch.schaefer@uni-tuebingen.de
date: April 2018
"""

import sys
import argparse
import pdb


try:
    import numpy as np
    import scipy.spatial
except:
    print("Cannot load numpy and/or scipy. Exiting.")
    sys.exit(1)
    
try:
    import h5py
except:
    print("Cannot load h5py module. Exiting.")
    sys.exit(1)


def return_unsorted_particle_id():
    for i in range(len(fragment)):
        if fragment[i] < 0:
            return i
    return -1

def return_id_of_tobesorted_particle():
    remain = fragment[fragment<0]
    print("\r Particles remaining %010d" % len(remain), end="")
    for i in range(len(fragment)):
        if fragment[i] == 0:
            return i
    return -1



""" 
stores indexlist for each particle which contains its interaction
partner
"""
def find_neighbours(radius):
    interaction_list = []
    for i in range(len(x)):
        tmp_interactions = tree.query_ball_point(x[i], radius)
        interaction_list.append(tmp_interactions)
        if not i%1000:
            print("\rGetting interactions for particle %d   " % i , end="")
            print("it got %d interactions" % len(tmp_interactions), end="")
    return interaction_list


"""
Description:
 we use three kinds of fragment ids
 < 0: particle has never been touched before and waits
 = 0: particle has been touched but not yet sorted
 > 0: particle is sorted 
"""
def find_fragments(radius):
    fragno = 1
    # first determine all particles with no interaction partners and
    # remove them from the scheme
    for i in range(len(fragment)):
        #indexlist = tree.query_ball_point(x[i], radius)
        indexlist = interaction_list[i]
        if len(indexlist) < 2:
            fragment[i] = fragno
            fragno = fragno+1
    print("\nFound lonely particles #", fragno-1)
    # particle which is not yet sorted:
    while (return_unsorted_particle_id() >= 0):
        #print("\rLooking for fragment no %d." % fragno, end="")
        print("\rLooking for fragment no %d.                    " % fragno)
        u_id = return_unsorted_particle_id()
        fragment[u_id] = 0
        i = 0
        while (return_id_of_tobesorted_particle() >= 0):
            u_id2 = return_id_of_tobesorted_particle()
            i = i + 1
            fragment[u_id2] = fragno
            #indexlist = tree.query_ball_point(x[u_id2], radius)
            indexlist = interaction_list[u_id2]
            indexlist.remove(u_id2)
            for j in indexlist:
                if fragment[j] < 0:
                    fragment[j] = 0
        fragno = fragno+1
    return fragno-1



"""
Description:
 we use three kinds of fragment ids
 < 0: particle has never been touched before and waits
 = 0: particle has been touched but not yet sorted
 > 0: particle is sorted 
"""
def find_fragments2(radius):
    fragno = 1
    # first determine all particles with no interaction partners and
    # remove them from the scheme
    for i in range(len(fragment)):
        #indexlist = tree.query_ball_point(x[i], radius)
        indexlist = interaction_list[i]
        if len(indexlist) < 2:
            fragment[i] = fragno
            fragno = fragno+1
    print("\nFound lonely particles #", fragno-1)
    # particle which is not yet sorted:

    # now work with one big list for each fragment
    while (return_unsorted_particle_id() >= 0):
        #print("\rLooking for fragment no %d." % fragno, end="")
        print("\rLooking for fragment no %d.                    " % fragno)
        u_id = return_unsorted_particle_id()
        fragment[u_id] = fragno
        i = 0
        current_size = 0
        last_size = -1
        # list of all interaction partners of u_id
        indexlist = interaction_list[u_id]
        while current_size != last_size:
            # go through all particles in indexlist 
            # if their fragmentid is < 0 add their neighbours to indexlist
            # and set their ids to fragno
            last_size = len(indexlist)
            for neighbour in indexlist:
                if fragment[neighbour] < 0:
                    new_neighbours = interaction_list[neighbour]
                    new_neighbours.remove(neighbour)
                    fragment[neighbour] = fragno
                    indexlist.extend(new_neighbours)
            current_size = len(indexlist)

        fragno = fragno+1
    return fragno-1


parser = argparse.ArgumentParser(description='Tool to identify fragments in a SPH particle distribution. All particles with damage = 1 will be neglected and fragments will be search according to the fragment interaction length. The new /fragment dataset will be appended to the input file. The input file format has to be hdf5 with the following datasets: locations /x, velocities /v, masses /m, damage /damage and materialID /material_type. Statistics about the identified fragments will be written to the textfile INPUT_FILE.frags.') 

parser.add_argument('--input_file', help='input file name', default='none')
parser.add_argument('--scalelength', help='maximum length between two particles that belong to the same fragment', default=-1, type=float)
parser.add_argument('--material', help='only consider particles with material ID MATERIAL', default='-1')


args = parser.parse_args()

if len(sys.argv) < 2:
    print ("Try --help to get a usage.")
    sys.exit(0)



# open h5 file and read it

try:
    h5file = h5py.File(args.input_file, "r")
except:
    print("Error. Cannot open file %s." % args.input_file)
    sys.exit(1)


try:
    xi = h5file['x'][...]
    vi = h5file['v'][...]
    mi = h5file['m'][...]
    try:
        di = h5file['damage'][...]
    except:
        print("Cannot find damage information in h5 file. Neglecting damage.")
        di = np.zeros(len(mi))
    matIDi = h5file['material_type'][...]
except:
    print("Error. Cannot read datasets from file.")
    sys.exit(1)


matID = matIDi[di<1]
x = xi[di<1]
v = vi[di<1]
m = mi[di<1]
d = di[di<1]
fragmenti = np.zeros(len(mi))
fragment = fragmenti[di<1]
print("%d particles found in the file, %d undamaged particles." % (len(mi), len(fragment[d<1])))

# set all fragment ids to -1
fragment += -1
fragmenti += -1

# build cKDTree
tree = scipy.spatial.cKDTree(x)

radius = float(args.scalelength)
wantedID = int(args.material)

if radius <= 0:
    print("Error. Scale length is zero or negative.")
    print("If you do not know which value to use, take the smoothing length of the simulation.")
    sys.exit(1)

if wantedID >= 0:
    print("Only considering particles with id %d." % wantedID)


print("Performing fragment search with length %g." % radius)


# search all neighbours for each particle within radius
interaction_list = find_neighbours(radius)

# number of fragments
nof = 0
nof = find_fragments2(radius)

# fill back the fragments to fragmenti
j = 0
for i in range(len(fragmenti)):
    if di[i] < 1:
        fragmenti[i] = fragment[j]
        j = j+1


print("\nNow writing to file %s." % args.input_file)
h5file.close()
h5file = h5py.File(args.input_file, 'a')
# first delete data set if it's in there
try:
    h5file.__delitem__('fragments')
except:
    pass

h5file.create_dataset('fragments', data=fragmenti)
h5file.close()

# determine center of mass (including damaged particles)
dim = np.shape(x)[1]

cgx = np.sum(x[:,0]*m)/np.sum(m)
cgy = np.sum(x[:,1]*m)/np.sum(m)
if dim > 2:
    cgz = np.sum(x[:,2]*m)/np.sum(m)
    cg = np.array([cgx, cgy, cgz])
else:
    cg = np.array([cgx, cgy])
total_m = np.sum(m)
print("Center of mass is at: \t", end="")
print(cg)
print("Total mass: %g." % total_m)
print("Total number of fragments: %d." % nof)
outputfile = args.input_file + '.frags'
print("Writing statistics of fragments to file %s" % outputfile)

# let's do some statistics with the fragments
fout = open(outputfile, 'w')
fout.write('# fragment statistics\n')
fout.write('# fragmentNo, number of particles, mass of fragment, center of mass of fragment, momentum of fragment\n')


# fragmentno runs from 1 to nof+1
for f in range(1, nof+1):
    indices = np.where(fragment == f)
    # mass of fragment
    mf = np.sum(m[indices])
    nop = np.shape(indices)[1]
    if nop < 1:
        continue
    # center of mass of fragment
    cgfx = np.sum(x[indices,0]*m[indices])/mf
    cgfy = np.sum(x[indices,1]*m[indices])/mf
    if dim > 2:
        cgfz = np.sum(x[indices,2]*m[indices])/mf
        cgf = np.array([cgfx, cgfy, cgfz])
    else:
        cgf = np.array([cgfx, cgfy])
    # impuls of fragment
    pfx = np.sum(m[indices]*v[indices,0])
    pfy = np.sum(m[indices]*v[indices,1])
    # write to file
    if dim > 2:
        pfz = np.sum(m[indices]*v[indices,2])
        out = [f, nop, mf, cgfx, cgfy, cgfz, pfx, pfy, pfz]
    else:
        out = [f, nop, mf, cgfx, cgfy, pfx, pfy]
    outstring = ' '.join(map(str, out)) + '\n'
    fout.write(outstring)





