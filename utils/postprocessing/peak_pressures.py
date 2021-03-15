#!/usr/bin/env python3

# Extracts the peak pressures for all particles in a range of miluphcuda HDF5 output files and adds them to one or some of these files.

# Christoph Burger 11/Dec/2017


import argparse
import h5py
import numpy as np
import shutil
import sys


parser = argparse.ArgumentParser(description = "Extracts the peak pressures for all particles in a range of miluphcuda HDF5 output files and adds them to one or some of these files.")
parser.add_argument("-i", help = "list of HDF5 input files to process", nargs = '+', metavar = "files", type = str, dest = "infiles")
parser.add_argument("-o", help = "list of HDF5 files the peak pressures should be attached to", nargs = '+', metavar = "files", type = str, dest = "outfiles")
parser.add_argument("-v", help = "be verbose", action = 'store_true')
args = parser.parse_args()

if len(sys.argv) == 1:
    print("add -h to get an usage.")
    sys.exit(1)

# extract particle number 'N_tot' from first infile:
if args.v:
    print("Extract total particle number from '{0}' ...".format(args.infiles[0]))

try:
    ifl = h5py.File(args.infiles[0], 'r')
except IOError:
    print("ERROR! Cannot open '{0}' for reading.".format(args.infiles[0]))
    sys.exit(1)

mattypes_id = ifl['material_type']  # actually it doesn't matter which quantity is read here
N_tot = len(mattypes_id[:])

ifl.close()

if args.v:
    print("Done. Found {0} SPH particles.".format(N_tot))



# initialize array for peak pressures:
peak_pressures = np.zeros(N_tot)



# iterate over all infiles to obtain peak pressures:
if args.v:
    print("Search all input files for peak pressures ...")

for infile in args.infiles:
    if args.v:
        print("  Process '{0}' ...".format(infile))

    try:
        ifl = h5py.File(infile, 'r')
    except IOError:
        print("ERROR! Cannot open '{0}' for reading.".format(infile))
        sys.exit(1)

    pressure_id = ifl['p']
    pressures = pressure_id[:]
    for i in range(0,N_tot):
        if pressures[i] > peak_pressures[i]:
            peak_pressures[i] = pressures[i]

    ifl.close()



# add array of peak pressures to all outfiles:
if args.v:
    print("Now add array of peak pressures to all outfiles ...")

for outfile in args.outfiles:
    if args.v:
        print("  Process '{0}' ...".format(outfile))

    try:
        ofl = h5py.File(outfile, 'r+')
    except IOError:
        print("ERROR! Cannot open '{0}' for writing.".format(outfile))
        sys.exit(1)

    ofl.create_dataset('peak_pressure', data = peak_pressures)

    ofl.close()
