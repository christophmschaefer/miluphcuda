#!/usr/bin/env python3

'''
Python3 script to trace a SPH particle's properties over some time in a miluphCUDA run via the HDF5 output files.

Christoph Burger 18/Feb/2019
'''


try:
    import matplotlib.pyplot as plt
    import sys
    import argparse
    import h5py
except:
    print("ERROR when trying to load necessary modules ...")
    sys.exit(1)


plt.style.use('ggplot')


parser = argparse.ArgumentParser(description="Script to trace a SPH particle's properties over some time in a miluphCUDA run via the HDF5 output files.")
parser.add_argument("--path", help = "path of the miluphCUDA directory to process", type = str)
parser.add_argument("--h5_files", help = "list of all HDF5 files to consider (enter something like '*.h5' for all of them)", nargs='+')
parser.add_argument("--index", help = "index of the particle to follow (in the HDF5 files)", type = int)
parser.add_argument("--t_min", help = "lower end of time range to consider, default is lowest time in list of HDF5 files", type = float, default = -1.0e30)
parser.add_argument("--t_max", help = "upper end of time range to consider, default is largest time in list of HDF5 files", type = float, default = 1.0e30)
parser.add_argument("--plot_all", help = "set to plot 4 subplots for rho, e, p, number-of-interactions and save them to 'trace_particle.png' (in path dir), not set by default", action = 'store_true')
parser.add_argument("--datafile", help = "set to write data for rho, e, p, number-of-interactions to output file 'trace_particle.data' (in path dir), not set by default", action = 'store_true')
parser.add_argument("-v", help = "be verbose", action = 'store_true')
args = parser.parse_args()


# initialize empty lists for quantities to read
times = []
rhos = []
es = []
ps = []
nois = []   # number of interactions


# loop over HDF5 files
for h5file in args.h5_files:
    # open HDF5 file
    try:
        ifl = h5py.File(h5file, 'r')
    except IOError:
        print("ERROR! Cannot open '{0}' for reading.".format(h5file) )
        sys.exit(1)
    if args.v:
        print("Opened file '{0}' ...".format(h5file) )
    
    # read time
    current_time = float( ifl['time'][0] )
    
    # read data on particle from file if in desired time range
    if current_time >= args.t_min  and  current_time <= args.t_max:
        times.append(current_time)
        rhos.append( ifl['rho'][args.index] )
        es.append( ifl['e'][args.index] )
        ps.append( ifl['p'][args.index] )
        nois.append( ifl['number_of_interactions'][args.index] )
        if args.v:
            print("    and read quantities at time = {0}.".format(current_time) )
    else:
        if args.v:
            print("    ... and ignoring it because time = {0} is out of desired range".format(current_time) )
    
    ifl.close()


# write data to file if desired
if args.datafile:
    # open datafile
    try:
        dfl = open(args.path+"/trace_particle.data", 'w')
    except IOError:
        print("ERROR! Cannot open '{0}' for writing.".format(args.path+"/trace_particle.data") )
        sys.exit(1)
    
    # write to datafile
    dfl.write("#  1.rho  2.e  3.p  4.number-of-interactions\n")
    for i in range(0,len(rhos)):
        dfl.write("{0:15g}\t{1:15g}\t{2:15g}\t{3:10d}\n".format(rhos[i], es[i], ps[i], nois[i]) )
    
    dfl.close()


# plot data if desired
if args.plot_all:
    fig, axarr = plt.subplots(4,1)
    
#    axarr[0].set_xlim(args.t_min, args.t_max)
#    axarr[0].set_ylim(args.a_min, args.a_max)
    axarr[0].set_xlabel("Time")
    axarr[0].set_ylabel("rho")
    axarr[0].plot(times, rhos)
    
    axarr[1].set_xlabel("Time")
    axarr[1].set_ylabel("e")
    axarr[1].plot(times, es)
    
    axarr[2].set_xlabel("Time")
    axarr[2].set_ylabel("p")
    axarr[2].plot(times, ps)
    
    axarr[3].set_xlabel("Time")
    axarr[3].set_ylabel("number-of-interactions")
    axarr[3].plot(times, nois)
    
    fig.savefig(args.path + "/trace_particle.png")
    plt.close()

