#!/usr/bin/env python3

"""
Plots alpha(p) from miluphcuda HDF5 output files + the theoretical crush curve
(parameters hardcoded below!), optionally for several files to one plot.

All particles should lie on or below the crush curve.

The input has to be HDF5 file(s) with the following data sets:

/p
/alpha_jutzi
/dalphadt
/time

authors: Christoph Schaefer, Christoph Burger
comments to: ch.schaefer@uni-tuebingen.de

last updated: 23/May/2021
"""


# set p-alpha params (from material.cfg)
crushcurve_style = 0    # 0 for quadratic, 1 for real/steep crush curve
alpha_0 = 1.25          # porjutzi_alpha_0
alpha_e = 1.25          # porjutzi_p_alpha_e
#alpha_t = 1.9           # porjutzi_p_alpha_t
p_e = 2e8               # porjutzi_p_elastic
#p_t = 6.8e7             # porjutzi_p_transition
p_c = 2e9               # porjutzi_p_compacted
#n1 = 12.0               # porjutzi_n1
#n2 = 3.0                # porjutzi_n2


try:
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    import sys
    import traceback
    import argparse
except Exception as e:
    print("ERROR! Cannot properly import Python modules. Exiting...")
    print(str(type(e).__name__))
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)


parser = argparse.ArgumentParser(description="Plots alpha(p) from miluphcuda HDF5 output files + the theoretical crush curve (parameters hardcoded in the script!).")
parser.add_argument("-v", help = "be verbose", action = 'store_true')
parser.add_argument("--files", help = "specify one or more (.h5) files to process", nargs='+', default = None)
parser.add_argument("--imagefile", help = "filename to write image to (default: p_vs_alpha.png)", default = "p_vs_alpha.png")
parser.add_argument("--mat_type", help = "select material type for plotting (default: all particles are plotted)", type=int, default = None)
args = parser.parse_args()

if args.files == None:
    print("You have to specify at least one file with --files. Exiting...")
    sys.exit(1)

n_files = len(args.files)
if args.v:
    print("\nReading {} file(s)...\n".format(n_files) )


# create fig
fig = plt.figure()
fig.set_size_inches(6.4, 4.8*n_files)
fig.subplots_adjust(left=0.11, right=0.96, hspace=0.26, bottom=0.1, top=0.95)


# loop over input files
n = 0
for currentfile in args.files:
    n += 1

    # load data from file
    try:
        f = h5py.File(currentfile, 'r')
    except: 
        print("Cannot open %s." % currentfile)
        sys.exit(1)
    if args.v:
        print("Processing {}...".format(currentfile) )

    time = f['time'][0]

    # extract only particles with fitting mat-type
    if args.mat_type is not None:
        mat_types = f['material_type'][:]
        pTmp = f['p'][:]
        alphaTmp = f['alpha_jutzi'][:]
        dalphadtTmp = f['dalphadt'][:]

        p = []
        alpha = []
        dalphadt = []
        for i in range(0, len(mat_types)):
            if mat_types[i] == args.mat_type:
                p.append(pTmp[i])
                alpha.append(alphaTmp[i])
                dalphadt.append(dalphadtTmp[i])
        p = np.asarray(p, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        dalphadt = np.asarray(dalphadt, dtype=float)

        if args.v:
            print("    found {} particles with mat-type {}".format(len(p), args.mat_type) )

    # extract all particles (all mat-types)
    else:
        p = f['p'][:]
        alpha = f['alpha_jutzi'][:]
        dalphadt = f['dalphadt'][:]

        if args.v:
            print("    found {} particles".format(len(p)) )

    assert len(p) == len(alpha) == len(dalphadt), "ERROR. Strange mismatch in array lengths..."

    f.close()


    # set plot limits
    p_min = np.amin(p)
    if args.v:
        print("    found min p: {:g}".format(p_min) )
    if p_min > 0.:
        p_min = 0.
    if p_min < -p_e:
        p_min = -p_e
    p_max = np.amax(p)
    if args.v:
        print("    found max p: {:g}".format(p_max) )
    if p_max < p_c:
        p_max = p_c
    if p_max > 1.5*p_c:
        p_max = 1.5*p_c
    alpha_min = np.amin(alpha)
    if args.v:
        print("    found min alpha: {:g}".format(alpha_min) )
    if alpha_min > 0.99:
        alpha_min = 0.99
    alpha_max = np.amax(alpha)
    if args.v:
        print("    found max alpha: {:g}".format(alpha_max) )
    if alpha_max < 1.01*alpha_0:
        alpha_max = 1.01*alpha_0

    # compute crush curve
    x = np.linspace(p_min, p_max, 1000)
    if crushcurve_style == 0:
        # quadratic crush curve
        y = np.ones(1000)*alpha_0
        for i in range(0, len(x)):
            if x[i] > p_e:
                y[i] = 1. + (alpha_e-1.) * (p_c-x[i])**2 / (p_c-p_e)**2
            if x[i] > p_c:
                y[i] = 1.
    elif crushcurve_style == 1:
        # real/steep crush curve
        y1 = (alpha_0-1)/(alpha_e-1) * (alpha_e-alpha_t) *  (p_t - x)**n1 / (p_t - p_e)**n1 + (alpha_0-1)/(alpha_e-1) * (alpha_t - 1) * (p_c - x)**n2 / (p_c - p_e)**n2 + 1
        y2 = (alpha_0-1)/(alpha_e-1) * (alpha_t - 1) * (p_c - x)**n2 / (p_c - p_e)**n2 + 1
    else:
        print("Invalid crushcurve_style. Exiting...")
        sys.exit(1)

    # plot it (add current subplot to fig)
    ax = fig.add_subplot(n_files, 1, n)
    if crushcurve_style == 0:
        ax.plot(x, y, '--', c='darkgray', label='crush curve')
    elif crushcurve_style == 1:
        ax.plot(x, y1, '--', c='darkgray', label='crush curve')
        ax.plot(x, y2, '--', c='darkgray', label='crush curve')

    if args.mat_type is not None:
        labeltext = "SPH particles (mat-type: {})".format(args.mat_type)
    else:
        labeltext = "SPH particles"
    sc = ax.scatter(p, alpha, c=dalphadt, s=2, label=labeltext)

    ax.set_xlim(p_min, p_max)
    ax.set_ylim(alpha_min, alpha_max)
    ax.grid(True, color='gray', linestyle=':')
    ax.legend(loc='upper right')
    ax.set_title("File: {}    Time: {:g}".format(currentfile, time) )
    ax.set_xlabel(r'Pressure [Pa]')
    ax.set_ylabel(r'Distention [1]')
    clb = plt.colorbar(sc, ax=ax)
    clb.set_label('dalpha/dt')


if n != n_files:
    print("Strange mismatch in number of files vs. number of plots...")
    sys.exit(1)

if args.v:
    print("\nWriting image to {}...\n".format(args.imagefile) )
fig.savefig(args.imagefile, dpi=200)
