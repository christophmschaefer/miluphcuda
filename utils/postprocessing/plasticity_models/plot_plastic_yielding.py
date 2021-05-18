#!/usr/bin/env python3

"""
Plots particles' shear stresses from miluphcuda HDF5 output files + the theoretical
yield stress limit (parameters hardcoded below!), optionally for several files to one plot.

The input has to be HDF5 file(s) with the following data sets:

/p
/deviatoric_stress
/time

authors: Christoph Schaefer, Patricia Buzzatto, Christoph Burger
comments to: christoph.burger@uni-tuebingen.de

last updated: 18/May/2021
"""


# set plasticity params (from material.cfg)
yield_stress = 1.5e9
cohesion = 1e5
friction_angle = 1.11   # this is mu_i = 2.0
#friction_angle = 0.98   # this is mu_i = 1.5
cohesion_damaged = 0.0
friction_angle_damaged = 0.675   # this is mu_d = 0.8
melt_energy = 1e6


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


parser = argparse.ArgumentParser(description="Plots particles' shear stresses from miluphcuda HDF5 output files + the theoretical yield stress limit (parameters hardcoded in the script!).")
parser.add_argument("-v", help = "be verbose", action = 'store_true')
parser.add_argument("--files", help = "specify one or more files to process", nargs='+', default = None)
parser.add_argument("--imagefile", help = "filename to write image to (default: plastic_yielding.png)", default = "plastic_yielding.png")
parser.add_argument("--mat_type", help = "select material type for plotting (default: all particles are plotted)", type=int, default = None)
parser.add_argument("--color", help = "select color-coding, either 'MELT_ENERGY', 'DAMAGE_TENSILE', or 'DAMAGE_TOTAL' (default: None)", default = None)
parser.add_argument("--pmin", help = "set min pressure for plot(s) (default: min value in file) (note: you may have to use '=' to set a negative value for this flag...)", type=float, default = None)
parser.add_argument("--pmax", help = "set max pressure for plot(s) (default: max value in file)", type=float, default = None)
parser.add_argument("--ymax", help = "set max shear stress for plot(s) (default: max value in file)", type=float, default = None)
args = parser.parse_args()

if args.files is None:
    print("You have to specify at least one file with --files. Exiting...")
    sys.exit(1)

n_files = len(args.files)
if args.v:
    print("\nReading {} file(s)...\n".format(n_files) )


# compute friction coefficients from friction angles
mu_i = np.tan(friction_angle)
mu_d = np.tan(friction_angle_damaged)
if args.v:
    print("Using parameters:")
    print("    yield stress: {:g}".format(yield_stress) )
    print("    cohesion: {:g}".format(cohesion) )
    print("    friction angle: {:g}   friction coefficient (mu_i): {:g}".format(friction_angle, mu_i) )
    print("    cohesion damaged: {:g}".format(cohesion_damaged) )
    print("    friction angle damaged: {:g}   friction coefficient damaged (mu_d): {:g}".format(friction_angle_damaged, mu_d) )
    print("    melt energy: {:g}".format(melt_energy) )


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
        ptmp = f['p'][:]
        Stmp = f['deviatoric_stress'][:]   # list of 9-element lists
        if args.color == 'MELT_ENERGY':
            colortmp = f['e'][:]
            colortmp /= melt_energy
        elif args.color == 'DAMAGE_TENSILE':
            colortmp = f['DIM_root_of_damage_tensile'][:]
            colortmp = colortmp**3
        elif args.color == 'DAMAGE_TOTAL':
            colortmp = f['damage_total'][:]

        p = []
        S = []
        if args.color is not None:
            color = []
        for i in range(0, len(mat_types)):
            if mat_types[i] == args.mat_type:
                p.append(ptmp[i])
                S.append(Stmp[i])
                if args.color is not None:
                    color.append(colortmp[i])
        p = np.asarray(p, dtype=float)
        S = np.asarray(S, dtype=float)
        assert len(p) == len(S), "ERROR. Strange mismatch in array lengths..."
        if args.color is not None:
            color = np.asarray(color, dtype=float)
            assert len(p) == len(color), "ERROR. Strange mismatch in array lengths..."

        if args.v:
            print("    found {} particles with mat-type {}".format(len(p), args.mat_type) )

    # extract all particles (all mat-types)
    else:
        p = f['p'][:]
        S = f['deviatoric_stress'][:]   # list of 9-element lists
        if args.color == 'MELT_ENERGY':
            color = f['e'][:]
            color /= melt_energy
        elif args.color == 'DAMAGE_TENSILE':
            color = f['DIM_root_of_damage_tensile'][:]
            color = color**3
        elif args.color == 'DAMAGE_TOTAL':
            color = f['damage_total'][:]

    # compute sqrt(J2)
    s11 = S[:, 0]   # list of S_11 for all particles
    s22 = S[:, 4]
    s33 = S[:, 8]
    s12 = S[:, 1]
    s23 = S[:, 5]
    s31 = S[:, 6]
    sqrt_J2 = np.sqrt( 1./6. * ( (s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2 ) + s12**2 + s23**2 + s31**2 )
# alternatively, direct computation of J2 (with identical results of course):
#    s13 = S[:, 2]
#    s21 = S[:, 3]
#    s32 = S[:, 7]
#    sqrt_J2 = np.sqrt( 0.5*(s11**2 + s12**2 + s13**2 + s21**2 + s22**2 + s23**2 + s31**2 + s32**2 + s33**2) )

    f.close()


    # set plot limits
    p_min = np.amin(p)
    if args.v:
        print("    found min p: {:g}".format(p_min) )
    if args.pmin is not None:
        p_min = args.pmin
    p_max = np.amax(p)
    if args.v:
        print("    found max p: {:g}".format(p_max) )
    if args.pmax is not None:
        p_max = args.pmax
    y_min = 0.
    y_max = np.amax(sqrt_J2)
    if args.v:
        print("    found max sqrt(J2): {:g}".format(y_max) )
    if args.ymax is not None:
        y_max = args.ymax


    # compute yield strength curves
    x = np.linspace(p_min, p_max, 1000)
    y_i = cohesion + mu_i * x / (1. + mu_i * x / (yield_stress - cohesion) )
    y_d = cohesion_damaged + mu_d * x
    y_DP = cohesion + mu_i * x

    # set yield strengths to zero left of their zeros
    y_i_zero = -cohesion * (yield_stress-cohesion) / (mu_i*yield_stress)  # zero of Y_i curve
    for i in range(0, len(x)):
        if x[i] < y_i_zero or y_i[i] < 0.:
            y_i[i] = 0.
        if y_d[i] < 0.:
            y_d[i] = 0.
        if y_DP[i] < 0.:
            y_DP[i] = 0.


    # plot it (add current subplot to fig)
    ax = fig.add_subplot(n_files, 1, n)
    ax.plot(x, y_DP, '--', c='cyan', label=r'$Y_\mathrm{Drucker-Prager}$')
    ax.plot(x, y_i, '--', c='blue', label=r'$Y_\mathrm{intact}$')
    ax.plot(x, y_d, '--', c='red', label=r'$Y_\mathrm{damaged}$')
    ax.axhline(y=cohesion, linestyle='--', c='darkgray', label='cohesion')
    ax.axhline(y=yield_stress, linestyle='--', c='grey', label='von Mises limit')

    if args.mat_type is not None:
        labeltext = "SPH particles (mat-type: {})".format(args.mat_type)
    else:
        labeltext = "SPH particles"
    if args.color is not None:
        sc = ax.scatter(p, sqrt_J2, c=color, s=2, label=labeltext)
    else:
        ax.scatter(p, sqrt_J2, c='black', s=2, label=labeltext)

    ax.set_xlim(p_min, p_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, color='gray', linestyle=':')
    ax.legend(loc='best')
    ax.set_title("File: {}    Time: {:g}".format(currentfile, time) )
    ax.set_xlabel(r'Pressure [Pa]')
    ax.set_ylabel(r'Shear stress ($\sqrt{J_2}$) [Pa]')

    if args.color is not None:
        clb = plt.colorbar(sc, ax=ax)
        if args.color == 'MELT_ENERGY':
            clb.set_label("e/e_melt")
        elif args.color == 'DAMAGE_TENSILE':
            clb.set_label("Damage tensile")
        elif args.color == 'DAMAGE_TOTAL':
            clb.set_label("Damage total")


if n != n_files:
    print("Strange mismatch in number of files vs. number of plots...")
    sys.exit(1)

if args.v:
    print("\nWriting image to {}...\n".format(args.imagefile) )
fig.savefig(args.imagefile, dpi=200)
