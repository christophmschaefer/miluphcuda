#!/bin/bash

# Produces plots which illustrate plastic yielding following the Collins plasticity model.

# Specify files to plot here:
FILES="../impact.000[2,4,6,8].h5 ../impact.001[0,2,4,6,8].h5"
#FILES="../impact.0005.h5 ../impact.00[1-9][0,5].h5"


# plot it
./plot_plastic_yielding.Basalt.py -v --files $FILES --imagefile plastic_yielding.png --mat_type 0 --color DAMAGE_TOTAL --pmin=-50e6 --pmax=50e6 --ymax=50e6

