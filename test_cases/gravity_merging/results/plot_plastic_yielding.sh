#!/bin/bash

# Produces plots which illustrate plastic yielding following the Collins plasticity model.

# Specify files to plot here:
FILES="../impact.0005.h5 ../impact.00[1-4][0,5].h5"


# plot it
./plot_plastic_yielding.Granite.py -v --files $FILES --imagefile plastic_yielding.png --mat_type 0 --color DAMAGE_TENSILE --pmin=-0.5e9 --pmax=1e9 --ymax=1.55e9

