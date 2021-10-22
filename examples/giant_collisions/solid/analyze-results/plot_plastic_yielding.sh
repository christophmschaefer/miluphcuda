#!/bin/bash

# Produces plots which illustrate plastic yielding following the Collins plasticity model.

# Specify files to plot here:
FILES="../impact.00[1,2,3][0,5].h5"


# plot for Granite
./plot_plastic_yielding.Granite.py -v --files $FILES --imagefile plastic_yielding.Granite.png --mat_type 1 --color DAMAGE_TENSILE --pmin=-1e9 --pmax=5e9 --ymax=1.55e9

# plot for Iron
./plot_plastic_yielding.Iron.py -v --files $FILES --imagefile plastic_yielding.Iron.png --mat_type 0 --pmin=-2e9 --pmax=1.5e10 --ymax=11e9

