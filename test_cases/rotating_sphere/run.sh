#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda

echo "Generating input file"
./mk_sphere.py > sphere.0000

nice -19 ./miluphcuda -v -n 100 -H -t 6.283185307179586 -f sphere.0000 -m material.cfg > output.txt 

# plot total_angular_momentum (from conserved_quantities.log)
./plot_L.py
# see file angular_momentum.png

