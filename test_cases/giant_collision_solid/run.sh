#!/bin/bash

# set path to CUDA libs:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

# miluphCUDA command line:
nice -19 ./miluphcuda -v -I rk2_adaptive -Q 1e-4 -N `wc -l impact.0000` -n 100 -H -t 100.0 -M 10.0 -f impact.0000 -m material.cfg -s -g  > output.txt
