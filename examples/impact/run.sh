#!/bin/bash

# Run script for impact example.

# If necessary, adapt the paths to the CUDA libs and the miluphcuda executable below, before running it.


# set path to CUDA libs [change if necessary]
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

# set path to miluphcuda executable [change if necessary]
MC=../../miluphcuda


# miluphcuda cmd line
$MC -v -A -f impact.0000 -H -I rk2_adaptive -Q 1e-4 -m material.cfg -n 200 -t 5e-4 1>miluphcuda.output 2>miluphcuda.error &

