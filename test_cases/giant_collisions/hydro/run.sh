#!/bin/bash

# set path to CUDA libs:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

# set path to miluphcuda executable [change if necessary]
MC=../../../miluphcuda

# miluphcuda command line:
$MC -v -A -f impact.0000 -g -H -I rk2_adaptive -Q 1e-4 -m material.cfg -M 5.0 -n 75 -t 100.0 -s 

