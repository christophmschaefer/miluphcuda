#!/bin/bash

# set path to CUDA libs [change if necessary]
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

# set path to miluphcuda executable [change if necessary]
MC=../../miluphcuda

# miluphcuda cmd line
$MC -f impact.0000 -m material.cfg -n 500 -t 1e-7 -I rk2_adaptive -v -H -A 1>miluphcuda.output 2>miluphcuda.error &

