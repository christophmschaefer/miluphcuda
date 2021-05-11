#!/bin/bash

# set path to CUDA libs [change if necessary]
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

# set path to miluphcuda executable [change if necessary]
MC=../../miluphcuda

# miluphcuda cmd line
$MC -f rod.0000 -m material.cfg -I rk2_adaptive -Q 1e-5 -v -n 250 -t 3e-2 -H -A  1>miluphcuda.output 2>miluphcuda.error &
#./miluphcuda -d 0 -v -I euler_pc -n 100 -H  -t 1e-6 -f rod.0000 -m material.cfg  > output.txt 
#./miluphcuda -Q 1e-4 -d 0 -v -n 1000 -H  -t 1e-2 -f rod.0368 -X  -m material.cfg  > output.txt 

