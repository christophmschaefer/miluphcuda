#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda/lib64
nice -19 ./miluphcuda -v  -I rk2_adaptive -n 5 -t 1e-4 -f fluids.0000 -m material.cfg > output.txt
