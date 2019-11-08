#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda/lib64
./miluphcuda  -f impact.0000 -m material.cfg -n 500 -t 1e-7 -I rk2_adaptive -v -H  > output.txt
