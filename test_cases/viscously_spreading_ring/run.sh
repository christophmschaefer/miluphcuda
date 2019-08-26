#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
nice -19 ./miluphcuda -v -n 600 -H -t 1000 -f viscous_ring.0000 -m material_viscously_spreading_ring.cfg  -N `wc -l viscous_ring.0000` > output.txt 
