#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
#./miluphcuda -d 0 -v -N `cat rod.0000 | wc -l` -I euler_pc -n 100 -H  -t 1e-6 -f rod.0000 -m material.cfg  > output.txt 
#./miluphcuda -Q 1e-4 -d 0 -v -N `cat rod.0000 | wc -l`  -n 1000 -H  -t 1e-2 -f rod.0368 -X  -m material.cfg  > output.txt 
./miluphcuda -Q 1e-4 -d 0 -v -N `cat rod.0000 | wc -l`  -n 1000 -H  -t 1e-2 -f rod.0000  -m material.cfg  > output.txt 2> error.txt
