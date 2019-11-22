#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
nice -19 miluphcuda -Q 1e-8 -v -n 100 -H -t 0.00228  -f shocktube.0000 -m material.cfg  > output.txt
