#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
nice -19 ./miluphcuda -v -n 500 -H -t 1 -f rings.0000 -m material.cfg > output.txt 
