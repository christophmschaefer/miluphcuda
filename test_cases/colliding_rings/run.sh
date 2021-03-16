#!/bin/bash

<<<<<<< HEAD
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
=======
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
>>>>>>> e584e9e456af9bc659413824335eb67544847775
nice -19 ./miluphcuda -v -n 500 -H -t 1 -f rings.0000 -m material.cfg > output.txt 
