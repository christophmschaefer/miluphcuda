#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64
#nice -19 ./miluphcuda -v -I rk2_adaptive -n 500 -H -t 1.0 -f rings_tc.0000 -m material.cfg -N `wc -l rings_tc.0000` --kernel cubic_spline > outputc.txt
nice -19 ../../miluphcuda -L 1e10 -v -I rk2_adaptive -n 500 -H -t 1 -f rings.0000 -m material.cfg -N `wc -l rings.0000` --kernel cubic_spline > output.txt 
#nice -19 ./miluphcuda -v -I rk2_adaptive -n 500 -H -t 1.0 -f rings_tw.0000 -m material.cfg -N `wc -l rings_tw.0000` --kernel wendlandc4 > outputw.txt
#nice -19 ./miluphcuda -v -I euler_pc -n 500 -H -t 1.0 -f rings.0000 -m material.cfg -N `wc -l rings.0000`
#nice -19 ./miluphcuda -v -I euler -n 1 -H -t 250 -f rings.0000 -m material.cfg -N `wc -l rings.0000` -M 1e-1
