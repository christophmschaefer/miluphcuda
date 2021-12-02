#!/bin/sh
nice -19 ./miluphcuda  -f dam.0000 -t 0.1 -n 100 -v -H > output.txt 2> error.txt
