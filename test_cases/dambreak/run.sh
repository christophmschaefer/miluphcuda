#!/bin/sh
nice -19 ./miluphcuda  -b 1.5 -k wendlandc2 -f dam.0000 -t 0.1 -n 100 -v -H > output.txt 2> error.txt
