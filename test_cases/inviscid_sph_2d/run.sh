#!/bin/bash
# Gresho vortex (t = 1) and planar Sod shock tube (t = 0.6) in 2D with ./miluphcuda (INVISCID_SPH 1) and, if
# present, ./miluphcuda_standard (INVISCID_SPH 0), then the checks (check.py, writes gresho.png and sod2d.png).
set -e
cd "$(dirname "$0")"
[ -x ./miluphcuda ] || { echo "no ./miluphcuda, run ./build.sh first"; exit 1; }
python3 initial_conditions.py
for v in inviscid standard; do
    exe=./miluphcuda
    [ $v = standard ] && exe=./miluphcuda_standard
    [ -x $exe ] || continue
    for s in "gresho 0.1" "sod2d 0.06"; do
        set -- $s
        d=run_${v}_$1
        mkdir -p $d
        cp $1.0000 material.cfg $d/
        echo "running $exe for $1 in $d"
        (cd $d && ../$exe -v -H -I rk2_adaptive -Q 1e-5 -k wendlandc2 -t $2 -n 10 -f $1.0000 -m material.cfg \
            > output.txt 2>&1) || { tail -20 $d/output.txt; exit 1; }
    done
done
python3 check.py
