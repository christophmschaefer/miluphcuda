#!/bin/bash
# Sod shock tube with ./miluphcuda (INVISCID_SPH 1) and, if present, ./miluphcuda_standard (INVISCID_SPH 0),
# 20 outputs up to t = 0.2, then the checks against the exact solution (check.py, writes sod.png).
set -e
cd "$(dirname "$0")"
[ -x ./miluphcuda ] || { echo "no ./miluphcuda, run ./build.sh first"; exit 1; }
python3 initial_conditions.py
runs=()
for v in inviscid standard; do
    exe=./miluphcuda
    [ $v = standard ] && exe=./miluphcuda_standard
    [ -x $exe ] || continue
    mkdir -p run_$v
    cp sod.0000 material.cfg run_$v/
    echo "running $exe in run_$v"
    (cd run_$v && ../$exe -v -H -I rk2_adaptive -Q 1e-6 -t 0.01 -n 20 -f sod.0000 -m material.cfg > output.txt 2>&1) \
        || { tail -20 run_$v/output.txt; exit 1; }
    runs+=(run_$v)
done
python3 check.py "${runs[@]}" --out sod.png
