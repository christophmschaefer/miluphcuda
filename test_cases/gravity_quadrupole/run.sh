#!/bin/bash
#
# gravity_quadrupole test case: accuracy and cost of the Barnes-Hut tree self-gravity for
#     mono:      GRAVITY_QUADRUPOLE 0, GRAVITY_OPEN_SOFTENED_NODES 0  (the old tree)
#     quad:      GRAVITY_QUADRUPOLE 1, GRAVITY_OPEN_SOFTENED_NODES 0
#     quad_open: GRAVITY_QUADRUPOLE 1, GRAVITY_OPEN_SOFTENED_NODES 1
# in 3D and 2D for several theta, compared to a direct summation (check_gravity.py).
#
# The executables are built from the sources in ../.. with the parameter.h of this directory (DIM set by sed),
# in build_<dim>d_<variant> here; include/parameter.h and build/ of the source tree are not touched.
#
# Adapt if necessary:
#   MAKE_ARGS   extra arguments for make, e.g. MAKE_ARGS="CUDA_DIR=/usr/local/cuda GPU_ARCH=-arch=sm_61"
#               the makefile has to use INC_DIR and BUILD_DIR like the Makefile in the repository
#   THETAS      list of opening angles
#   DIMS        list of dimensions, "3 2"
#   SKIP_BUILD  1: use the executables of a previous run.sh (e.g. for other THETAS)
# needs python3 with numpy and h5py

set -e

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

HERE=$(pwd)
SRC=$(cd ../.. && pwd)
THETAS=${THETAS:-"0.3 0.5 0.7 0.9"}
DIMS=${DIMS:-"3 2"}
MAKE_ARGS=${MAKE_ARGS:-}
SKIP_BUILD=${SKIP_BUILD:-0}
VARIANTS="mono:0:0 quad:1:0 quad_open:1:1"

for dim in $DIMS; do
    echo "Generating input file for DIM $dim"
    ./mk_setup.py --dim $dim > setup_${dim}d.0000
    [ $SKIP_BUILD = 1 ] && continue
    for v in $VARIANTS; do
        IFS=: read name quad open <<< "$v"
        bdir=$HERE/build_${dim}d_$name
        inc=$bdir/include
        rm -rf $inc
        mkdir -p $inc
        for f in $SRC/include/*.h; do
            ln -s $f $inc/
        done
        rm $inc/parameter.h
        sed -e "s/^#define DIM .*/#define DIM $dim/" \
            -e "s/^#define GRAVITY_QUADRUPOLE .*/#define GRAVITY_QUADRUPOLE $quad/" \
            -e "s/^#define GRAVITY_OPEN_SOFTENED_NODES .*/#define GRAVITY_OPEN_SOFTENED_NODES $open/" \
            parameter.h > $inc/parameter.h
        echo "Building ${dim}d_$name"
        make -C $SRC -j 8 $MAKE_ARGS INC_DIR=$inc BUILD_DIR=$bdir > build_${dim}d_$name.log 2>&1 \
            || { echo "build of ${dim}d_$name failed, see build_${dim}d_$name.log"; exit 1; }
    done
done

for dim in $DIMS; do
    material=material.cfg
    [ $dim = 2 ] && material=material_2d.cfg
    for v in $VARIANTS; do
        IFS=: read name quad open <<< "$v"
        for theta in $THETAS; do
            dir=run_${dim}d_${name}_${theta}
            echo "Running $dir"
            rm -rf $dir
            mkdir $dir
            cd $dir
            ln -s ../setup_${dim}d.0000 setup.0000
            # 5 outputs of 2e-4 s from rest (Euler): the particles move by less than 1e-9 m before setup.0001.h5, so
            # its gravitational acceleration can be compared to the direct sum over its positions; the further steps
            # are only for the timing (DEBUG_RHS_RUNTIMES)
            $HERE/build_${dim}d_$name/miluphcuda -v -H -A -I euler -n 5 -t 2e-4 -f setup.0000 -m ../$material \
                -s -a $theta > output.txt 2> error.txt
            cd $HERE
        done
    done
done

{
    status=0
    for dim in $DIMS; do
        echo "DIM $dim"
        ./check_gravity.py --prefix run_${dim}d --thetas $THETAS --variants mono quad quad_open || status=1
        echo
    done
    exit $status
} 2>&1 | tee results.txt
exit ${PIPESTATUS[0]}
