#!/bin/bash
# Builds the two executables of this test (test_cases/inviscid_sph_2d) from the miluphcuda source tree with
# parameter.h of this directory:
#   miluphcuda           INVISCID_SPH 1
#   miluphcuda_standard  INVISCID_SPH 0 (constant alpha = 1, beta = 2, for comparison)
# usage: ./build.sh [source tree, default ../..]
# make variables from the environment, e.g.  CUDA_DIR=/usr/local/cuda-12.8 GPU_ARCH=-arch=sm_61 JOBS=8 ./build.sh
set -e
here=$(cd "$(dirname "$0")" && pwd)
src=$(cd "${1:-$here/../..}" && pwd)
for v in 1 0; do
    b=$here/build_inviscid$v
    rm -rf "$b"
    mkdir -p "$b"
    cp -r "$src/src" "$src/include" "$src/Makefile" "$b/"
    sed -E "s/^#define INVISCID_SPH [01]/#define INVISCID_SPH $v/" "$here/parameter.h" > "$b/include/parameter.h"
    cp "$here/boundary.cu" "$b/src/boundary.cu"   # frozen outer layers, see there
    grep -q "^#define INVISCID_SPH $v" "$b/include/parameter.h"
    echo "building with INVISCID_SPH $v in $b"
    make -C "$b" -j "${JOBS:-8}" ${CUDA_DIR:+CUDA_DIR="$CUDA_DIR"} ${GPU_ARCH:+GPU_ARCH="$GPU_ARCH"} \
        > "$b/make.log" 2>&1 || { tail -30 "$b/make.log"; echo "build failed, see $b/make.log"; exit 1; }
done
cp "$here/build_inviscid1/build/miluphcuda" "$here/miluphcuda"
cp "$here/build_inviscid0/build/miluphcuda" "$here/miluphcuda_standard"
echo "built $here/miluphcuda and $here/miluphcuda_standard"
