#!/bin/bash
# Compile miluphcuda for a matrix of parameter.h switch combinations.
#
# usage: utils/compile_matrix.sh [-j N] [-c configfile] [-k] [-n]
#   -j N           parallel make jobs (default 8)
#   -c configfile  one configuration per line: <name> KEY=VAL KEY=VAL ...  (default: built-in list below)
#                  lines starting with # are ignored
#   -k             keep the build directories (default: removed after a successful build)
#   -n             no make: compile every src/*.cu with "$NVCC -dc" and device-link only (needs NVCC, NVCCFLAGS);
#                  useful without libhdf5/libconfig for linking; extra -I flags via EXTRA_INCLUDES
#
# Every configuration is built in its own copy of src/, include/ and the Makefile (under ${TMPDIR:-/tmp}),
# starting from the current include/parameter.h. A switch that does not exist in parameter.h is an error.
# Exit status: number of failed configurations.

set -u
JOBS=8
CONF=""
KEEP=0
NOMAKE=0
while getopts "j:c:kn" opt; do
    case $opt in
        j) JOBS=$OPTARG ;;
        c) CONF=$OPTARG ;;
        k) KEEP=1 ;;
        n) NOMAKE=1 ;;
        *) sed -n '2,15p' "$0"; exit 1 ;;
    esac
done

ROOT=$(cd "$(dirname "$0")/.." && pwd)
NVCC=${NVCC:-nvcc}
NVCCFLAGS=${NVCCFLAGS:-"-arch=sm_61 -O3 -Wno-deprecated-gpu-targets"}

read -r -d '' DEFAULT_CONFIGS <<'EOF'
default
hydro                   SOLID=0 HYDRO=1 FRAGMENTATION=0 PALPHA_POROSITY=0 STRESS_PALPHA_POROSITY=0 COLLINS_PLASTICITY_SIMPLE=0
dim2                    DIM=2
dim1                    DIM=1 TENSORIAL_CORRECTION=0 TENSORIAL_CORRECTION_CONSERVE_ANGULAR_MOMENTUM=0
kernelsum               INTEGRATE_DENSITY=0
kernelsum_tc_drhodt     INTEGRATE_DENSITY=0 TENSORIAL_CORRECTION_FOR_DRHODT=1
shepard                 SHEPARD_CORRECTION=1 TENSORIAL_CORRECTION_CONSERVE_ANGULAR_MOMENTUM=0
shepard_avg_kernelsum   SHEPARD_CORRECTION=1 TENSORIAL_CORRECTION_CONSERVE_ANGULAR_MOMENTUM=0 AVERAGE_KERNELS=1 INTEGRATE_DENSITY=0
average_kernels         AVERAGE_KERNELS=1
smlcorr                 SML_CORRECTION=1
smlcorr_avg_shepard     SML_CORRECTION=1 AVERAGE_KERNELS=1 SHEPARD_CORRECTION=1 TENSORIAL_CORRECTION_CONSERVE_ANGULAR_MOMENTUM=0
smlcorr_no_intsml       SML_CORRECTION=1 VARIABLE_SML=0 INTEGRATE_SML=0
fixed_noi               INTEGRATE_SML=0 FIXED_NOI=1
deal_with_too_many      DEAL_WITH_TOO_MANY_INTERACTIONS=1 TOO_MANY_INTERACTIONS_KILL_PARTICLE=0
collins_fragmentation   COLLINS_PLASTICITY_SIMPLE=0 COLLINS_PLASTICITY=1 FRAGMENTATION=1
palpha_return_mapping   PALPHA_RETURN_MAPPING=1
no_palpha               PALPHA_POROSITY=0 STRESS_PALPHA_POROSITY=0
no_fast_scheme          FAST_INTEGRATION_SCHEME=0
no_tensorial            TENSORIAL_CORRECTION=0 TENSORIAL_CORRECTION_CONSERVE_ANGULAR_MOMENTUM=0
EOF

if [ -n "$CONF" ]; then
    CONFIGS=$(grep -v '^\s*#' "$CONF" | grep -v '^\s*$')
else
    CONFIGS=$DEFAULT_CONFIGS
fi

BASE=$(mktemp -d "${TMPDIR:-/tmp}/compile_matrix.XXXXXX")
nfail=0
printf "%-26s %s\n" "configuration" "result"
while read -r name kvs; do
    [ -z "$name" ] && continue
    dir="$BASE/$name"
    mkdir -p "$dir"
    cp -r "$ROOT/src" "$ROOT/include" "$dir/"
    cp "$ROOT/Makefile" "$dir/" 2>/dev/null
    ok=1
    for kv in $kvs; do
        k=${kv%%=*}; v=${kv#*=}
        if ! grep -q "^#define $k " "$dir/include/parameter.h"; then
            echo "  $name: switch $k not found in parameter.h"; ok=0
        fi
        sed -i "s/^#define $k [-0-9]\+/#define $k $v/" "$dir/include/parameter.h"
    done
    if [ $ok -eq 1 ]; then
        if [ $NOMAKE -eq 0 ]; then
            (cd "$dir" && make -j "$JOBS" > build.log 2>&1) || ok=0
        else
            export NVCC NVCCFLAGS dir EXTRA_INCLUDES
            ls "$dir"/src/*.cu | xargs -P "$JOBS" -I{} sh -c \
                '$NVCC $NVCCFLAGS -x cu -c -dc -DVERSION=\"matrix\" -I"$dir/include" ${EXTRA_INCLUDES:-} \
                 -o "$(echo {} | sed "s/\.cu$/.o/")" {} > {}.log 2>&1 || touch "$dir/FAILED"'
            cat "$dir"/src/*.cu.log >> "$dir/build.log"
            [ -e "$dir/FAILED" ] && ok=0
            [ $ok -eq 1 ] && { $NVCC $NVCCFLAGS -dlink "$dir"/src/*.o -o "$dir/dlink.o" >> "$dir/build.log" 2>&1 || ok=0; }
        fi
    fi
    if [ $ok -eq 1 ]; then
        printf "%-26s ok  %s\n" "$name" "$kvs"
        [ $KEEP -eq 0 ] && rm -rf "$dir"
    else
        printf "%-26s FAIL  %s  (log: %s)\n" "$name" "$kvs" "$dir/build.log"
        grep -m3 "error" "$dir/build.log" | sed 's/^/      /'
        nfail=$((nfail+1))
    fi
done <<< "$CONFIGS"
[ $nfail -eq 0 ] && [ $KEEP -eq 0 ] && rmdir "$BASE" 2>/dev/null
echo "$nfail failed configuration(s)"
exit $nfail
