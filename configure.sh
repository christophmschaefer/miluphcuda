#!/bin/sh
# Configure the CUDA GPU architecture in the miluphcuda Makefile.
# Usage: ./configure.sh [Makefile]
# Override detection with, for example: GPU_ARCH=sm_86 ./configure.sh

set -u

makefile=${1:-Makefile}

error() { printf 'Error: %s\n' "$*" >&2; exit 1; }
warn()  { printf 'Warning: %s\n' "$*" >&2; }

[ "$#" -le 1 ] || error 'Usage: configure.sh [Makefile]'
[ -f "$makefile" ] || error "Makefile not found: $makefile"
[ -w "$makefile" ] || error "Makefile is not writable: $makefile"

command -v g++ >/dev/null 2>&1 || warn 'g++ not found in PATH; check the compiler setting in the Makefile.'

if command -v nvcc >/dev/null 2>&1; then
    printf 'CUDA compiler: %s\n' "$(command -v nvcc)"
elif [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    printf 'CUDA compiler: %s\n' "$CUDA_HOME/bin/nvcc"
else
    warn 'nvcc not found; check the CUDA installation and Makefile paths.'
fi

# Prefer an explicit override, which also works on login nodes without GPUs.
arch=${GPU_ARCH:-}
if [ -z "$arch" ]; then
    command -v nvidia-smi >/dev/null 2>&1 || error 'nvidia-smi unavailable; specify GPU_ARCH=sm_XX.'
    gpu_info=$(nvidia-smi --query-gpu=compute_cap,name --format=csv,noheader -i 0 2>/dev/null) || \
        error 'Cannot query GPU 0; specify GPU_ARCH=sm_XX.'
    capability=${gpu_info%%,*}
    capability=$(printf '%s' "$capability" | tr -d '[:space:]')
    arch=$capability
    printf 'Detected GPU: %s\n' "${gpu_info#*,}"
fi

# Accept 8.6, 86, sm_86, or -arch=sm_86.
case $arch in
    -arch=sm_*) arch=${arch#-arch=sm_} ;;
    sm_*)       arch=${arch#sm_} ;;
esac
case $arch in
    *.*)
        major=${arch%%.*}
        minor=${arch#*.}
        case $major in ''|*[!0-9]*) error "Invalid GPU architecture: $arch" ;; esac
        case $minor in ''|*[!0-9]*) error "Invalid GPU architecture: $arch" ;; esac
        arch=${major}${minor}
        ;;
esac
case $arch in
    ''|*[!0-9]*) error "Invalid GPU architecture: $arch" ;;
esac

# Match only an actual Makefile assignment, not GPU_ARCH_FLAGS or comments.
grep -Eq '^[[:space:]]*GPU_ARCH[[:space:]]*[:?+]?=' "$makefile" || \
    error "No GPU_ARCH assignment found in $makefile"

# Keep the original Makefile intact if writing the replacement fails.
dir=$(dirname "$makefile")
tmp=$(mktemp "$dir/.configure.XXXXXX") || error 'Cannot create temporary file.'
trap 'rm -f "$tmp"' EXIT HUP INT TERM

awk -v value="-arch=sm_$arch" '
    /^[[:space:]]*GPU_ARCH[[:space:]]*[:?+]?=/ {
        if (!updated) { print "GPU_ARCH = " value; updated = 1 }
        next
    }
    { print }
' "$makefile" > "$tmp" || error 'Failed to update Makefile.'

# Preserve file permissions when supported.
chmod --reference="$makefile" "$tmp" 2>/dev/null || chmod 644 "$tmp"
mv -f "$tmp" "$makefile" || error 'Failed to replace Makefile.'
trap - EXIT HUP INT TERM

printf 'Set GPU_ARCH = -arch=sm_%s in %s\n' "$arch" "$makefile"
printf 'Configuration complete. Review CUDA paths and run make.\n'
