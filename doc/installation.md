# Installation

miluphcuda runs on a single Nvidia GPU. It is developed and used on Linux.

## Dependencies

**CUDA toolkit**
Install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads). The `Makefile` currently points to
CUDA 12.8; any recent version should work, but `GPU_ARCH` must be supported by the toolkit you use (see below).

**libconfig**
Parses the material config file. Install the package of your distribution (Debian/Ubuntu: `libconfig-dev`) or build
the [latest version](https://hyperrealm.github.io/libconfig/) from source:

    ./configure
    make
    sudo make install

**HDF5** (optional, but recommended)
Needed for HDF5 input and output files (`HDF5IO` in `parameter.h`, cmd-line flags `-H` and `-X`). Debian/Ubuntu:
`libhdf5-dev`. Many of the additional output fields (e.g. `MORE_OUTPUT`, `MORE_ANEOS_OUTPUT`, the total
integrated strain) are written only to HDF5 files.

**Python** (optional)
Many tools in `utils/`, `examples/` and `test_cases/` are Python 3 scripts using `numpy`, `h5py`, `scipy` and
`matplotlib`.

## Compilation

1. Edit the compile-time options in `include/parameter.h`, see [compile-time-options.md](compile-time-options.md).
   The `parameter.h` files in `examples/` and `test_cases/` are complete copies that you can use as a starting point.
2. Adapt the `Makefile` to your system (see below). `configure.sh` performs a few checks (compiler, CUDA path) and
   tries to determine the compute capability of GPU 0 via `nvidia-smi`; its lookup list only contains older GPU
   models, so you will often have to set `GPU_ARCH` yourself.
3. Run `make`. The executable is `build/miluphcuda`, object files go to `build/obj/`.

`make clean` removes `build/`. Since nearly all physics is selected by preprocessor switches, you have to recompile
(`make clean && make` is the safe choice) after every change in `parameter.h` or in one of the other headers listed in
[compile-time-options.md](compile-time-options.md). The object files depend on all headers and on the `Makefile`, so
a plain `make` also rebuilds everything after a header change.

## Makefile settings

| Variable | Meaning |
|---|---|
| `CC` | host compiler, also passed to `nvcc` via `-ccbin` (default `/usr/bin/g++`) |
| `CUDA_DIR` | CUDA installation, e.g. `/usr/local/cuda-12.8` or `/usr/local/cuda` |
| `GPU_ARCH` | compute capability of your GPU, e.g. `-arch=sm_61` for the GTX 1080 (Ti), `-arch=sm_86` for the RTX 30 series, `-arch=sm_89` for the RTX 40 series / L40, `-arch=sm_90` for the H100 |
| `INCLUDE_DIRS` | include paths, adjust the HDF5 path (`/usr/include/hdf5/serial` on Debian/Ubuntu) |
| `LDFLAGS` | library paths and libraries; without HDF5 use the commented second line, which drops `-lhdf5` |
| `NVFLAGS` | `nvcc` flags; for debugging with `cuda-gdb` use the commented line with `-G -lineinfo` |

**Important:** `GPU_ARCH` has to match the compute capability of the GPU. Look it up at
<https://developer.nvidia.com/cuda-gpus> or with `nvidia-smi --query-gpu=compute_cap --format=csv`. A wrong value
may compile without complaint and then crash or produce wrong results.

The version string printed by miluphcuda is taken from `git describe` at compile time; `MILUPHCUDA_VERSION` in
`include/miluph.h` is printed in addition.

## Running on a cluster

* `./miluphcuda -G` lists the GPUs on the host, `-d <id>` selects one.
* At run time the CUDA libraries must be found, e.g. `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`.
* miluphcuda writes its log to stdout and warnings/errors to stderr; redirect both into files for production runs.

## Building the documentation

The PDF needs a LaTeX installation with `pdflatex`, `bibtex` and the usual packages (`natbib`, `siunitx`,
`booktabs`, `hyperref`, ...):

    cd doc
    make

or `make documentation` in the main directory.
