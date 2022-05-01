# miluphcuda

miluphcuda is a 3D Smoothed Particle Hydrodynamics (SPH) code, mainly developed for modeling astrophysical collision and
impact processes. It supports several rheology models for elasto-plastic solid bodies, as well as granular-like materials.

* miluphcuda is the CUDA port of the original miluph code.
* miluphcuda runs on single Nvidia GPUs with compute capability 5.0 and higher.

For an overview and quick start guide continue reading, for further details see the
[documentation](https://christophmschaefer.github.io/miluphcuda/index.html).


## Main features

* hydro and solid body modeling
* arbitrary number of interacting materials
* several material strength models for elasto-plastic flow
* fragmentation model for brittle-failure (Grady-Kipp)
* porosity models (P-alpha, epsilon-alpha, Sirono)
* self-gravity (via Barnes-Hut tree)
* various equations of state including tabulated (e.g., ANEOS)


## Quick start

1. Install dependencies and compile (see below).
2. Run one or more of the scenarios in `examples/`. You find detailed instructions there in the respective `USAGE.md`.

For generic simulations, based on your own initial conditions and/or material model choices, see *Usage* below.


## Compilation

Download the latest version of the code or clone the github repo.

After installation of the required dependencies (see below), compile via the `Makefile`:

1. run `configure.sh` which performs some tests and system-dependent configuration
2. run `make` to produce the `miluphcuda` executable

It may be necessary to make some changes to the `Makefile` even if `configure.sh` succeeds.
Compiler errors will let you know if that happens.

Note: Many important options for the material model and numerics are set at compile time in `parameter.h`,
i.e., you have to run the `Makefile` (but not `configure.sh`) again once you make changes there.


## Dependencies

**CUDA**  
Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) to compile and run code on the GPU.

**libconfig**  
Used for parsing the material config file.  
To ensure best functionality, install the [latest version](https://hyperrealm.github.io/libconfig):

1. unpack the .tar.gz archive
2. sudo ./configure
3. sudo make
4. sudo make install

**HDF5**  
If you want to use HDF5 for input and/or output files, install `libhdf5` and `libhdf5-dev`.


## Usage

The basic usage philosophy for miluphcuda is:

* Basic simulation-wide settings, mainly for the material model and numerics, are specified in `parameter.h` at compile-time.  
  See comments there for details.
* Material-specific parameters, mainly for the material model and equation of state, are set
  in a material config file (typically `material.cfg`) and processed at run-time.  
  See `material-config/CREATE-MATERIAL-CONFIG.md` for details on the file structure and available settings.  
  A collection of parameters for various materials is available in `material-config/material_data/`.
* All other main options are controlled via cmd-line flags. Check `./miluphcuda -h` once compiled.

For each simulation, you need:

* an input file which specifies the initial conditions (SPH particle distribution and their properties),
  check `./miluphcuda --format` for the required format
* a material config file, typically `material.cfg`, see above
* suitable cmd-line options for miluphcuda, check `./miluphcuda -h`

Specifically the `examples/`, but also the `test_cases/` provide various scenarios and use-cases.


## Developers

in somewhat arbitrary order:

Christoph M. Schaefer,
Sven Riecker,
Oliver Wandel,
Thomas I. Maindl,
Samuel Scherrer,
Janka Werner,
Christoph Burger,
Marius Morlock,
Evita Vavilina,
Michael Staneker,
Maximilian Rutz.


## References

 * [A versatile smoothed particle hydrodynamics code for graphic cards](https://ui.adsabs.harvard.edu/link_gateway/2020A&C....3300410S/doi:10.1016/j.ascom.2020.100410)
 * [A smooth particle hydrodynamics code to model collisions between solid, self-gravitating objects](https://ui.adsabs.harvard.edu/link_gateway/2016A&A...590A..19S/doi:10.1051/0004-6361/201528060)

The code is also listed in the [ACSL](https://ascl.net/1911.023).

