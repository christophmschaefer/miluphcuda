# Running simulations

For each simulation you need

* a binary compiled with the right options in `include/parameter.h` ([compile-time-options.md](compile-time-options.md)),
* an input file with the initial particle distribution,
* a material config file, typically `material.cfg`
  ([CREATE-MATERIAL-CONFIG.md](../material-config/CREATE-MATERIAL-CONFIG.md)),
* suitable command-line options.

The quickest way to get started is to run one of the scenarios in `examples/` (see their `USAGE.md`) and to modify it.

A typical call (from `examples/impact/run.sh`):

    miluphcuda -v -A -H -f impact.0000 -m material.cfg -I rk2_adaptive -Q 1e-4 -n 200 -t 5e-4 \
        1>miluphcuda.output 2>miluphcuda.error

This reads `impact.0000`, writes 200 HDF5 output files `impact.0001.h5` ... `impact.0200.h5` at the times
`5e-4`, `1e-3`, ... `0.1` (no ASCII output because of `-A`), and integrates with the adaptive Runge-Kutta scheme
with a relative accuracy of `1e-4`.

## Command-line options

`./miluphcuda -h` prints the options of your binary. Options marked HDF5 exist only if compiled with `HDF5IO`.

| Option | Argument | Default | Meaning |
|---|---|---|---|
| `-h`, `--help` | | | print help and exit |
| `-v`, `--verbose` | | off | verbose output on stdout; prints, among others, all material parameters read from the config file and the integrator settings. Recommended. |
| `-Y`, `--format` | | | print the input and output file formats and the compile-time options of the binary, then exit |
| `-G`, `--information` | | | print information about the Nvidia GPUs on the host |
| `-d`, `--device_id` | int | 0 | GPU to use |
| `-f`, `--filename` | name | `disk.0000` | input file, see [Input files](#input-files) |
| `-X`, `--hdf5_input` | | off | HDF5: read the input from `<name>.h5`, where `<name>` is given with `-f` (without `.h5`) |
| `-r`, `--restart` | | off | the ASCII input file is an output file of an earlier run (output format instead of input format) |
| `-m`, `--materialconfig` | name | `material.cfg` | material config file |
| `-n`, `--num` | int | | number of output files to write; with a restart, the number of additional ones |
| `-t`, `--timeperstep` | value | 1.0 | simulation time between two output files |
| `-T`, `--starttime` | value | 0.0 | start time; read from the file for HDF5 input |
| `-I`, `--integrator` | name | `rk2_adaptive` | time integrator, see [Integrators](#integrators) |
| `-Q`, `--precision` | value | 1e-5 | relative accuracy of `rk2_adaptive` |
| `-M`, `--maxtimestep` | value | | upper limit of the time step; for `euler` the (fixed) time step |
| `-F`, `--firsttimestep` | value | | `rk2_adaptive` and `kdk`: (upper limit of the) first time step; default is the value of `-M`, or otherwise of `-t` |
| `-k`, `--kernel` | name | `cubic_spline` | SPH kernel, see [Kernels](#kernels) |
| `-s`, `--selfgravity` | | off | self-gravity, computed with the Barnes-Hut tree |
| `-a`, `--theta` | value | 0.5 | opening criterion of the Barnes-Hut tree; setting it also switches on self-gravity |
| `-D`, `--directselfgravity` | | off | self-gravity by direct summation, O(N²), only for small particle numbers |
| `-g`, `--decouplegravity` | | off | do not recompute the tree gravity in every right-hand side evaluation, see [Self-gravity](#self-gravity) |
| `-H`, `--hdf5_output` | | off | HDF5: write HDF5 output files in addition to the ASCII files |
| `-A`, `--no_ascii_output` | | off | do not write ASCII output files (use together with `-H`) |
| `-c`, `--cons_qu_file` | name | `conserved_quantities.log` | log file for the conserved quantities |
| `-e` | name | `binary_system.log` | log file for `BINARY_INFO` (not listed by `-h`) |
| `-L`, `--angular_momentum` | value | off | stop the simulation once the relative change of the total angular momentum exceeds `value` |
| `-b`, `--boundary_ratio` | value | 0 | ratio of additional ghost boundary particles (`GHOST_BOUNDARIES`); use the short form `-b`, the long form does not take the argument |

## Integrators

All integrators write an output file at multiples of `-t`; the time step is independent of the output interval.

| Name | Order | Time step | Notes |
|---|---|---|---|
| `rk2_adaptive` | 2 | adaptive, error control | default and recommended for most applications. Embedded Runge-Kutta scheme with error estimate from positions, velocities, density and energy, and additional limits (Courant condition, damage change, ...), configured in `include/rk2adaptive.h`. Rejected steps are repeated with a smaller time step. |
| `kdk` | 2 | adaptive, no error control | **cms_devel.** Kick-drift-kick leapfrog with one right-hand side evaluation per step. The time step is set before each step from the Courant condition and optional limits (`include/kdk.h`). Conserves the total angular momentum up to round-off if the forces are torque-free (`TENSORIAL_CORRECTION_CONSERVE_ANGULAR_MOMENTUM`, no XSPH, no boundaries, no tree gravity). |
| `euler` | 1 | fixed (`-M`) | for tests only |
| `euler_pc` | 2 | Courant and force limit, at most `-M` | predictor-corrector with predictor step dt; minimum values for the limits on the change rates in `pc_values.dat` |
| `monaghan_pc` | 2 | Courant and force limit, at most `-M` | predictor-corrector with predictor step dt/2 (Monaghan); `pc_values.dat` as above |
| `heun_rk4` | 2/4 | Courant and force limit, at most `-M` | Heun scheme for the SPH particles coupled with RK4 for gravitating point masses |

The physics (pressure, plasticity, damage, porosity, ...) is the same for all integrators. For p-alpha porosity with
`rk2_adaptive` see the time-step limits in [porosity.md](porosity.md).

## Kernels

All kernels have compact support with radius `h`, i.e. the smoothing length `h` of miluphcuda is the radius of the
interaction sphere, not half of it. Choose the kernel with `-k`:

| Name | Kernel |
|---|---|
| `cubic_spline` (default) | M4 cubic spline (Monaghan & Lattanzio 1985) |
| `quintic_spline` | M6 quintic spline |
| `wendlandc2`, `wendlandc4`, `wendlandc6` | Wendland C2, C4, C6 (Dehnen & Aly 2012) |
| `spiky` | spiky kernel of Desbrun & Cani (1996), 2D and 3D |

The definitions are in the PDF documentation and in `src/kernel.cu`; `utils/kernels/plot_kernels.py` plots them.
When you change the kernel, adapt the smoothing length: the Wendland kernels and the quintic spline need more
interaction partners than the cubic spline.

## Input files

### ASCII

The input file `<string>.XXXX` contains one line per particle with whitespace-separated columns. `XXXX` is a four-digit
number; output files are numbered consecutively starting from `XXXX`+1. The columns depend on the compile-time
options. **Always check them with `./miluphcuda -Y`** for your binary. In the most common case (`DIM 3`, `SOLID`,
`INTEGRATE_DENSITY`, `INTEGRATE_ENERGY`, `READ_INITIAL_SML_FROM_PARTICLE_FILE`) they are

    x y z  vx vy vz  mass  density  energy  sml  material-type  [flaw data]  S_xx S_xy S_xz S_yx ... S_zz  [porosity data]  [activation thresholds]

in this order:

1. positions (DIM columns), velocities (DIM columns), mass
2. density (if `INTEGRATE_DENSITY`), specific internal energy (if `INTEGRATE_ENERGY`), smoothing length (if
   `READ_INITIAL_SML_FROM_PARTICLE_FILE`)
3. material type, i.e. the material `ID` in the config file; `-1` deactivates a particle
4. `JC_PLASTICITY`: plastic strain, temperature
5. `FRAGMENTATION`: number of flaws, DIM-root of the tensile damage
6. `SOLID`: deviatoric stress tensor, DIM×DIM columns, row by row
7. `SIRONO_POROSITY`: 9 columns, see `-Y`
8. `PALPHA_POROSITY`: distention `alpha_jutzi`, pressure (the pressure column is read but not used as initial
   condition; set it to 0)
9. `EPSALPHA_POROSITY`: distention, volumetric strain
10. `FRAGMENTATION`: the activation threshold strains of the particle (as many as its number of flaws)

Every particle line must contain all columns, also for materials that do not use them (e.g. the stress tensor of a
gas). Empty lines are not allowed.

The ASCII input format is not the output format. To restart from an ASCII output file use `-r` (see below).

### HDF5

With `-X -f impact.0100` the file `impact.0100.h5` is read. It needs at least the datasets `/x`, `/v`, `/m`, `/rho`,
`/e`, `/sml`, `/material_type` and the datasets for the active physics (e.g. `/deviatoric_stress`, `/alpha_jutzi`,
flaw data); the time is read from `/time`. HDF5 output files of miluphcuda can be used directly as input.
Note: if an ASCII file with the name given by `-f` (without `.h5`) exists, the number of particles is taken from that
file.

### Gravitating point masses

With `GRAVITATING_POINT_MASSES`, the point masses are read from `<string>.XXXX.mass` (or `<string>.XXXX.mass.h5`), one
line per point mass: position (DIM), velocity (DIM), mass, `rmin`, `rmax`. SPH particles closer than `rmin` or farther
than `rmax` from a point mass are removed from the simulation (or accreted, see `PARTICLE_ACCRETION`).

## Output files

For every output time, miluphcuda writes

* `<string>.XXXX`: ASCII output (unless `-A`), columns see `./miluphcuda -Y`. These contain additional columns
  compared to the input (number of interaction partners, pressure, local strain, number of activated flaws, ...).
* `<string>.XXXX.h5`: HDF5 output (with `-H`). Use `h5ls -r` to see the datasets. Besides the state variables
  (`/x`, `/v`, `/m`, `/rho`, `/e`, `/p`, `/sml`, `/material_type`, `/soundspeed`, `/number_of_interactions`,
  `/deviatoric_stress`, `/time`, ...), it contains, depending on the switches: damage (`/DIM_root_of_damage_tensile`,
  `/DIM_root_of_damage_porjutzi`, `/damage_total`, `/number_of_activated_flaws`, `/activation_thresholds`, ...),
  porosity (`/alpha_jutzi`, `/dalphadt`, ...), the accumulated equivalent plastic strain `/total_plastic_strain`
  (all plasticity models) and the total integrated (von Mises equivalent) strain `/total_integrated_strain`
  (`SOLID`), extrema since the last output (`/p_min`, `/p_max`, `/rho_min`, `/rho_max`, `/e_min`, `/e_max`,
  `/cs_min`, `/cs_max`, with `MORE_OUTPUT`), ANEOS quantities (`/aneos_T`, `/aneos_cs`, `/aneos_entropy`,
  `/aneos_phase_flag`, with `MORE_ANEOS_OUTPUT`), time derivatives (`/drhodt`, `/dedt`, `/ddeviatoric_stress_dt`,
  ...), accelerations (`/a`, `/a_grav`), and the tree depth.
* `<string>.XXXX.info`: time, particle numbers, total mass, kinetic and internal energy, momentum and angular momentum.
* `<string>.XXXX.mass` / `.mass.h5`: the gravitating point masses.

During the run:

* `conserved_quantities.log` (name set by `-c`): one line per output time with time, particle numbers, total mass,
  kinetic and internal energy, gravitational energy (`OUTPUT_GRAV_ENERGY`), momentum, angular momentum, and the
  barycentre. The header line lists the columns.
* `binary_system.log` (`BINARY_INFO`): semi-major axis and eccentricity of a binary of point masses.
* stdout: progress, time steps and the limiting criteria; stderr: warnings and errors. With `-v` much more is printed.
  Warnings to watch for: particles with too many interaction partners (`MAX_NUM_INTERACTIONS`), ill-conditioned
  tensorial correction matrices (with `DEBUG_DEVEL`), particles above the crush curve (p-alpha return mapping).

## Restarts

* From an HDF5 output file: `-X -f impact.0100 -n <additional outputs> ...`. Time and all state variables are read
  from the file, the numbering continues with `0101`.
* From an ASCII output file: `-r -f impact.0100 -T <time of that file> ...`. ASCII files contain no time, so you have
  to give it with `-T` (see the `.info` file).

Use the same binary (compile-time options) and the same material config file as for the original run. Quantities that
are not in the file start from zero; miluphcuda warns about this (e.g. for the plastic strain).

## Self-gravity

* `-s` computes self-gravity with a Barnes-Hut tree (monopoles only) with opening angle `-a` (default 0.5).
* `-D` uses direct summation instead, only reasonable for small particle numbers.
* `-g` decouples the gravity computation from the hydro time step: the gravitational accelerations are recomputed
  only every 10th right-hand side evaluation, or earlier if more than 0.1 % of the particles have moved to another
  tree node since the last computation; in between, the stored accelerations are used. This saves a lot of time
  when the hydro time step is much smaller than the gravitational time scale. The criterion is under revision.
* The gravitational constant (and with it the unit system) is set in the material config file (`global.c_gravity`,
  default SI).

## Examples and test cases

`examples/` contains complete, documented scenarios with expected results:

* `examples/impact/`: oblique impact of a basalt projectile (6 km/s) into a basalt half-sphere target
* `examples/giant_collisions/{hydro,solid}`: collision of two planetary bodies with self-gravity, hydro and solid version

`test_cases/` contains standard tests and validation cases, each with its own `parameter.h`, `material.cfg`,
`run.sh` and instructions:

| Test case | Tests |
|---|---|
| `shocktube` | shock tube |
| `sedov` | Sedov-Taylor blast wave |
| `dambreak` | dam break with simple boundary particles (own `boundary.cu`) |
| `colliding_rings` | two colliding elastic rings in 2D (Monaghan 2000), elasticity, tensile instability, plasticity |
| `rotating_sphere` | rigid rotation of a solid sphere, checks the tensorial correction and angular momentum conservation |
| `tensile_rod` | 2D rod pulled apart, plasticity and fragmentation (own `boundary.cu`) |
| `nakamura` | laboratory impact experiment of Nakamura & Fujiwara (1991), Grady-Kipp fragmentation |
| `gravity_merging` | merging of self-gravitating bodies |
| `viscously_spreading_ring` | viscously spreading ring (Speith & Kley 2003), Navier-Stokes |

Copy the `parameter.h` of a test case to `include/` (and the `boundary.cu` to `src/` where present), recompile,
and follow the instructions in the directory.

## Pre- and postprocessing tools

In `utils/`:

* `preprocessing/calc_giant_impact_geometry`: initial geometry of giant collisions
* `preprocessing/stl`: particle distributions from STL files
* `postprocessing/create_xdmf.py`: XDMF files for the HDF5 output, to open them in ParaView or VisIt
* `postprocessing/fast_identify_fragments_and_calc_aggregates`: fast fragment identification and gravitationally
  bound aggregates (C, see its `QuickStart.md`)
* `postprocessing/identify_fragments.py`, `determine_radius.py`, `density_contour.py`, `peak_pressures.py`,
  `trace_particle.py`, `map_sph_to_grid.c`
* `postprocessing/p_alpha_model/p_alpha_convergence.py`: checks how well the particles follow the crush curve
* `postprocessing/eps_alpha_model/`, `postprocessing/plasticity_models/`: diagnostics and plots of the yield curves
* `postprocessing/povray`, `blender`, `volume_rendering`: visualization
* `kernels/plot_kernels.py`: plots of the kernel functions
