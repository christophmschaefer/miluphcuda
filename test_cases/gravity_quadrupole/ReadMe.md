### Gravity quadrupole test case

Accuracy and cost of the Barnes-Hut tree self-gravity (`-s`) with the switches `GRAVITY_QUADRUPOLE` and
`GRAVITY_OPEN_SOFTENED_NODES` in parameter.h, compared to a direct summation.

Setup (`mk_setup.py`, particles at rest): a sphere with R = 1 km and a centrally concentrated mass distribution,
and next to it a prolate ellipsoid (600 x 300 x 300 m), both on a cubic lattice with 10% jitter, sml = 2.1 lattice
spacings; 25674 particles, sml = 116.7 m (`material.cfg`). The 2D version (`mk_setup.py --dim 2`) is a disc and an
ellipse on a square lattice, 5902 particles, sml = 52.5 m (`material_2d.cfg`). In 2D, miluphcuda uses the 3D law of
gravity in the plane, and so does the direct summation. Hydro only (Tillotson granite with rho = rho_0, e = 0, i.e. no pressure), so the particles do not move
during the short run and the gravitational acceleration of the output (`/a_grav`) can be compared to the direct sum
over the output positions, with the softening of miluphcuda, `G m_j r / max(r, h_i)^3`.

How to run it:
1. adapt `MAKE_ARGS` in run.sh if necessary (CUDA path, GPU_ARCH)
2. start `./run.sh` in this directory

run.sh builds three executables for each of DIM 3 and DIM 2 from the sources in ../.. with the parameter.h of this
directory, with DIM and the two switches set by sed (include/parameter.h and build/ of the source tree are not
touched; `DIMS="3"` for 3D only, `SKIP_BUILD=1` to rerun with the executables of a previous run):

| variant   | GRAVITY_QUADRUPOLE | GRAVITY_OPEN_SOFTENED_NODES |
|-----------|--------------------|-----------------------------|
| mono      | 0                  | 0 (the tree as before)      |
| quad      | 1                  | 0                           |
| quad_open | 1                  | 1                           |

runs each for theta = 0.3, 0.5, 0.7, 0.9 (in `run_<dim>d_<variant>_<theta>`) and calls `check_gravity.py` for each
DIM, which prints the relative error
|a_tree - a_direct| / |a_direct| (median, 90%, 99%, max; max is typically from a particle near the centre of the sphere (disc)
where |a_direct| -> 0), the 99% error normalised with the rms acceleration, the net force |sum m a| / sum m |a| (the
tree is not antisymmetric), and the median duration of selfgravity per call (from DEBUG_RHS_RUNTIMES, including
calculateCentersOfMass). The test passes if quad_open has a smaller median and 99% error than mono for every theta,
in each DIM. The output goes to results.txt.

Things to look at: the error of quad and quad_open compared to mono at the same theta, the theta at which
quad_open reaches the accuracy of mono at theta = 0.5 (the default), and t_grav at these two settings.
