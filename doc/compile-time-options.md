# Compile-time options

Most of the physical model and the numerics is selected by preprocessor switches. You have to recompile after
every change. `./miluphcuda -Y` prints the switches the binary was compiled with.

The main file is `include/parameter.h`. The column *repo* gives the value in `include/parameter.h` on `cms_devel`;
it is not a recommendation for your application. `0` means off, `1` on.
Inconsistent combinations are rejected at compile time by `include/checks.h`, see the [list at the end](#combinations-rejected-at-compile-time).

The equations behind the switches are given in the PDF documentation.

## parameter.h

### Basic model

| Switch | repo | Meaning |
|---|---|---|
| `DIM` | 3 | number of spatial dimensions, 1, 2 or 3 |
| `SOLID` | 1 | continuum mechanics with deviatoric stress, σ = −p 1 + S |
| `HYDRO` | 0 | hydrodynamics, only the (scalar) pressure; choose exactly one of `SOLID` and `HYDRO` |
| `REAL_HYDRO` | 0 | additionally set p = 0 wherever p < 0 |
| `INTEGRATE_DENSITY` | 1 | integrate the continuity equation; with 0 the density is the kernel sum Σ m_j W_ij. With 1 a material can still use the kernel sum, see `density_via_kernel_sum` in the material config |
| `INTEGRATE_ENERGY` | 1 | integrate the energy equation; required for energy-dependent EOS (Tillotson, ANEOS, ideal gas) |
| `SPH_EQU_VERSION` | 1 | SPH form of the momentum and energy equation: 1 = (σ_i/ρ_i² + σ_j/ρ_j²), 2 = (σ_i + σ_j)/(ρ_i ρ_j). If in doubt, use 1. `TENSORIAL_CORRECTION_CONSERVE_ANGULAR_MOMENTUM` needs 1. |

### Artificial and physical viscosity

| Switch | repo | Meaning |
|---|---|---|
| `ARTIFICIAL_VISCOSITY` | 1 | standard Monaghan α-β artificial viscosity; α and β per material in the config file (defaults 1 and 2) |
| `BALSARA_SWITCH` | 0 | Balsara (1995) switch, reduces the artificial viscosity in shear flows |
| `INVISCID_SPH` | 0 | time-dependent viscosity coefficients following Cullen & Dehnen (2010) |
| `NAVIER_STOKES` | 0 | physical viscosity (Navier-Stokes); choose one of the following two |
| `SHAKURA_SUNYAEV_ALPHA` | 0 | ν from the α prescription (`physical_viscosity.alpha_shakura`) |
| `CONSTANT_KINEMATIC_VISCOSITY` | 0 | constant ν (`physical_viscosity.nu`) |
| `KLEY_VISCOSITY` | 0 | artificial bulk viscosity of Schäfer et al. (2004); needs `NAVIER_STOKES` |

### Consistency corrections and related

| Switch | repo | Meaning |
|---|---|---|
| `SHEPARD_CORRECTION` | 0 | zeroth-order consistency (Shepard normalization of the kernel) |
| `TENSORIAL_CORRECTION` | 1 | first-order consistency: kernel gradients are multiplied with the inverse of the moment matrix (Randles & Libersky 1996). Needed to reproduce rigid rotation of solid bodies; used in the momentum and energy equation and in the velocity gradients (strain rate and rotation rate). The scheme is chosen in `src/kernel.cu`, see below. |
| `TENSORIAL_CORRECTION_FOR_DRHODT` | 0 | use the corrected velocity divergence also in the continuity equation (and in the sml equation). Off by default: with 1, the density can decouple from the particle geometry and the time step collapse. |
| `TENSORIAL_CORRECTION_CONSERVE_ANGULAR_MOMENTUM` | 0 | **cms_devel.** Build the moment matrix (and every velocity gradient using it) with the weight m_j/ρ_i instead of m_j/ρ_j. With `SPH_EQU_VERSION 1`, the momentum equation then conserves angular momentum exactly (in time with `-I kdk`), wherever the correction matrix is not reset to the identity. Needs `TENSORIAL_CORRECTION`, `SPH_EQU_VERSION 1`, the old-school scheme in `kernel.cu`, no `SHEPARD_CORRECTION`. |
| `ARTIFICIAL_STRESS` | 0 | artificial stress against the tensile instability (Monaghan 2000, Gray et al. 2001); needs the group `artificial_stress` in the material config; only for `SOLID` |
| `XSPH` | 0 | XSPH velocity smoothing for the particle advection |

### Smoothing length and neighbours

| Switch | repo | Meaning |
|---|---|---|
| `VARIABLE_SML` | 1 | the smoothing length changes in time; requires exactly one of the following two |
| `FIXED_NOI` | 0 | h is adapted to give a fixed number of interaction partners (`interactions` in the material config) |
| `INTEGRATE_SML` | 1 | dh/dt = h/DIM ∇·v is integrated; h is limited by `factor_sml_min` and `factor_sml_max` (material config) |
| `READ_INITIAL_SML_FROM_PARTICLE_FILE` | 1 | read h per particle from the input file instead of using `sml` from the material config |
| `SML_CORRECTION` | 0 | grad-h correction terms (experimental) |
| `AVERAGE_KERNELS` | 0 | 0: W_ij = W(r_ij, (h_i+h_j)/2); 1: W_ij = (W(r_ij, h_i) + W(r_ij, h_j))/2 |
| `MAX_NUM_INTERACTIONS` | 256 | maximum number of interaction partners per particle (fixed array size, memory!) |
| `TOO_MANY_INTERACTIONS_KILL_PARTICLE` | 1 | a particle with more than `MAX_NUM_INTERACTIONS` partners gets h = 0 and the run continues (announced on stdout); with 0 the run stops unless the following switch is set |
| `DEAL_WITH_TOO_MANY_INTERACTIONS` | 0 | instead, reduce h of that particle until it has fewer than `MAX_NUM_INTERACTIONS` partners |

### Strength (plasticity) models

Only for `SOLID`. Choose at most one of the following models; `VON_MISES_PLASTICITY` can additionally be combined
with Drucker-Prager or Mohr-Coulomb as an upper limit of the yield strength. The material parameters are listed in
[CREATE-MATERIAL-CONFIG.md](../material-config/CREATE-MATERIAL-CONFIG.md); `utils/postprocessing/plasticity_models/`
plots the yield curves.

| Switch | repo | Meaning |
|---|---|---|
| `VON_MISES_PLASTICITY` | 0 | constant yield strength `yield_stress` |
| `DRUCKER_PRAGER_PLASTICITY` | 0 | Drucker-Prager, √J₂ ≤ A + 3Bp with A, B from `cohesion` and `friction_angle`; yield strength 0 for sufficiently negative p; no cap on negative pressure |
| `MOHR_COULOMB_PLASTICITY` | 0 | Y = c + μp for p > 0 (μ = tan of `friction_angle`), Y = c + p for −c < p < 0; negative pressures are capped at −c |
| `COLLINS_PLASTICITY` | 0 | pressure-dependent strength of intact (Lundborg) and damaged material, mixed with the damage (Collins et al. 2004, Jutzi 2015); without `FRAGMENTATION` only the intact strength is used |
| `COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY` | 0 | factor (1 − e/e_melt) on the intact strength; only with `COLLINS_PLASTICITY` |
| `COLLINS_PLASTICITY_SIMPLE` | 1 | only the Lundborg curve for p > 0, Y = c + p for −c < p < 0, negative pressures capped at −c; not with `FRAGMENTATION` |
| `LOW_DENSITY_WEAKENING` | 0 | reduce the strength (the cohesion, for von Mises the yield strength, for Collins only the damaged cohesion) below the reference density; parameters in the group `low_density_weakening` |
| `VISCOUS_REGOLITH` | 0 | regolith as viscous fluid (experimental) |
| `PURE_REGOLITH` | 0 | Bui et al. (2008) soil model (experimental) |
| `JC_PLASTICITY` | 0 | Johnson-Cook plasticity (experimental, to be removed) |

The five standard models (von Mises, Drucker-Prager, Mohr-Coulomb, Collins, Collins simple) accumulate the
equivalent plastic strain from the radial return (output `/total_plastic_strain`).

### Fragmentation

| Switch | repo | Meaning |
|---|---|---|
| `FRAGMENTATION` | 0 | Grady-Kipp tensile damage model (Benz & Asphaug 1995); needs the flaw distribution in the input file |
| `DAMAGE_ACTS_ON_S` | 0 | damage reduces also the deviatoric stress, S → (1−D) S; damage always reduces negative pressure. Not with `COLLINS_PLASTICITY`. |
| `MAX_NUM_FLAWS` | 1 | maximum number of activation flaws per particle (array size); set to 1 without `FRAGMENTATION` |

### Porosity

See [porosity.md](porosity.md) for the p-alpha model.

| Switch | repo | Meaning |
|---|---|---|
| `PALPHA_POROSITY` | 1 | p-alpha model (Jutzi et al. 2008), EOS types 5, 6, 13 |
| `STRESS_PALPHA_POROSITY` | 1 | the deviatoric stress is also affected by the distention (Jutzi et al. 2008, eq. 45 ff.) |
| `PALPHA_F_ACTS_ON_ROTATION` | 0 | 1: the factor f scales the whole Hooke law including the Jaumann rotation terms, as in Jutzi et al. (2008); 0: f acts only on the volumetric part, rigid rotation of a porous particle is reproduced exactly |
| `PALPHA_ELASTIC_RELOADING` | 1 | **cms_devel.** The distention keeps the memory of the maximum pressure; an unloaded particle reloads elastically until it reaches the crush curve again. 0: Jutzi et al. (2008), eqs. 22–23, every pressure increase above `porjutzi_p_elastic` compacts (ratcheting under pressure oscillations) |
| `PALPHA_ELASTIC_RELOADING_TOL` | 1e-6 | **cms_devel.** tolerance for "below the crush curve", α < α_cc(p) − tol |
| `PALPHA_RETURN_MAPPING` | 0 | **cms_devel.** Determine α from ρ and e by solving the crush-curve condition (return mapping) instead of integrating dα/dt. α then follows the crush curve independently of the time step. Needs `PALPHA_ELASTIC_RELOADING 1`. |
| `SIRONO_POROSITY` | 0 | Sirono model modified by Geretshauser et al. (EOS type 10) |
| `EPSALPHA_POROSITY` | 0 | ε-α model of Wünnemann et al. (2006) (EOS type 11) |

### ANEOS

| Switch | repo | Meaning |
|---|---|---|
| `ANEOS_VAPOR_NO_STRENGTH` | 0 | particles of ANEOS materials (EOS types 7, 13) in the two-phase liquid/vapour state (ANEOS phase flag 2, also set for the ideal-gas fallback above the table) get S = 0 and no negative pressure. The zeroing is done where σ is assembled, so a particle that re-enters the solid phase starts with S = 0. Only meaningful for `SOLID`. |
| `MORE_ANEOS_OUTPUT` | 0 | see [Output](#output) |

### Fast integration scheme

| Switch | repo | Meaning |
|---|---|---|
| `FAST_INTEGRATION_SCHEME` | 1 | after a per-material transition time, the material is switched to a linear medium with low sound speed, so that the CFL condition allows much larger time steps (Raducan & Jutzi 2022, Jutzi et al. 2022). Only meaningful for the late, low-velocity stage of granular flow. Configured per material by the group `fast_integration` in the material config; materials without that group are not switched. |

### Gravitating point masses

| Switch | repo | Meaning |
|---|---|---|
| `GRAVITATING_POINT_MASSES` | 0 | additional point masses, read from `<input>.mass` |
| `PARTICLE_ACCRETION` | 0 | check whether a particle is bound to a point mass and has crossed its accretion radius `rmin` |
| `UPDATE_SINK_VALUES` | 0 | with `PARTICLE_ACCRETION`: add mass, momentum and centre of mass of accreted particles to the sink, the particle is removed; otherwise it continues orbiting |

### Boundaries

| Switch | repo | Meaning |
|---|---|---|
| `BOUNDARY_PARTICLE_ID` | −1 | material type used for boundary particles, see `src/boundary.cu` (experimental) |
| `GHOST_BOUNDARIES` | 0 | ghost particles for boundaries, ratio set by `-b` (experimental) |

### Output

| Switch | repo | Meaning |
|---|---|---|
| `HDF5IO` | 1 | HDF5 input and output (needs libhdf5) |
| `MORE_OUTPUT` | 1 | HDF5 output of the minimum and maximum of p, ρ, e and c_s of each particle since the last output |
| `MORE_ANEOS_OUTPUT` | 0 | HDF5 output of temperature, sound speed, entropy and phase flag from the ANEOS tables; only for ANEOS materials |
| `OUTPUT_GRAV_ENERGY` | 0 | compute the gravitational energy by direct summation at output times (expensive for large N) |
| `BINARY_INFO` | 0 | write `binary_system.log` with semi-major axis and eccentricity of a binary of point masses |

## Other headers

These settings rarely need to be changed. They are compiled in as well.

### include/timeintegration.h

| Setting | Value | Meaning |
|---|---|---|
| `COURANT_FACT` | 0.4 | Courant number; note that h is the full kernel support radius |
| `FORCES_FACT` | 0.2 | factor for the time-step limit from the accelerations, dt ~ √(h/a) |
| `NUM_THREADS_*` | | CUDA block sizes of the kernels; lower them if a kernel fails to launch on your GPU |
| `MAXDEPTH` | 64 | maximum depth of the Barnes-Hut tree |

### include/rk2adaptive.h (`-I rk2_adaptive`)

| Setting | Value | Meaning |
|---|---|---|
| `RK2_USE_COURANT_LIMIT` | 1 | limit dt by the CFL condition before each step, with the signal speed √(c_s² + 4/3 μ/ρ) for `SOLID` |
| `RK2_USE_FORCES_LIMIT` | 0 | limit dt by the accelerations |
| `RK2_USE_DAMAGE_LIMIT` | 1 | limit the change of the (DIM-root of the) damage per step to `RK2_MAX_DAMAGE_CHANGE` (0.15) |
| `RK2_USE_VELOCITY_ERROR`, `RK2_USE_DENSITY_ERROR`, `RK2_USE_ENERGY_ERROR` | 1 | quantities used in the error estimate (positions are always used) |
| `RK2_USE_VELOCITY_ERROR_POINTMASSES` | 0 | velocity error check also for point masses |
| `RK2_LIMIT_PRESSURE_CHANGE` | 0 | p-alpha: reject steps with a pressure change larger than the smallest `porjutzi_p_elastic` of all materials |
| `RK2_LIMIT_ALPHA_CHANGE` | 0 | p-alpha: reject steps with a change of α larger than `RK2_MAX_ALPHA_CHANGE` (1e-2); no effect with `PALPHA_RETURN_MAPPING` |
| `RK2_LOCATION_SAFETY` | 0.1 | positions: changes smaller than this times h are not considered in the error estimate |
| `MIN_VEL_CHANGE_RK2`, `RK2_TINY_DENSITY`, `RK2_TINY_ENERGY` | 10, 1e-2, 10 | the corresponding thresholds for velocity, density and energy (in code units) |
| `RK2_TIMESTEP_SAFETY` | 0.9 | safety factor for the next time step |
| `SMALLEST_DT_ALLOWED` | 1e-16 | the run is aborted if dt falls below |

The thresholds `MIN_VEL_CHANGE_RK2`, `RK2_TINY_DENSITY` and `RK2_TINY_ENERGY` are absolute values; adapt them if your
code units differ strongly from SI.

### include/kdk.h (`-I kdk`, cms_devel)

| Setting | Value | Meaning |
|---|---|---|
| `KDK_COURANT_FACT` | `COURANT_FACT` | Courant number, CFL condition always used |
| `KDK_USE_FORCES_LIMIT` | 1 | limit dt by the accelerations (`FORCES_FACT`) |
| `KDK_USE_DAMAGE_LIMIT` | 1 | limit the change of the DIM-root of the tensile damage per step to `KDK_MAX_DAMAGE_CHANGE` (0.15) |
| `KDK_MAX_ALPHA_CHANGE` | 1e-2 | limit the change of α per step (only with time integration of α, i.e. without `PALPHA_RETURN_MAPPING`) |
| `KDK_SMALLEST_DT` | 1e-16 | the run is aborted if dt falls below |

### src/kernel.cu (tensorial correction)

| Setting | Value | Meaning |
|---|---|---|
| `USE_OLDSCHOOL_KERNEL_GRADIENT_CORRECTION_SCHEME` | 1 | classical correction: invert the moment matrix; if the inverse is ill-conditioned (\|det\| < 0.2 or > 5, or an entry larger than `MAX_ABS_TENSORIAL_CORRECTION_ENTRY`), the identity is used for that particle |
| `USE_WEIGHTED_KERNEL_GRADIENT_CORRECTION_SCHEME` | 0 | blend between inverse and identity weighted by det of the moment matrix (Ren et al. 2023). Experimental, do not use. |
| `MIN_NUMBER_OF_INTERACTIONS_FOR_TENSORIAL_CORRECTION_TO_WORK` | 0 | below this number of partners the identity is used |
| `MAX_ABS_TENSORIAL_CORRECTION_ENTRY` | 5.0 | maximum absolute entry of the correction matrix |

### include/miluph.h

`DEBUG_*` switches for diagnostic output (time step, tree, gravity, rhs run times, I/O, pressure, ...). Several are
on in the repository and produce a lot of output on stdout; switch them off for production runs.
`MILUPHCUDA_VERSION` is the version string.

## Combinations rejected at compile time

`include/checks.h` stops the compilation with a message, among others, for:

* not exactly one of `SOLID` and `HYDRO`
* more than one plasticity model, except `VON_MISES_PLASTICITY` with `DRUCKER_PRAGER_PLASTICITY` or
  `MOHR_COULOMB_PLASTICITY`; `JC_PLASTICITY` with any other model; any plasticity model, `FRAGMENTATION`,
  `VISCOUS_REGOLITH` or `PURE_REGOLITH` without `SOLID`
* `COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY` without `COLLINS_PLASTICITY`
* `COLLINS_PLASTICITY` with `DAMAGE_ACTS_ON_S`; `DAMAGE_ACTS_ON_S` without `FRAGMENTATION`
* `MOHR_COULOMB_PLASTICITY` or `COLLINS_PLASTICITY_SIMPLE` with `FRAGMENTATION`
* `LOW_DENSITY_WEAKENING` without one of the five standard plasticity models; with `COLLINS_PLASTICITY` but
  without `FRAGMENTATION`
* `TENSORIAL_CORRECTION_CONSERVE_ANGULAR_MOMENTUM` without `TENSORIAL_CORRECTION`, with `SPH_EQU_VERSION 2`, or
  with `SHEPARD_CORRECTION`
* `VARIABLE_SML` with both or none of `FIXED_NOI` and `INTEGRATE_SML`
* `NAVIER_STOKES` with both or none of the two viscosity models; `KLEY_VISCOSITY` without `NAVIER_STOKES`
* `ARTIFICIAL_STRESS` with `HYDRO`
* `STRESS_PALPHA_POROSITY` or `PALPHA_RETURN_MAPPING` without `PALPHA_POROSITY`; `PALPHA_RETURN_MAPPING` without
  `PALPHA_ELASTIC_RELOADING 1`
* `MORE_OUTPUT` or `MORE_ANEOS_OUTPUT` without `HDF5IO`
* `PARTICLE_ACCRETION` with `DIM 1`
