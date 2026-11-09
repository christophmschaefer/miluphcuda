How to set up the material config file for miluphcuda
=====================================================

last updated: 09/Nov/2026 (state of branch `cms_devel`)

Christoph Burger, Christoph Schäfer

----------------------------------------------------------------


Purpose
-------

miluphcuda supports multiple materials, and multiple rheologies. Many important settings for
the material model, especially material parameters, are parsed in a structured config file,
typically named *material.cfg*. In addition to material parameters, some numerical settings
(e.g., for the smoothing length) are also configured via that file.

The config file is processed by the *libconfig* library.
Check http://hyperrealm.github.io/libconfig/ for the latest version and documentation.

**Note**: This file is one of three places where you configure miluphcuda settings. The other two are:

* *parameter.h* (at compile time), see `doc/compile-time-options.md`
* *cmd-line flags*, see `doc/running.md`

The config file is passed to miluphcuda by the `-m` cmd-line option.

----------------------------------------------------------------


File structure
--------------

The config file consists of key-value pairs, organized in groups for different materials,
equation of state, etc. Groups are blocks enclosed in `{...}`.

The group *global* contains settings affecting all materials.
Settings for individual materials are organized in the list *materials*.

The basic file structure, containing *n* different materials, looks like:

        global = {
            ...
        }

        materials =
        (
            {
                ID = 0
                name = ...
                ...

                eos = {
                    type = ...
                    ...
                }
            },
            {
                ID = 1
                ...
            },
            .
            .
            .
            {
                ID = n-1
                ...
            }
        );

The available/required settings depend on the used material model(s), and are listed and described below.

Config files can be conveniently assembled by including external content using:

        @include "<file>"

Include directives can also be nested.

The `material.cfg` file in this directory contains an example setup, comprising several materials and material models.

Note that, for now, the `eos = {...}` group contains not only equation-of-state-specific entries,
but also most other parameters of the material model (elastic constants, strength, porosity).

----------------------------------------------------------------


Where to find material parameters?
----------------------------------

* The `material_data/` directory contains parameters for various materials.

* You can also check the `test_cases/` and `examples/` directories for various material setups.

----------------------------------------------------------------


Good to know...
---------------

* The entries in the config file should match their data type. Integer keys (e.g. `ID`, `eos.type`,
  `crushcurve_style`) need integer values. For float keys, write the value in floating point notation,

        cohesion = 1000.0   or   cohesion = 1.0e3

  libconfig requires a period or an exponent for floats; miluphcuda switches on the libconfig auto-conversion,
  so that `cohesion = 1000` is converted to a float as well, but older versions of the code did not.

* The code prints out all read material parameters if the verbose flag `-v` is used. Check them.

* If values are omitted, default values are assumed (see below). In most cases, these default values are not what you want.

* Keys that are not listed here are ignored (e.g. `aneos_e_norm` in some files in `material_data/`, which is used by
  external tools for initial conditions).

----------------------------------------------------------------


Global settings
---------------

The group *global* (see file structure above) contains options affecting all materials:

        Key         Type    Default         Details
        ___________________________________________

        c_gravity   float   6.67408e-11     grav. constant, determining code units (default is SI),
                                            check material_data/grav-const.cfg for some options

The optional group *disk* is only needed for the locally isothermal gas (EOS type 12) and the Shakura-Sunyaev
viscosity:

        Key                 Type    Default     Details
        _______________________________________________

        disk.scale_height   float   none        H/r of the disc; c_s = v_kep * scale_height

----------------------------------------------------------------


Equations of state
------------------

The EOS of a material is set by `eos.type`. The numbers correspond to `enum EquationOfStates` in
`include/pressure.h`:

        type    name                                parameters (in group eos)
        ________________________________________________________________________________

        -1      EOS_TYPE_IGNORE                     particle is ignored (deactivated)
         0      EOS_TYPE_POLYTROPIC_GAS             polytropic_K, polytropic_gamma        p = K rho^gamma
         1      EOS_TYPE_MURNAGHAN                  rho_0, bulk_modulus, n
         2      EOS_TYPE_TILLOTSON                  till_*, see below
         3      EOS_TYPE_ISOTHERMAL_GAS             isothermal_soundspeed                 p = c_s^2 rho
         4      EOS_TYPE_REGOLITH                   Bui et al. (2008) soil model (experimental), p = -tr(sigma)/3
         5      EOS_TYPE_JUTZI                      p-alpha porosity + Tillotson matrix
         6      EOS_TYPE_JUTZI_MURNAGHAN            p-alpha porosity + Murnaghan matrix
         7      EOS_TYPE_ANEOS                      ANEOS (or other tabulated EOS in that format)
         8      EOS_TYPE_VISCOUS_REGOLITH           experimental, do not use
         9      EOS_TYPE_IDEAL_GAS                  polytropic_gamma                      p = (gamma-1) rho e
        10      EOS_TYPE_SIRONO                     Sirono porosity model
        11      EOS_TYPE_EPSILON                    eps-alpha porosity + Tillotson matrix
        12      EOS_TYPE_LOCALLY_ISOTHERMAL_GAS     disk.scale_height and point masses    p = c_s^2 rho
        13      EOS_TYPE_JUTZI_ANEOS                p-alpha porosity + ANEOS matrix

The equations are given in the PDF documentation (`doc/miluphcuda_documentation.pdf`).

----------------------------------------------------------------


Individual material settings
----------------------------

Here, all settings for individual materials are described, ordered mainly by material model and/or
equation of state. For the keys in the following lists, the top level is assumed to be a specific material
and subgroups within materials are denoted by dot syntax. For example, *ID* is on the top level of some material,
and *eos.type* is the key *type* in the subgroup *eos*. (See above for the overall file structure.)

All units are SI, unless noted otherwise (strictly: the units are those of your initial conditions and of
`c_gravity`).


**Always mandatory**

        Key         Type    Default     Details
        _______________________________________

        ID          int     none        material ID, has to fit SPH particles' material-type
                                        in input file, IDs have to be 0, 1, 2,...

        eos.type    int     none        equation of state, see table above

--------------------------------

**Always optional**

        Key             Type    Default                             Details
        ___________________________________________________________________

        name            str     none                                name of material, currently not
                                                                    processed by miluphcuda

        density_floor   float   1% of the reference density         minimum density (absolute value)
                                (matrix density if porous), or      allowed for material
                                0. for gases

        energy_floor    float   -1e30 (i.e., none)                  minimum specific internal energy
                                                                    allowed for material

--------------------------------

**Smoothing length**

        Key                 Type    Default     Details
        _______________________________________________

        sml                 float   0.          smoothing length (= radius of the kernel support); only used
                                                if READ_INITIAL_SML_FROM_PARTICLE_FILE is not set in parameter.h
        interactions        int     -           wanted number of interaction partners, only for
                                                VARIABLE_SML + FIXED_NOI
        factor_sml_min      float   1.          with VARIABLE_SML: sml is kept >= factor_sml_min * sml
        factor_sml_max      float   1.          with VARIABLE_SML: sml is kept <= factor_sml_max * sml

Note: with VARIABLE_SML the defaults of 1. for both factors keep the smoothing length constant.

--------------------------------

**Elastic constants**

Which elastic constants are required depends on the used material model and equation of state.
The shear modulus is always required for solid simulations (`SOLID` is set in parameter.h).
The bulk modulus may be required for sound speed estimates, the Grady-Kipp fragmentation model, etc.

        Key                 Type      Default
        _____________________________________

        eos.bulk_modulus    float     0.
        eos.shear_modulus   float     0.

Note: Young's modulus is computed from them, E = 9 K mu / (3 K + mu).

--------------------------------

**Density via kernel sum**

        Key                         Type    Default     Details
        _______________________________________________________

        eos.density_via_kernel_sum  int     0           with INTEGRATE_DENSITY: 1 computes the density of this
                                                        material via the kernel sum instead of the continuity
                                                        equation

--------------------------------

**Artificial viscosity**

This is the standard artificial viscosity.

You need to set `ARTIFICIAL_VISCOSITY` in parameter.h (at compile time), and in the material config file:

        Key                             Type      Default
        _________________________________________________

        artificial_viscosity.alpha      float     1.
        artificial_viscosity.beta       float     2.

--------------------------------

**Artificial stress**

Needed (mandatory group) if `ARTIFICIAL_STRESS` is set in parameter.h:

        Key                                     Type    Default     Details
        ___________________________________________________________________

        artificial_stress.exponent_tensor       float   0.          exponent n of (W_ij/W(Delta p))^n
        artificial_stress.epsilon_stress        float   0.          factor epsilon of the artificial stress
        artificial_stress.mean_particle_distance float  0.          Delta p, the initial particle spacing

--------------------------------

**Physical viscosity (Navier-Stokes)**

Needed (mandatory group) if `NAVIER_STOKES` is set in parameter.h:

        Key                                 Type    Default     Details
        _______________________________________________________________

        physical_viscosity.nu               float   0.          kinematic viscosity, for
                                                                CONSTANT_KINEMATIC_VISCOSITY
        physical_viscosity.alpha_shakura    float   0.          alpha, for SHAKURA_SUNYAEV_ALPHA
        physical_viscosity.eta              float   0.          shear viscosity
        physical_viscosity.zeta             float   0.          bulk viscosity

--------------------------------

**EoS: Murnaghan**

To use the Murnaghan EoS for some material, set

        eos.type = 1

and the following parameters:

        Key                 Type    Default             Details
        _______________________________________________________

        eos.rho_0           float   0.                  reference density
        eos.bulk_modulus    float   0.
        eos.n               float   1.

        eos.rho_limit       float   0.9                 for rho/rho_0 < rho_limit the pressure is set
                                                        to zero; intended to avoid unphysically high
                                                        negative pressures where the material would
                                                        actually fragment or form droplets

        eos.cs_limit        float   0.                  lower limit for sound speed (in m/s)

Note: For n=1 the Murnaghan EoS reduces to the simple *liquid EoS*.

--------------------------------

**EoS: Tillotson**

To use the Tillotson EoS for some material, set

        eos.type = 2

and the following parameters:

        Key                 Type    Default             Details
        _______________________________________________________

        eos.till_rho_0      float   0.                  reference density
        eos.till_A          float   0.
        eos.till_B          float   0.
        eos.till_E_0        float   0.
        eos.till_E_iv       float   0.
        eos.till_E_cv       float   0.
        eos.till_a          float   0.
        eos.till_b          float   0.
        eos.till_alpha      float   0.
        eos.till_beta       float   0.

        eos.rho_limit       float   0.9                 for cold (i.e., if e <= till_E_iv) expanded states
                                                        with rho/rho_0 < rho_limit the pressure is set to
                                                        zero; intended to avoid unphysically high negative
                                                        pressures where the material would actually
                                                        fragment or form droplets; for till_E_iv < e <
                                                        till_E_cv only the vapour contribution is kept

        eos.cs_limit        float   1% of approx.       lower limit for sound speed (in m/s), default
                                    bulk sound speed    is 1% of sqrt(till_A/till_rho_0); can be used
                                                        to avoid unphysical values, like imaginary sound
                                                        speeds for negative pressures

        eos.polytropic_gamma float  not set             optional; if set (has to be > 1), completely
                                                        vaporized material (e > 100 till_E_cv and
                                                        rho < till_rho_0) is treated as ideal gas,
                                                        p = (polytropic_gamma - 1) rho e; without it the
                                                        Tillotson EoS is used throughout, which approaches
                                                        an ideal gas with gamma = till_a + 1 for large e;
                                                        note that p jumps at e = 100 till_E_cv unless
                                                        polytropic_gamma is close to till_a + 1

Note: for type 2, negative specific internal energies are set to zero in the EoS.

--------------------------------

**EoS: ANEOS (tabulated)**

To use a tabulated EoS (generated with ANEOS, or any table in the same format), set

        eos.type = 7

and the following parameters:

        Key                     Type    Default             Details
        ___________________________________________________________

        eos.table_path          str     none                path to the table file
        eos.n_rho               int     0                   number of density points in the table
        eos.n_e                 int     0                   number of energy points in the table
        eos.aneos_rho_0         float   0.                  reference density
        eos.aneos_bulk_cs       float   0.                  bulk sound speed (for defaults of cs_limit,
                                                            cs_porous)
        eos.aneos_molar_mass    float   0. (warning)        molar mass (kg/mol), for the ideal gas fallback
        eos.aneos_gamma         float   0. (warning)        adiabatic index of the ideal gas fallback, used
                                                            for e above the table maximum
        eos.cs_limit            float   1% of aneos_bulk_cs lower limit for sound speed (in m/s)

Pressure and sound speed are interpolated bilinearly in the table. Energies below the table minimum are clamped to
the cold curve, energies above the maximum use the ideal gas fallback p = (aneos_gamma - 1) rho e.
The tables can be generated externally, e.g. with the *miluphANEOS* project.
With `MORE_ANEOS_OUTPUT` or `ANEOS_VAPOR_NO_STRENGTH`, temperature, entropy and the phase flag are read from the
table as well.

--------------------------------

**EoS: gases**

        Key                         Type    Default     Details
        _______________________________________________________

        eos.polytropic_K            float   0.          type 0
        eos.polytropic_gamma        float   0.          types 0 and 9
        eos.isothermal_soundspeed   float   0.          type 3

--------------------------------

**Plasticity model: von Mises**

The von Mises model constitutes a single, constant yield strength.

You need to set `VON_MISES_PLASTICITY` in parameter.h (at compile time), and in the material config file:

        Key                 Type    Default
        ___________________________________

        eos.yield_stress    float   0.

--------------------------------

**Plasticity model: Collins**

The Collins model is based on a pressure-dependent yield strength, which also depends on whether
the material is intact or fragmented (*damaged*). See comments in parameter.h for more details.

You need to set `COLLINS_PLASTICITY` in parameter.h (at compile time), and in the material config file:

        Key                             Type    Default     Details
        ___________________________________________________________

        eos.cohesion                    float   0.          strength at p = 0 of intact material
        eos.friction_angle              float   0.          in rad
        eos.cohesion_damaged            float   0.          strength at p = 0 of damaged material
        eos.friction_angle_damaged      float   0.          in rad
        eos.yield_stress                float   0.          limit of the intact strength for p -> infinity

        eos.tensile_pressure_limit      float   0.          optional, >= 0; residual tensile strength of fully
                                                            damaged material: negative pressure is reduced by
                                                            (1-damage), but not beyond
                                                            -max(cohesion_damaged, tensile_pressure_limit)

        eos.melt_energy                 float   0.          to use this you need to additionally set
                                                            COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY
                                                            in parameter.h (see comments there)

--------------------------------

**Plasticity model: Collins simple**

This is a simplified version of the Collins model. See comments in parameter.h for more details.

You need to set `COLLINS_PLASTICITY_SIMPLE` in parameter.h (at compile time), and in the material config file:

        Key                     Type    Default     Details
        ___________________________________________________

        eos.cohesion            float   0.
        eos.friction_angle      float   0.          in rad
        eos.yield_stress        float   0.

--------------------------------

**Plasticity model: Drucker-Prager**

This is a linear dependence of the yield strength on pressure, intended for granular-like materials,
therefore the yield strength decreases to zero for p < 0.

You need to set `DRUCKER_PRAGER_PLASTICITY` in parameter.h (at compile time), and in the material config file:

        Key                     Type    Default     Details
        ___________________________________________________

        eos.cohesion            float   0.
        eos.friction_angle      float   0.          in rad

        eos.yield_stress        float   0.          optional upper limit for the yield strength,
                                                    where you have to additionally set
                                                    VON_MISES_PLASTICITY in parameter.h for this

--------------------------------

**Plasticity model: Mohr-Coulomb**

This is a linear dependence of the yield strength on pressure, intended for granular-like materials,
therefore the yield strength decreases to zero for p < 0.
Note that for p < 0 the slope is not set by the friction coefficient, but = 1 (i.e., the zero is at -cohesion).
A negative-pressure cap is applied: negative pressures are limited to zero of yield strength curve (at -cohesion).

You need to set `MOHR_COULOMB_PLASTICITY` in parameter.h (at compile time), and in the material config file:

        Key                     Type    Default     Details
        ___________________________________________________

        eos.cohesion            float   0.
        eos.friction_angle      float   0.          in rad

        eos.yield_stress        float   0.          optional upper limit for the yield strength,
                                                    where you have to additionally set
                                                    VON_MISES_PLASTICITY in parameter.h for this

--------------------------------

**Plasticity model: Johnson-Cook (experimental)**

For `JC_PLASTICITY`: `eos.jc_y0`, `eos.jc_B`, `eos.jc_n`, `eos.jc_m`, `eos.jc_edot0`, `eos.jc_C`, `eos.jc_Tref`,
`eos.jc_Tmelt`, `eos.Cp`, `eos.CV`. This model will be removed.

--------------------------------

**Fragmentation model: Grady-Kipp**

To activate the full Grady-Kipp model set `FRAGMENTATION` and `DAMAGE_ACTS_ON_S` in parameter.h (at compile time).
This is currently only possible globally for all materials at once.
Whether you want `DAMAGE_ACTS_ON_S` depends mainly on the desired interaction with the plasticity model (see comments in parameter.h).

The flaw distribution (activation thresholds) is part of the input file. The following parameters are only needed
for pre-processing tools that generate it:

        Key                         Type    Default     Details
        _______________________________________________________

        fragmentation.weibull_k     float   none        not processed by miluphcuda
        fragmentation.weibull_m     float   none        not processed by miluphcuda

--------------------------------

**Low-density weakening model**

Additional strength reduction for low-density states (below the reference density).

For most plasticity models, this is done by reducing the cohesion, and by that the whole yield envelope.
For COLLINS_PLASTICITY only the damaged cohesion is reduced, for VON_MISES_PLASTICITY all (constant) yield strength is reduced.

Strength reduction increases with decreasing density, where the functional form is set by the parameters below.
It drops from 1 to gamma between rho0 and eta_limit, with slope alpha, then it drops further from gamma to zero
between eta_limit and rho=0, with slope beta.
When combined with porosity, all densities are matrix densities.
Implemented for EoS types 1, 2 and 5.

You need to set `LOW_DENSITY_WEAKENING` in parameter.h (at compile time), and in the material config file:

        Key                                 Type    Default   Details
        _______________________________________________________________

        low_density_weakening.eta_limit     float   0.        defines transition density rho between
                                                              power law regimes, with eta = rho/rho0,
                                                              where rho0 is the reference density
                                                              (i.e., eta_limit in [0,1])
        low_density_weakening.alpha         float   0.        power-law exponent above eta_limit
        low_density_weakening.beta          float   0.        power-law exponent below eta_limit
        low_density_weakening.gamma         float   1.        sets cohesion-reducing factor at eta_limit
                                                              (i.e., gamma in [0,1])

Note: The defaults simply disable all low-density weakening for the respective material.

--------------------------------

**Porosity model: P-alpha**

See `doc/porosity.md` for a practical guide.

The P-alpha porosity model can be used together with several EoS, which all have a distinct *eos.type*
when used together with the P-alpha model:

* *eos.type = 5* ... P-alpha + Tillotson EoS
* *eos.type = 6* ... P-alpha + Murnaghan EoS
* *eos.type = 13* ... P-alpha + ANEOS

The parameters of the respective EoS refer to the matrix material (e.g., `till_rho_0` is the matrix density).
In addition, you need the following:

        Key                         Type    Default             Details
        _______________________________________________________________

        eos.crushcurve_style        int     0                   0 ... quadratic crush curve
                                                                1 ... steep crush curve (Jutzi et al. 2008)
                                                                2 ... experimental (Blum et al. 2023), constants
                                                                      hard-coded in pressure.cu
                                                                4 ... experimental (Malamud 2023), constants
                                                                      hard-coded in pressure.cu
        eos.porjutzi_p_elastic      float   0.                  pressure where compaction starts
        eos.porjutzi_p_transition   float   0.                  only required for crush curve style 1
        eos.porjutzi_p_compacted    float   0.                  pressure of full compaction

        eos.porjutzi_alpha_0        float   1.                  distention on the crush curve at p_elastic
        eos.porjutzi_alpha_e        float   1.                  only required for crush curve style 1
        eos.porjutzi_alpha_t        float   1.                  only required for crush curve style 1

        eos.porjutzi_n1             float   0.                  only required for crush curve style 1
        eos.porjutzi_n2             float   0.                  only required for crush curve style 1

        eos.cs_porous               float   50% of approx.      sound speed in uncompacted material
                                            bulk sound speed    (in m/s), where the actual sound speed
                                                                is a linear interpolation between this
                                                                and the sound speed in the matrix
                                                                material, based on distention

--------------------------------

**Porosity model: eps-alpha**

For `EPSALPHA_POROSITY` and *eos.type = 11* (Tillotson matrix):

        Key                         Type    Default     Details
        _______________________________________________________

        eos.porepsilon_kappa        float   0.          exponent kappa of the exponential compaction regime
        eos.porepsilon_alpha_0      float   0.          initial distention
        eos.porepsilon_epsilon_e    float   0.          volumetric strain at the elastic limit
        eos.porepsilon_epsilon_x    float   0.          volumetric strain at the transition to the power-law
                                                        regime
        eos.porepsilon_epsilon_c    float   0.          volumetric strain of full compaction

--------------------------------

**Porosity model: Sirono**

For `SIRONO_POROSITY` and *eos.type = 10*: `eos.porsirono_K_0`, `eos.porsirono_rho_0`, `eos.porsirono_rho_s`,
`eos.porsirono_gamma_K`, `eos.porsirono_alpha`, `eos.porsirono_pm`, `eos.porsirono_phimax`, `eos.porsirono_phi0`,
`eos.porsirono_delta`. See `src/porosity.cu` and `src/pressure.cu`.

--------------------------------

**Fast integration scheme**

With `FAST_INTEGRATION_SCHEME` set in parameter.h, a material that contains the (optional) group
`fast_integration` is switched at `transition_time` to a linear elastic medium with a low sound speed
(Raducan & Jutzi 2022, Jutzi et al. 2022), allowing much larger time steps in the late, low-velocity stage.
Supported EoS types: 1, 2, 7 (become type 1 with n = 1) and 5, 6, 13 (become type 6 with n = 1). Materials without
the group are never switched.

        Key                                 Type    Default             Details
        _______________________________________________________________________

        fast_integration.transition_time    float   none (mandatory)    simulation time of the switch, >= 0
        fast_integration.bulk_modulus       float   none (mandatory)    bulk modulus K of the new medium, > 0
        fast_integration.poisson_ratio      float   keep current        Poisson ratio nu of the new medium;
                                                                        the shear modulus becomes
                                                                        3K(1-2nu)/(2(1+nu)); the deviatoric
                                                                        stress is rescaled with the ratio of
                                                                        the shear moduli
        fast_integration.rho_0              float   reference density   reference density of the new medium
                                                    of the EoS
        fast_integration.rho_limit          float   1.                  rho_limit of the new medium (no
                                                                        tension below rho_0 with the default)

The sound speed after the switch is sqrt(K/rho_0). Example (regolith target):

        fast_integration = {
            transition_time = 2.0
            bulk_modulus = 7.8e5
        };
