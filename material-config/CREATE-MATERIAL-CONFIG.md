How to set up the material config file for miluphcuda
=====================================================

last updated: 05/Mar/2021

Christoph Burger, Christoph Schäfer  
christoph.burger@uni-tuebingen.de

--------------------------------

Purpose
-------

miluphcuda supports multiple materials, and multiple rheologies. Many important settings for
the material model, especially material parameters, are parsed in a structured config file,
typically named *material.cfg*. In addition to material parameters, some numerical settings
(e.g., for the smoothing length) are also configured via that file.

The config file is processed by the *libconfig* library.
Check http://hyperrealm.github.io/libconfig/ for the latest version and documentation.

**Note**: This file is one of three places where you configure miluphcuda settings. The other two are:

* *parameter.h* (at compile time)
* *cmd-line flags*

The config file is passed to miluphcuda by the `-m` cmd-line option.

--------------------------------

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
                    ...
                }
            },
            {
                ID = 1
                name = ...
                ...

                eos = {
                    type = ...
                    ...
                    ...
                }
            },
            .
            .
            .
            {
                ID = n-1
                name = ...
                ...

                eos = {
                    type = ...
                    ...
                    ...
                }
            }
        );

The available/required settings depend on the used material model(s), and are listed and described below.

Config files can be conveniently assembled by including external content using:

        @include "<file>"

Include directives can also be nested.

The `material.cfg` file in this directory contains an example setup, comprising several materials and material models.

Note that, for now, the `eos = {...}` group regularly contains not only equation-of-state-specific entries,
but also further parameters related to the material model.

--------------------------------

Where to find material parameters?
----------------------------------

* The `material_data/` directory contains parameters for various materials.

* You can also check the `test_cases/` and `examples/` directories for various material setups.

--------------------------------

Good to know...
---------------

* The entries in the config file have to match their data type. For example, the ID tag has to be
  an integer value whereas the value for the cohesion has to be given in floating point notation:

        cohesion = 1000

  will not work but

        cohesion = 1000.0   or   cohesion = 1.0e3

  are valid entries. The libconfig specs indicate that a period is *always* mandatory for floats...

* The code prints out all read material parameters if the verbose flag `-v` is used.

* If values are ommitted, default values are assumed (see below). In most cases, these default values are not what you want.


--------------------------------

Global settings
-------------------------

The group *global* (see file structure above) contains options affecting all materials:

        Key         Type    Default         Details
        ___________________________________________

        c_gravity   float   6.67408e-11     grav. constant, determining code units (default is SI),
                                            check material_data/grav-const.cfg for some options


--------------------------------

Individual material settings
----------------------------

Here, all settings for individual materials are described, ordered mainly by material model and/or
equation of state. For the keys in the following lists, the top level is assumed to be a specific material
and subgroups within materials are denoted by dot syntax. For example, *ID* is on the top level of some material,
and *eos.type* is the key *type* in the subgroup *eos*. (See above for the overall file structure.)

All units are SI, unless noted otherwise.

This list is currently not exhaustive.


**Always mandatory**

        Key         Type    Default     Details
        _______________________________________

        ID          int     none        material ID, has to fit SPH particles' material-type
                                        in input file, IDs have to be 0, 1, 2,...

        eos.type    int     none        set equation of state for material, for a complete list see
                                        https://christophmschaefer.github.io/miluphcuda/pressure_8h.html
                                        or look directly into pressure.h

--------------------------------

**Always optional**

        Key             Type    Default                             Details
        ___________________________________________________________________

        name            str     none                                name of material, currently not
                                                                    processed by miluphcuda

        density_floor   float   1% of bulk/reference density, or    minimum density (absolute value)
                                1% of matrix if porous, or          allowed for material
                                0. for gases

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

        eos.rho_limit       float   0.9                 for expanded states with rho/rho_0 < rho_limit
                                                        the pressure is set to zero (if e < till_E_cv)

        eos.cs_limit        float   1% of approx.       lower limit for sound speed (in m/s),
                                    bulk sound speed    default is 1% of sqrt(till_A/till_rho_0)

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

        eos.cohesion                    float   0.
        eos.friction_angle              float   0.          in rad
        eos.cohesion_damaged            float   0.
        eos.friction_angle_damaged      float   0.          in rad
        eos.yield_stress                float   0.

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

This is a linear dependence of the yield strength on pressure.

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

This is a linear dependence of the yield strength on pressure.

You need to set `MOHR_COULOMB_PLASTICITY` in parameter.h (at compile time), and in the material config file:

        Key                     Type    Default     Details
        ___________________________________________________

        eos.cohesion            float   0.
        eos.friction_angle      float   0.          in rad

        eos.yield_stress        float   0.          optional upper limit for the yield strength,
                                                    where you have to additionally set
                                                    VON_MISES_PLASTICITY in parameter.h for this

--------------------------------

**Fragmentation model: Grady-Kipp**

To activate the Grady-Kipp model set `FRAGMENTATION` in parameter.h (at compile time). This is currently
only possible globally for all materials at once.
The following parameters are currently only needed for some pre-processing tools.

        Key                         Type    Default     Details
        _______________________________________________________

        fragmentation.weibull_k     float   none        not processed by miluphcuda
        fragmentation.weibull_m     float   none        not processed by miluphcuda

--------------------------------

**Porosity model: P-alpha**

The P-alpha porosity model can be used together with several EoS, which all have a distinct *eos.type*
when used together with the P-alpha model:

* *eos.type = 5* ... P-alpha + Tillotson EoS
* *eos.type = 6* ... P-alpha + Murnaghan EoS
* *eos.type = 13* ... P-alpha + ANEOS

In addition to the parameters for the respective EoS, you need the following:

        Key                 Type    Default             Details
        _______________________________________________________

        ...
        ...
        ...

        eos.cs_porous       float   50% of approx.      sound speed in uncompacted material (in m/s),
                                    bulk sound speed    the actual sound speed is interpolation between
                                                        this and the matrix cs, based on distention


