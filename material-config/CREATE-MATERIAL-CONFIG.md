How to set up the material config file for miluphcuda
=====================================================

last updated: 12/Feb/2021

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

The basic file structure, containing *n* different materials, looks like:

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

The available/required settings, depending on the used material model(s), are listed and described below.

The `material.cfg` file in this directory contains an example setup, comprising several materials and material models.

Note that, for now, the `eos = {...}` group regularly contains not only equation-of-state-specific entries,
but also further parameters related to the material model.

--------------------------------

Where to find material parameters?
----------------------------------

* The `material_data/` directory contains parameters for various materials.

* You can also check the `test_cases/` directory for example material setups.

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

* If values are ommitted, default values are assumed (see below). Normally, these default values are not what you want.

--------------------------------

Available settings
------------------

Here, all settings are described, ordered mainly by material model and/or equation of state.
For the keys in the following lists, the top level is assumed to be a specific material and
subgroups within materials are denoted by dot syntax. For example, *ID* is on the top level
of some material, and *eos.type* is the key *type* in the subgroup *eos*.

This list is currently not exhaustive.


**Always mandatory**:

        Key     Type    Default     Details

        ID      int     none        material ID, has to fit SPH particles' material-type
                                    in input file, have to be 0, 1, 2,...

**Always optional**:

        Key     Type    Default     Details

        name    str     none        name of material, currently not processed by miluphcuda


