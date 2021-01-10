How to set up the material config file for miluphcuda
=====================================================

last updated: 08/Jan/2021

Christoph Burger  
christoph.burger@uni-tuebingen.de

--------------------------------

Purpose
-------

`miluphcuda` supports simulations comprising multiple materials, and multiple rheologies.
Many important settings for the material model and numerics, and especially material parameters,
are provided in a structured config file, processed by the _libconfig_ library.

Check http://hyperrealm.github.io/libconfig/ for the latest version.

Note: Important settings for `miluphcuda` are also set via cmd-line flags, and at compile time in `parameter.h`.

--------------------------------

File structure
--------------

The config file (typically named `material.cfg`) consists of key-value pairs,
organized in groups for different materials, equation of state, etc.

The basic structure of the material config file, containing `n` different materials, looks like:

        materials =
        (
            {
                ID = 0
                name = ...
                ...
            },
            {
                ID = 1
                name = ...
                ...
            },
            .
            .
            .
            {
                ID = n-1
                name = ...
                ...
            }
        );

The `material.cfg` file in this directory contains an example setup, comprising several materials and material models.

--------------------------------

Where to find material parameters?
----------------------------------

* The directory `material_data/` contains parameters for various materials.

* You can also check the `test_cases/` for example material setups.

--------------------------------

Good to know...
----------------------------------

The entries in the config file have to match their data type. For example, the ID tag has
to be an integer value whereas the value for the cohesion has to be given in floating point
notation:

    cohesion = 1000

will not work but

    cohesion = 1000.0

or

    cohesion = 1e3

are valid entries.

The code prints out all read material parameters if the verbose flag `-v` is used.
If values are ommitted default values are assumed. Normally, these default values are not what you
want :ghost:.




