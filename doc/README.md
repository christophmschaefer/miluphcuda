# miluphcuda documentation

This directory contains the documentation of miluphcuda. It describes the state of the branch `cms_devel`;
features that are not yet merged into `main` are marked as such.

The documentation is split into a practical part (Markdown, readable directly on GitHub) and a reference for the
physical model and its numerical implementation (LaTeX/PDF).

| Document | Contents |
|---|---|
| [installation.md](installation.md) | dependencies, compilation, `Makefile` settings, GPU architecture |
| [running.md](running.md) | how a simulation is set up and started, command-line options, integrators, kernels, input and output files, restarts, examples and test cases, pre- and postprocessing tools |
| [compile-time-options.md](compile-time-options.md) | all switches in `include/parameter.h`, the numerical settings in the other headers, and the combinations that are rejected at compile time |
| [../material-config/CREATE-MATERIAL-CONFIG.md](../material-config/CREATE-MATERIAL-CONFIG.md) | the material config file (`material.cfg`): all keys, defaults and the EOS types |
| [porosity.md](porosity.md) | practical guide to porous materials with the p-alpha model: switches, material parameters, initial conditions, time integration |
| [miluphcuda_documentation.pdf](miluphcuda_documentation.pdf) | physical model and numerics: governing equations, equations of state, strength, damage and porosity models, SPH discretization, tensorial correction, time integration, fast integration scheme, self-gravity |

The PDF is built from `miluphcuda_documentation.tex` and `miluphcuda_documentation.bib`:

    cd doc
    make

## Three places for settings

miluphcuda is configured in three places, and most questions about "why does my simulation do X" are answered by
looking at all three:

1. **Compile time**, `include/parameter.h` (physics and numerics for the whole simulation) and a few more headers for
   numerical details, see [compile-time-options.md](compile-time-options.md). You have to recompile after every change.
   `./miluphcuda -Y` prints the switches the binary was compiled with.
2. **Material config file**, typically `material.cfg`, read at run time (material parameters, EOS, per-material
   numerical settings such as the smoothing length), see
   [CREATE-MATERIAL-CONFIG.md](../material-config/CREATE-MATERIAL-CONFIG.md).
3. **Command line**, everything else (input file, integrator, output times, self-gravity, ...), see
   [running.md](running.md) and `./miluphcuda -h`.

## Citing

If you use miluphcuda, please cite

* Schäfer, C. M., Wandel, O. J., Burger, C., Maindl, T. I., Malamud, U., Buruchenko, S. K., Sfair, R., Audiffren,
  H., Vavilina, E., Winter, P. M. (2020): *A versatile smoothed particle hydrodynamics code for graphic cards*,
  Astronomy and Computing 33, 100410, [doi:10.1016/j.ascom.2020.100410](https://doi.org/10.1016/j.ascom.2020.100410)
* Schäfer, C., Riecker, S., Maindl, T. I., Speith, R., Scherrer, S., Kley, W. (2016): *A smooth particle
  hydrodynamics code to model collisions between solid, self-gravitating objects*, A&A 590, A19,
  [doi:10.1051/0004-6361/201528060](https://doi.org/10.1051/0004-6361/201528060)

and the ASCL record [ascl:1911.023](https://ascl.net/1911.023).

We do not support the use of the code for military purposes.
