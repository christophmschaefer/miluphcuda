# miluphcuda

miluphcuda is a smoothed particle hydrodynamics code

Check out the [documentation](https://christophmschaefer.github.io/miluphcuda/index.html)!

* miluphcuda is the cuda port of the original miluph code.
* miluphcuda can be used to model fluids and solids.
* miluphcuda runs on single Nvidia GPUs with compute capability 5.0 and higher.


## Main features

* SPH hydro and solid
* self-gravity (via Barnes-Hut tree)
* porosity models (P-alpha, epsilon-alpha, Sirono)

## Hardware requirements

miluphcuda runs on single NVIDIA GPUs using CUDA.

## Developers

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

The code is listed in the [ACSL](https://ascl.net/1911.023).

## Literature

For more information, please consider reading the following publications

 * [A versatile smoothed particle hydrodynamics code for graphic cards](https://ui.adsabs.harvard.edu/link_gateway/2020A&C....3300410S/doi:10.1016/j.ascom.2020.100410)
 * [A smooth particle hydrodynamics code to model collisions between solid, self-gravitating objects](https://ui.adsabs.harvard.edu/link_gateway/2016A&A...590A..19S/doi:10.1051/0004-6361/201528060)

