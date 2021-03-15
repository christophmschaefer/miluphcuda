# miluphcuda

miluphcuda is a smoothed particle hydrodynamics (**SPH**)
code.

## Related pages

* [GitHub repository](https://github.com/christophmschaefer/miluphcuda)
* [miluphcuda documentation (pdf)](miluphcuda_documentation.pdf) mostly outdated :frown:

## Project structure

* **material_data**: config files for different materials
* **test_cases**: test cases for miluphcuda

## About

* miluphcuda is the cuda port of the original miluph code.
* miluphcuda can be used to model fluids and solids.
* miluphcuda runs on a single Nvidia GPU with compute capability 5.0 and higher.

Please see additional the following papers/links

* [A Smooth Particle Hydrodynamics Code to Model Collisions Between Solid, Self-Gravitating Objects](http://dx.doi.org/10.1051/0004-6361/201528060)
* [A versatile smoothed particle hydrodynamics code for graphic cards](https://www.sciencedirect.com/science/article/abs/pii/S2213133720300640?via%3Dihub)
* [ASCL record](https://ascl.net/1911.023)

Please cite one of these papers and the ASCL record if you use the code.


### Main features

* SPH hydro and solid
* self-gravity (via Barnes-Hut tree) or direct summation for a small number of particles
* porosity models (P-alpha, epsilon-alpha, Sirono)
* several equation of states (ideal gas, polytropic, Murnaghan, Tillotson, ANEOS and support for tabulated EOS)


## Developers
in somewhat arbitrary order

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
Maximilian Rutz,
Hugo Audiffren.
