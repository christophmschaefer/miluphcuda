# Miluphcuda

Miluphcuda is a smoothed particle hydrodynamics (**SPH**) 
code.

## Related pages

* [GitHub repository](https://github.com/christophmschaefer/miluphcuda)
* [Miluphcuda documentation (pdf)](miluphcuda_documentation.pdf)

## Project structure

* **material_data**: config files for different materials
* **test_cases**: test cases for miluphcuda

## About

* miluphcuda is the cuda port of the original miluph code.
* miluphcuda can be used to model fluids and solids. 
* miluphcuda runs on single Nvidia GPUs with compute capability 5.0 and higher.


### Main features

* SPH hydro and solid
* self-gravity (via Barnes-Hut tree)
* porosity models (P-alpha, epsilon-alpha, Sirono)


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
