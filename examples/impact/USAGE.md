Impact example for miluphcuda
---------------------------------------

Christoph Burger  
christoph.burger@uni-tuebingen.de

last updated: 25/Apr/2021

-----------------------------------------

This example represent an impact scenario of a basalt projectile onto a basalt, half-sphere target.

The spherical projectile has a radius of 0.5m, impacts with 6km/s, at an impact angle (to the vertical) of 30 deg.

All material is modeled as a porous solid, with a porosity of 20% (i.e., an initial distention of 1.25).

Applied rheology models:

* p-alpha porosity model
* Collins model for plastic flow
* Grady-Kipp model for fragmentation
* Tillotson equation of state

A zero-gravity environment is assumed.

The scenario uses ~50k SPH particles, with a runtime on the order of several hours on most current GPUs (benchmarked on a GTX 970).

-----------------------------------------

**To run the example:**

1. Compile miluphcuda using the `parameter.h` file from this directory.  
   Don't forget to also adapt the miluphcuda Makefile to your system.
2. Unpack `impact.0000.gz`.
3. Adapt the start script `run.sh` to your system (path to CUDA libs and to miluphcuda executable) and execute it.
4. Wait for the simulation to finish (200 output files).
   Output to stdout and stderr is written to `miluphcuda.output` and `miluphcuda.error`, respectively.

-----------------------------------------

**Check/visualize the results**

* You can visualize the simulation for example with *Paraview*. Find the latest release at https://www.paraview.org/.  
  
  First, enter the simulation directory and run

        utils/postprocessing/create_xdmf.py
  which creates an xdmf file (*paraview.xdmf* by default). You can use the default cmd-line options,
  so no need to set anything explicitly. Then start Paraview and either

    * directly open the created paraview.xdmf and choose settings yourself
    * load the prepared Paraview state in *paraview.pvsm* (*File -> Load State*), and select
      the created `paraview.xdmf` file under *Choose File Names*

* Compare your results (e.g., visualized with Paraview) to the animations in the `results/` directories,
  which show the interior structure (cut view) during the impact for various quantities.

-----------------------------------------

**Where to go from here?**

You can easily build on this example for setting up your own simulations. It is straight-forward to run different
scenarios, with varying impactor masses, impact parameters, and even material compositions and parameters.

We use an external tool for creating the required initial conditions, which is easy to use.
Just drop me an e-mail if you are interested.

Enjoy!

