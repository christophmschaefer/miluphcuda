Impact example for miluphcuda
---------------------------------------

Christoph Burger  
christoph.burger@uni-tuebingen.de

last updated: 15/Nov/2021

----------------------------------------------------------------------------------


This example represent an impact scenario of a basalt projectile onto a basalt, half-sphere target.

The spherical projectile has a radius of 0.5m, impacts with 6km/s, at an impact angle (to the vertical) of 30 deg.

All material is modeled as a porous solid, with a porosity of 20% (i.e., an initial distention of 1.25).

Applied rheology models:

* p-alpha porosity model
* Collins model for plastic flow
* Grady-Kipp model for fragmentation
* Tillotson equation of state

A zero-gravity environment is assumed.

The scenario uses ~60k SPH particles, with a runtime on the order of one hour on most current GPUs (benchmarked on a GTX 970).

----------------------------------------------------------------------------------


**To run the example:**

1. Compile miluphcuda using the `parameter.h` file from this directory.  
   Don't forget to also adapt the miluphcuda Makefile to your system.
2. Unpack `impact.0000.gz` to this directory.
3. Adapt the start script `run.sh` to your system (path to CUDA libs and to miluphcuda executable) and execute it.
4. Wait for the simulation to finish (200 output files).
   Output to stdout and stderr is written to `miluphcuda.output` and `miluphcuda.error`, respectively.

----------------------------------------------------------------------------------


**Check/visualize the results**

* You can visualize the simulation for example with *Paraview*. Find the latest release at https://www.paraview.org/.  
  
  First, enter the simulation directory and run

        utils/postprocessing/create_xdmf.py

  (without any arguments) which creates an xdmf file (*paraview.xdmf* by default). You can use the default cmd-line
  options, so no need to set anything explicitly. Note that the xdmf file contains only metadata, i.e., you still
  need the .h5 files it points to in the same directory.  
  Then start Paraview and either

    * directly open the created xdmf file and choose settings yourself
    * load the prepared Paraview state in `analyze-results/paraview.pvsm` (*File -> Load State*),
      and select the created paraview.xdmf file under *Choose File Names*
      note that this shows you a slice of the target by default, for better visibility of the cratering process
        (note: if your Paraview version is not compatible with the state file version (check the first
        few lines of paraview.pvsm), either try a closer Paraview version, or load paraview.xdmf directly)

* Compare your results (e.g., visualized with Paraview) to the animations in `expected-results/animations/`,
  which show the cratering process (a sclice of the target) for various quantities.

* A full view of the (half-sphere) target and the impact process is visualized in `expected-results/animations/full.mp4`.

* An important aspect of the porosity model is that the *crush curve* (the curve distention vs. pressure which is
  followed during compaction) is modeled correctly. Since the pressures that lead to compaction are typically released
  again quickly, the SPH particles will eventually lie either on or below the crush curve -- if the model is accurate.  
  You can check this by running for example

        analyze-results/plot_p_alpha_convergence.py -v --files impact.000[1-4].h5

  which plots SPH particles + the theoretical crush curve for output frames 1 through 4. Take a look at
  `expected-results/p_vs_alpha.first4frames.png` to see what the plots should qualitatively look like for the above four frames.
  You can of course specify any number/sequence of frames you are interested in for plotting.

* You can also visualize the workings of the Collins plasticity model by running `analyze-results/plot_plastic_yielding.sh`.
  This produces plots for shear stress vs. pressure, including the respective yield limit curves.

----------------------------------------------------------------------------------


**Where to go from here?**

You can easily build on this example for setting up your own simulations.
We use an external tool for creating the required initial conditions (i.e., `impact.0000`), which you can use to run
different scenarios, with varying impactor masses, impact parameters, and even material compositions and parameters.
Just drop me an e-mail if you are interested.

Take a look at the timestep statistics at the very bottom of *miluphcuda.output*. If you are not satisfied you may try
to adjust the integrator accuracy on the cmd-line and/or the compile-time settings for the integrator in *rk2adaptive.h*.

Enjoy!

