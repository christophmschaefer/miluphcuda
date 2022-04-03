Giant Collision examples for miluphcuda
---------------------------------------

Christoph Burger  
christoph.burger@uni-tuebingen.de

last updated: 15/Nov/2021

----------------------------------------------------------------------------------


These examples represent a collision of a ~1/10 Moon-mass body with a ~Moon-mass target.
Both bodies have a 30 mass% iron core and a 70 mass% rocky mantle.
The collision velocity is two times the mutual escape velocity (at the moment of first contact).
The impact angle is 45 deg.

You can choose between two material rheologies:

* *hydro* -- simple strengthless bodies
* *solid* -- includes solid-body physics, the Grady-Kipp fragmentation model, and the Collins plasticity model

The Tillotson EoS is applied to model the thermodynamic response.

The scenarios use ~60k SPH particles, with runtimes around 1h for *hydro*,
and 2h for *solid* on most current GPUs (benchmarked on a GTX 970).

----------------------------------------------------------------------------------


**To run the examples:**

1. Compile miluphcuda using the `parameter.h` file from the respective directory (hydro or solid).  
   Don't forget to also adapt the miluphcuda Makefile to your system.
2. Unpack `impact.0000.gz`.
3. Adapt the start script `run.sh` to your system (path to CUDA libs and to miluphcuda executable) and execute it.
4. Wait for the simulation to finish (75 output files).
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
      and select the created `paraview.xdmf` file under *Choose File Names*
        (note: if your Paraview version is not compatible with the state file version (check the first
        few lines of paraview.pvsm), either try a closer Paraview version, or load paraview.xdmf directly)

* Compare your results (e.g., visualized with Paraview) to the animations in `expected-results/`,
  which show the interior structure of the colliding bodies (cut views) for various quantities.

* For the *solid* example, you can also visualize the workings of the Collins plasticity model by
  running `analyze-results/plot_plastic_yielding.sh`. This produces plots for shear stress vs. pressure,
  including the respective yield limit curves.

----------------------------------------------------------------------------------


**Where to go from here?**

You can easily build on those examples for setting up your own simulations.
We use an external tool for creating the required initial conditions (i.e., `impact.0000`), which you can use to run
different scenarios, with varying masses, collision parameters, and even material compositions, and pre-collision
rotation. Just drop me an e-mail if you are interested.

Take a look at the timestep statistics at the very bottom of *miluphcuda.output*. If you are not satisfied you may try
to adjust the integrator accuracy on the cmd-line and/or the compile-time settings for the integrator in *rk2adaptive.h*.

You can also run further postprocessing on the results, e.g., to find all collision fragments and
gravitationally bound aggregates of those. The necessary tools are included in

        miluphcuda/utils/postprocessing/fast_identify_fragments_and_calc_aggregates/

Check the QuickStart guide there for more details.

Enjoy!

