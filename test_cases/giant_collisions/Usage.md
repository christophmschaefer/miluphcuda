Giant Collision test cases for miluphcuda
-----------------------------------------

Christoph Burger  
christoph.burger@uni-tuebingen.de

last updated: 16/Feb/2021

-----------------------------------------

These test cases represent a collision of a ~1/10 Moon-mass body with a ~Moon-mass target.
Both bodies have a 30 mass% iron core and a 70 mass% rocky mantle.
The collision velocity is two times the mutual escape velocity (at the moment of first contact).
The impact angle is 45 deg.

You can choose between two material rheologies:

* *hydro* -- simple strengthless bodies
* *solid* -- includes solid-body physics, the Grady-Kipp fragmentation model, and the Collins plasticity model

The Tillotson EoS is applied to model the thermodynamic response.

The scenarios use ~60k SPH particles, with runtimes around 1.5h for *hydro*,
and 3h for *solid* on most current GPUs (benchmarked on a GTX 970).

-----------------------------------------

**To run the test cases:**

1. Compile miluphcuda using the `parameter.h` file from the respective directory (hydro or solid).  
   Don't forget to also adapt the Makefile to your system.
2. Unpack `impact.0000.gz`.
3. Adapt the start script `run.sh` to your system (path to CUDA libs and to miluphcuda executable) and execute it.
4. Wait for the simulation to finish (75 output files).
   Output to stdout and stderr is written to `miluphcuda.output` and `miluphcuda.error`, respectively.

-----------------------------------------

**Check/visualize the results**

* You can visualize the simulation for example with *Paraview*. Find the latest release at https://www.paraview.org/.  
  
  First, enter your simulation directory and run

        utils/postprocessing/create_xdmf.py
  which creates an xdmf file (`paraview.xdmf` by default). Then start Paraview and load the created xdmf file.
  You can use the default cmd-line options, so no need to set anything explicitly.

* Compare your results (e.g., visualized with Paraview) to the animations in the `results/` directories,
  which show the interior structure of the colliding bodies (cut views) for various quantities.

-----------------------------------------

**Where to go from here?**

You can easily build on those test cases once you got them running. It is straight-forward to run different scenarios,
with varying masses, collision parameters, and even material compositions or pre-collision rotation. We use an external
tool for creating the required initial conditions, which is easy to use. Just drop me an e-mail if you are interested.

You can also run further postprocessing on the results, e.g., to find all collision fragments and
gravitationally bound aggregates of those. The necessary tools are included in

        miluphcuda/utils/postprocessing/fast_identify_fragments_and_calc_aggregates/

Enjoy!

