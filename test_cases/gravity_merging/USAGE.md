Gravity merging test case for miluphcuda
----------------------------------------

Christoph Burger  
christoph.burger@uni-tuebingen.de

last updated: 19/May/2021

-----------------------------------------

This is a test case of "gently" merging spheres, in a mass range at the transition of the strength and gravity regime.
The bodies have masses of 3e20 kg and 1e20 kg, respectively, and start from rest, directly next to each other.
The default settings are self-gravity + Collins strength model (including fragmentation with Grady-Kipp).

-----------------------------------------

**To run the test case:**

1. Compile miluphcuda with the `parameter.h` file from this directory.  
   Don't forget to also adapt the Makefile to your system.
2. Unpack `initials/impact.0000.gz` and move it to this directory.
3. Adapt the start script `run.sh` to your system (path to CUDA libs and to miluphcuda executable) and execute it.
4. Wait for the simulation to finish (75 output files).

The setup consists of ~50k SPH particles, with a runtime around 1h on most current GPUs (benchmarked on a GTX 970).

-----------------------------------------

**Check/visualize the results**

You can visualize the simulation for example with *Paraview*. Find the latest release at https://www.paraview.org/.  
First, enter the simulation directory and run

        utils/postprocessing/create_xdmf.py
which creates an xdmf file (*paraview.xdmf* by default). You can use the default cmd-line options,
so no need to set anything explicitly. Then start Paraview and either

* directly open the created paraview.xdmf and choose settings yourself
* load the prepared Paraview state in *results/paraview.pvsm* (*File -> Load State*), and select
  the created paraview.xdmf file under *Choose File Names*

* compare results (e.g., visualized with Paraview) to the animations in `results/`, which show
  the interior structure of the colliding bodies (cut views) for different quantities.

* You can also visualize the workings of the Collins plasticity model by running `results/plot_plastic_yielding.sh`.
  This produces plots for shear stress vs. pressure, including the respective yield limit curves.

