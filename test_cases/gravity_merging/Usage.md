Gravity merging test case for miluphcuda
----------------------------------------

Christoph Burger  
christoph.burger@uni-tuebingen.de

last updated: 23/Feb/2021

-----------------------------------------

This is a test case of gently merging spheres, in a mass range at the interplay of strength and gravity.
The bodies have masses of 3e20 kg and 1e20 kg, respectively, and start from rest, just next to each other.
The default settings are self-gravity + Collins strength model (including fragmentation).

-----------------------------------------

**To run the test case:**

1. Compile miluphcuda using the `parameter.h` file from this directory.  
   Don't forget to also adapt the Makefile to your system.
2. Unpack `impact.0000.gz`.
3. Adapt the start script `run.sh` to your system (path to CUDA libs and to miluphcuda executable) and execute it.
4. Wait for the simulation to finish (75 output files).
   Output to stdout and stderr is written to `miluphcuda.output` and `miluphcuda.error`, respectively.

-----------------------------------------

**Check/visualize the results**

You can visualize the simulation for example with *Paraview*. Find the latest release at https://www.paraview.org/.  
First, enter the simulation directory and run

        utils/postprocessing/create_xdmf.py
which creates an xdmf file (*paraview.xdmf* by default). You can use the default cmd-line options,
so no need to set anything explicitly. Then start Paraview and either

* directly open the created paraview.xdmf and choose settings yourself
* load the prepared Paraview state in *paraview.pvsm* (*File -> Load State*), and select
  the created paraview.xdmf file under *Choose File Names*

