Quick Start Guide
-----------------
-----------------

fast_identify_fragments
-----------------------

and

calc_aggregates
---------------
---------------

last updated: 31/Jan/2021

Christoph Burger  
christoph.burger@uni-tuebingen.de

---------------------------------


Purpose
-------

The toolchain consisting of `fast_identify_fragments` and `calc_aggregates`
is intended for postprocessing miluphcuda` SPH simulations.
It allows to identify spatially connected clumps of SPH particles (_fragments_),
and subsequently find gravitationally bound collections of those fragments (_aggregates_).


Compilation and dependencies
----------------------------
Use the Makefile to build the executables. Type simply

        make
to build both tools, or use

        make fast_identify_fragments
or

        make calc_aggregates
to build either of them individually.

**HDF5 libraries**  
For building `fast_identify_fragments` (not for `calc_aggregates`) the HDF5 dev libraries are required.


fast_identify_fragments
-----------------------
This tool identifies _fragments_ in a miluphcuda output file.
Fragments are defined as clumps of particles separated by up to a smoothing length (sml).

* HDF5 and ASCII miluphcuda output files are supported
* currently only constant sml is supported, which is read from the miluphcuda output file

The tool is based on building and searching an octree with runtimes on the order of seconds, even for large particle numbers.

Check

        ./fast_identify_fragments -?
for details on usage options.


calc_aggregates
---------------
Based on the output of `fast_identify_fragments` (typically a .frag file),
this tool computes gravitationally bound collections of fragments, referred to as _aggregates_.

The final mass, composition and kinetics of the largest + optionally the second-largest aggregate(s)
can be computed via an iterative solver.
Two, three, or four different materials are currently supported.

Check

        ./calc_aggregates -?
for details on usage options.
