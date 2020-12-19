Quick Start Guide
-----------------

fast_identify_fragments
-----------------------

and

calc_aggregates
---------------
---------------

last updated: 26/Nov/2020

Christoph Burger  
christoph.burger@uni-tuebingen.de

---------------------------------


Purpose
-------

The toolchain consisting of `fast_identify_fragments` and `calc_aggregates`
is intended for postprocessing results of `miluphcuda` SPH simulations.
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
_Fragments_ are defined as clumps of particles separated by up to a smoothing length.

* Both, HDF5 and ASCII miluphcuda output files are supported.
* Currently only constant smoothing length is supported, which is read directly from the miluphcuda output file.

The tool is based on building and searching an octree with runtimes on the order of seconds, even for large particle numbers.

Refer to

        ./fast_identify_fragments -?
for details on the usage options.


calc_aggregates
---------------
Based on the output of `fast_identify_fragments` (typically a `*.frag` file),
this tool computes gravitationally bound collections of _fragments_,
referred to as _aggregates_.

The final mass, composition and kinetics of the largest and optionally also the second-largest aggregate(s)
can be computed via an iterative solver.
Two, three, or four different materials are currently supported.

Refer to

        ./calc_aggregates -?
for details on the usage options.
