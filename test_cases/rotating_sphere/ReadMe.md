### Rotating sphere test case

Standard SPH cannot handle rigid rotations.
This test case is one of the standard tests to check if the TENSORIAL_CORRECTION 
implementation works.

How to run it:
1. copy parameter.h to the root source directory of miluphcuda (usually cp parameter.h ../../)
2. compile 
3. copy the binary to this directory
4. start run.sh in this directory

The initial particle distribution is a solid sphere with R=1, rotating with omega_z=2 pi /100 c_s/R, where c_s denotes the
sound speed, which is set to 1.

The test runs only for some minutes. In the end, either check the file **conserved_quantities.log** for the total
angular momentum (column 1 is time, column 12 is the absolute value of the total angular momentum of all particles), or
take a look at **angular_momentum.png**.
