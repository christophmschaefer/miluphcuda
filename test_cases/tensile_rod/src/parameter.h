#ifndef _PARAMETER_H
#define _PARAMETER_H

// Dimension of the problem
#define DIM 2

// physics

// integrate the energy equation
// integrate the continuity equation
// if set to 0, the density will be calculated using the standard SPH sum \sum_i m_j W_ij
#define INTEGRATE_ENERGY 0

// model solid bodies with stress tensor \sigma^{\alpha \beta} = -p \delta^{\alpha \beta} + S^{\alpha \beta}
// if set to 0, there is only pressure
#define INTEGRATE_DENSITY 1

// damage model following Benz & Asphaug 1995
// this needs some preprocessing of the initial particle distribution since activation thresholds 
// have to be distributed among the particles
#define SOLID 1
#define FRAGMENTATION 1
#define SYMMETRIC_STRESSTENSOR 0

// SPH stuff
// for the tensile instability fix
// you do not need this
#define ARTIFICIAL_STRESS 0

// standard SPH alpha/beta viscosity
// you need this
#define ARTIFICIAL_VISCOSITY 1

// for linear consistency 
// you need this
#define TENSORIAL_CORRECTION 0

// plastic flow conditions
// you can choose between
// 1 simple von Mises plasticity with a constant yield strength -> 
//          yield_stress =   in material.cfg file
// 2 Drucker Prager yield criterion -> yield strength is given by
//   the condition \sqrt(J_2) + A * I_1 + B = 0
//   with I_1: first invariant of stress tensor
//        J_2: second invariant of stress tensor
//        A, B: Drucker Prager constants
//              which are calculated from angle of internal friction and cohesion
//      in material.cfg: friction_angle = 
//                       cohesion = 
//  3 Mohr-Coulomb yield criterion -> yield strength is given by
//         yield_stress = tan(friction_angle) \times pressure + cohesion
//      in material.cfg: friction_angle = 
//                       cohesion = 
//  4 a pressure dependent yield strength following Gareth Collins' 2004 paper and
//   Martin Jutzi's implementation in his 2015 paper. 
//          yield_stress is different for damaged and intact rock
//      first, the yield stress for intact rock y_i is given by
//      y_i =  cohesion + \mu P / (1 + \mu P/(Y_m - cohesion) )
//      where yield_stress is the yield stress for P=0 and Y_m is the shear strength at P=\infty
//      \mu is the coefficient of internal friction
//      the yield strength for (fully) damaged rock y_d is given by
//      y_d = \mu_d \times P 
//      where \mu_d is the coefficient of friction of the *damaged* material
//      y_d is limited to y_d <= y_i
//      for this model, following parameters in material.cfg are obligatory
//          yield_stress = Y_M
//          cohesion = 
//          friction_angle = 
//  NOTE: units are: friction angle = rad
//                   cohesion = Pascal
//  if you do not know what this is, choose 1 or nothing

#define VON_MISES_PLASTICITY 1
//  WARNING: choose only one of the following three options
//  this will be fixed in a later version of the code
#define MOHR_COULOMB_PLASTICITY 0
#define DRUCKER_PRAGER_PLASTICITY 0
#define COLLINS_PRESSURE_DEPENDENT_YIELD_STRENGTH 0

#if MOHR_COULOMB_PLASTICITY && DRUCKER_PRAGER_PLASTICITY
#error choose only one of the three available plastic flow rules
#endif
#if MOHR_COULOMB_PLASTICITY && COLLINS_PRESSURE_DEPENDENT_YIELD_STRENGTH
#error choose only one of the three available plastic flow rules
#endif
#if DRUCKER_PRAGER_PLASTICITY && COLLINS_PRESSURE_DEPENDENT_YIELD_STRENGTH
#error choose only one of the three available plastic flow rules
#endif

// model regolith as viscous fluid -> experimental setup
#define VISCOUS_REGOLITH 0
// use Johnson-Cook plasticity model -> experimental
#define JC_PLASTICITY 0


// porosity models
// P-Alpha model implemented following Jutzi 200x
#define PALPHA_POROSITY 0

// constants
// maximum number of activation threshold per particle -> fixed array size, only needed for 
// FRAGMENTATION. if not used, set to 1
#define MAX_NUM_FLAWS 24
// maximum number of interactions per particle -> fixed array size
#define MAX_NUM_INTERACTIONS 150



// gravitational constant in SI
#define C_GRAVITY_SI 6.67259e-11
// gravitational constant in AU
#define C_GRAVITY_AU 3.96425141E-14

// if set to 1 and INTEGRATE_DENSITY is 1, the density will not be lower than 1% rho_0 from
// material.cfg
// note: see additionally boundaries.cu with functions beforeRHS and afterRHS for boundary conditions
#define DENSITY_FLOOR 0 // DENSITY FLOOR sets a minimum density for all particles. the floor density is 1% of the lowest density in material.cfg


// EoS stuff
// these are the implemented equation of states
// do nothing, no pressure whatsoever, ignore this particle
#define EOS_TYPE_IGNORE -1
// polytropic EOS for gas, needs polytropic_K and polytropic_gamma in material.cfg file
#define EOS_TYPE_POLYTROPIC_GAS 0
// Murnaghan EOS for solid bodies, see Melosh "Impact Cratering" for reference
// needs bulk_modulus, rho_0 and n in material.cfg
#define EOS_TYPE_MURNAGHAN 1
// Tillotson EOS for solid bodies, see Melosh "Impact Cratering" for reference
// needs alot of parameters in material.cfg:
// yield_stress till_rho_0 till_A till_B till_E_0 till_E_iv till_E_cv till_a till_b till_alpha till_beta
// bulk modulus and shear modulus are needed to calculate the sound speed and the crack growth speed
#define EOS_TYPE_TILLOTSON 2
// this is pure molecular hydrogen at 10 K 
#define EOS_TYPE_ISOTHERMAL_GAS 3
// The Bui et al. 2008 soil model
#define EOS_TYPE_REGOLITH 4
// Tillotson EOS with p-alpha model by Jutzi et al.
#define EOS_TYPE_JUTZI 5
// Murnaghan EOS with p-alpha model by Jutzi et al.
#define EOS_TYPE_JUTZI_MURNAGHAN 6
// ANEOS -> EXPERIMENTAL DO NOT USE 
#define EOS_TYPE_ANEOS 7
// describe regolith as a viscous material -> EXPERIMENTAL DO NOT USE
#define EOS_TYPE_VISCOUS_REGOLITH 8
// ideal gas equation, set polytropic_gamma in material.cfg
#define EOS_TYPE_IDEAL_GAS 9

// helping and debugging flag. set pressure to zero if it's tension
#define REAL_HYDRO 0 // set p = 0 if p < 0


// sph stuff
// use either BSPLINE (somewhat standard) kernel
// or WENDLAND (to avoid particle clumping)
// note: Wendland kernel is not implemented for DIM==1
#define USE_BSPLINE_KERNEL 1
#define USE_WENDLAND_KERNEL 0 


// if set to 1, the smoothing length is not fixed for each material type
// choose either FIXED_NOI for a fixed number of interaction partners following
// the ansatz by Hernquist and Katz
// or choose INTEGRATE_SML if you want to additionally integrate an ODE for
// the sml following the ansatz by Benz and integrate the ODE for the smoothing length
// d sml / dt  = sml/DIM * 1/rho  \nabla velocity
#define VARIABLE_SML 0
#define FIXED_NOI 0
#define INTEGRATE_SML 0


// important switch: if the simulations yields at some point too many interactions for
// one particle (given by MAX_NUM_INTERACTIONS), then its smoothing length will be set to 0
// and the simulation continues. It will be announced on *stdout* when this happens
// if set to 0, the simulation stops in such a case 
#define DEAL_WITH_TOO_MANY_INTERACTIONS 1

// additional smoothing of the velocity field
// hinders particle penetration
// see Morris and Monaghan 1984
#define XSPH 0

// parameters for the artificial stress -> normally not needed
// for test_cases/colliding rings, artificial stress prevents the
// tensile instability
// see Monaghan 2000 (SPH without a tensile instability) for details
#define EXPONENT_TENSOR 4
#define EPSILON_STRESS 0.2
#define MEAN_PARTICLE_DISTANCE 0.075 // für colliding rings
#define EPS_JACOBI 1e-10 // jacobiverfahren eigenwerte

// boundaries EXPERIMENTAL, please do not use this....
#define BOUNDARY_PARTICLE_ID -1 
#define GHOST_BOUNDARIES 0


// use HDF5 (needs libhdf5-dev and libhdf5)
#define HDF5IO 1






#if USE_BSPLINE_KERNEL && USE_WENDLAND_KERNEL
#error specifiy only one kernel
#endif

#if VISCOUS_REGOLITH && !SOLID
#error turn SOLID on when using VISCOUS_REGOLITH
#endif

#if VON_MISES_PLASTICITY && !SOLID
#error turn SOLID on when using VON_MISES_PLASTICITY
#endif

#if JC_PLASTICITY && !SOLID
#error turn SOLID on when using JC_PLASTICITY
#endif

#if FRAGMENTATION && !SOLID
#error turn SOLID on when using FRAGMENTATION
#endif

#if COHESION_FOR_DAMAGED_MATERIAL && !FRAGMENTATION
# error turn on FRAGMENTATION when using COHESION_FOR_DAMAGED_MATERIAL
#endif

#endif
