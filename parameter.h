/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com and Thomas I. Maindl
 *
 * @section     LICENSE
 * Copyright (c) 2019 Christoph Schaefer
 *
 * This file is part of miluphcuda.
 *
 * miluphcuda is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * miluphcuda is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef _PARAMETER_H
#define _PARAMETER_H

// Dimension of the problem
#define DIM 3

// Basic physical model, choose one of the following:
// SOLID solves continuum mechanics with material strength, and stress tensor \sigma^{\alpha \beta} = -p \delta^{\alpha \beta} + S^{\alpha \beta}
// HYDRO solves only the Euler equation, and there is only (scalar) pressure
#define SOLID 1
#define HYDRO 0
// set additionally p to 0 if p < 0
#define REAL_HYDRO 0

// add additional point masses to the simulation, read from file <filename>.mass
// format is location velocities mass r_min r_max, where location and velocities are vectors with size DIM and
// r_min/r_max are min/max distances of sph particles to the bodies before they are taken out of the simulation
#define GRAVITATING_POINT_MASSES 0

// sink particles (set point masses to be sink particles)
#define PARTICLE_ACCRETION 0 // check if particle is bound to one of the sink particles (crossed the accretion radius, rmin); if also UPDATE_SINK_VALUES 1: particle is accreted and ignored afterwards, else: continues orbiting without being accreted
#define UPDATE_SINK_VALUES 0 // add to sink the quantities of the accreted particle: mass, velocity and COM

// integrate the energy equation
// when setting up a SOLID simulation with Tillotson or ANEOS, it must be set to 1
#define INTEGRATE_ENERGY 1

// integrate the continuity equation
// if set to 0, the density will be calculated using the standard SPH sum \sum_i m_j W_ij
#define INTEGRATE_DENSITY 1

// adds viscosity to the Euler equation
#define NAVIER_STOKES 0
// choose between two different viscosity models
#define SHAKURA_SUNYAEV_ALPHA 0
#define CONSTANT_KINEMATIC_VISCOSITY 0
// artificial bulk viscosity according to Schaefer et al. (2004)
#define KLEY_VISCOSITY 0

// This is the damage model following Benz & Asphaug (1995). Set FRAGMENTATION to activate it.
// The damage acts always on pressure, but only on deviator stresses if DAMAGE_ACTS_ON_S is
// activated too, which is an important switch depending on the plasticity model (see comments there).
// Note: The damage model needs distribution of activation thresholds in the input file.
#define FRAGMENTATION 1
#define DAMAGE_ACTS_ON_S 0

// Choose the SPH representation to solve the momentum and energy equation:
// SPH_EQU_VERSION 1: original version with HYDRO dv_a/dt ~ - (p_a/rho_a**2 + p_b/rho_b**2)  \nabla_a W_ab
//                                     SOLID dv_a/dt ~ (sigma_a/rho_a**2 + sigma_b/rho_b**2) \nabla_a W_ab
// SPH_EQU_VERSION 2: slighty different version with
//                                     HYDRO dv_a/dt ~ - (p_a+p_b)/(rho_a*rho_b)  \nabla_a W_ab
//                                     SOLID dv_a/dt ~ (sigma_a+sigma_b)/(rho_a*rho_b)  \nabla_a W_ab
// If you do not know what to do, choose SPH_EQU_VERSION 1.
#define SPH_EQU_VERSION 1

// for the tensile instability fix
// you do not need this
#define ARTIFICIAL_STRESS 0

// standard SPH alpha/beta viscosity
#define ARTIFICIAL_VISCOSITY 1
// Balsara switch: lowers the artificial viscosity in regions without shocks
#define BALSARA_SWITCH 0

// INVISCID SPH (see Cullen & Dehnen paper)
#define INVISCID_SPH 0

// consistency switches
// for zeroth order consistency
#define SHEPARD_CORRECTION 0
// for linear consistency
// add tensorial correction tensor to dSdt calculation -> better conservation of angular momentum
#define TENSORIAL_CORRECTION 1

// Available plastic flow conditions:
// (if you do not know what this is, choose (1) or nothing)
//   (1) Simple von Mises plasticity with a constant yield strength:
#define VON_MISES_PLASTICITY 0
//   (2) Drucker-Prager (DP) yield criterion -> yield strength is given by the condition \sqrt(J_2) + A * I_1 + B = 0
//       with I_1: first invariant of stress tensor
//          J_2: second invariant of stress tensor
//          A, B: DP constants, which are calculated from angle of internal friction and cohesion
#define DRUCKER_PRAGER_PLASTICITY 0
//   (3) Mohr-Coulomb (MC) yield criterion
//       -> yield strength is given by yield_stress = tan(friction_angle) \times pressure + cohesion 
#define MOHR_COULOMB_PLASTICITY 0
//       Note: DP and MC are intended for granular-like materials, therefore the yield strength simply decreases (linearly) to zero for p<0.
//       Note: For DP and MC you can additionally choose (1) to impose an upper limit for the yield stress.
//   (4) Pressure dependent yield strength following Collins et al. (2004) and the implementation in Jutzi (2015)
//       -> yield strength is different for damaged (Y_d) and intact material (Y_i), and averaged mean (Y) in between:
//              Y_i = cohesion + \mu P / (1 + \mu P / (yield_stress - cohesion) )
//          where *cohesion* is the yield strength for P=0 and *yield_stress* the asymptotic limit for P=\infty
//          \mu is the coefficient of internal friction (= tan(friction_angle))
//              Y_d = cohesion_damaged + \mu_d \times P
//          where \mu_d is the coefficient of friction of the *damaged* material
//              Y = (1-damage)*Y_i + damage*Y_d
//              Y is limited to <= Y_i
//       Note: If FRAGMENTATION is not activated only Y_i is used.
//             DAMAGE_ACTS_ON_S is not allowed for this model, since the limiting of S already depends on damage.
//       If you want to additionally model the influence of some (single) melt energy on the yield strength, then activate
//       COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY, which adds a factor (1-e/e_melt) to the yield strength.
#define COLLINS_PLASTICITY 1
#define COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY 0
//   (5) Simplified version of the Collins et al. (2004) model, which uses only the
//       strength representation for intact material (Y_i), irrespective of damage.
//       Unlike in (4), Y decreases to zero (following a linear yield strength curve) for p<0.
//       In addition, negative pressures are limited to the pressure corresponding to
//       yield strength = 0 (i.e., are set to this value when they get more negative).
#define COLLINS_PLASTICITY_SIMPLE 0
// Note: The deviator stress tensor is additionally reduced by FRAGMENTATION (i.e., damage) only if
//       DAMAGE_ACTS_ON_S is set. For most plasticity models it depends on the use case whether this
//       is desired, only for COLLINS_PLASTICITY it is not reasonable (and therefore not allowed).

// model regolith as viscous fluid -> experimental setup, only for powerusers
#define VISCOUS_REGOLITH 0
// use Bui model for regolith -> experimental setup, only for powerusers
#define PURE_REGOLITH 0
// use Johnson-Cook plasticity model -> experimental setup, only for powerusers
#define JC_PLASTICITY 0

// Porosity models:
// p-alpha model implemented following Jutzi (200x)
#define PALPHA_POROSITY 1          // pressure depends on distention
#define STRESS_PALPHA_POROSITY 1 // deviatoric stress is also affected by distention
// Sirono model modified by Geretshauser (2009/10)
#define SIRONO_POROSITY 0
// Epsilon-Alpha model implemented following Wuennemann
#define EPSALPHA_POROSITY 0

// max number of activation thresholds per particle, only required for FRAGMENTATION, otherwise set to 1
#define MAX_NUM_FLAWS 32
// maximum number of interactions per particle -> fixed array size
#define MAX_NUM_INTERACTIONS 512

// if set to 1, the smoothing length is not fixed for each material type
// choose either FIXED_NOI for a fixed number of interaction partners following
// the ansatz by Hernquist and Katz
// or choose INTEGRATE_SML if you want to additionally integrate an ODE for
// the sml following the ansatz by Benz and integrate the ODE for the smoothing length
// d sml / dt  = sml/DIM * 1/rho  \nabla velocity
// if you want to specify an individual initial smoothing length for each particle (instead of the material
// specific one in material.cfg) in the initial particle file, set READ_INITIAL_SML_FROM_PARTICLE_FILE to 1
#define VARIABLE_SML 1
#define FIXED_NOI 0
#define INTEGRATE_SML 1
#define READ_INITIAL_SML_FROM_PARTICLE_FILE 1

// correction terms for sml calculation: adds gradient of the smoothing length to continuity equation, equation of motion, internal energy equation
#define SML_CORRECTION 0

// if set to 0, h = (h_i + h_j)/2  is used to calculate W_ij
// if set to 1, W_ij = ( W(h_i) + W(h_j) ) / 2
#define AVERAGE_KERNELS 0


// important switch: if the simulations yields at some point too many interactions for
// one particle (given by MAX_NUM_INTERACTIONS), then its smoothing length will be set to 0
// and the simulation continues. It will be announced on *stdout* when this happens
// if set to 0, the simulation stops in such a case unless DEAL_WITH_TOO_MANY_INTERACTIONS is used
#define TOO_MANY_INTERACTIONS_KILL_PARTICLE 0
// important switch: if the simulations yields at some point too many interactions for
// one particle (given by MAX_NUM_INTERACTIONS), then its smoothing length will be lowered until
// the interactions are lower than MAX_NUM_INTERACTIONS
#define DEAL_WITH_TOO_MANY_INTERACTIONS 0

// additional smoothing of the velocity field
// hinders particle penetration
// see Morris and Monaghan 1984
#define XSPH 0

// boundaries EXPERIMENTAL, please do not use this....
#define BOUNDARY_PARTICLE_ID -1
#define GHOST_BOUNDARIES 0
// note: see additionally boundaries.cu with functions beforeRHS and afterRHS for boundary conditions

// IO options
#define HDF5IO 1    // use HDF5 (needs libhdf5-dev and libhdf5)
#define MORE_OUTPUT 0   //produce additional output to HDF5 files (p_max, p_min, rho_max, rho_min); only ueful when HDF5IO is set
#define MORE_ANEOS_OUTPUT 0 // produce additional output to HDF5 files (T, cs, entropy, phase-flag); only useful when HDF5IO is set; set only if you use the ANEOS eos, but currently not supported for porosity+ANEOS
#define OUTPUT_GRAV_ENERGY 0    // compute and output gravitational energy (at times when output files are written); of all SPH particles (and also w.r.t. gravitating point masses and between them); direct particle-particle summation, not tree; option exists to control costly computation for high particle numbers
#define BINARY_INFO 0   // generates additional output file (binary_system.log) with info regarding binary system: semi-major axis, eccentricity if GRAVITATING_POINT_MASSES == 1

#endif
