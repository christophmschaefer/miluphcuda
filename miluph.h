/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com
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


#ifndef _MILUPH_H
#define _MILUPH_H

#define MILUPHCUDA_VERSION "devel"

// debug flags, mainly for additional output
#define DEBUG_TIMESTEP 1
#define DEBUG_LINALG 1
#define DEBUG_TREE 1
#define DEBUG_TREE_TO_FILE 0
#define DEBUG_GRAVITY 1
#define DEBUG_RHS 1
#define DEBUG_RHS_RUNTIMES 1
#define DEBUG_MISC 1
#define DEBUG_IO 0
#define DEBUG_DEVEL 1
// define if you want to pass around cudaVerify()
#undef NDEBUG  // NO DEBUG 


#define FALSE 0
#define TRUE 1

#include <math.h>
#include <errno.h>
#include <getopt.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <libconfig.h>
#include "cuda_utils.h"
#include "parameter.h"
#include "io.h"
#include "timeintegration.h"
#include "cuda_profiler_api.h"
#include "checks.h"

/// structure for sph particle
struct Particle {
    double *x0; ///< the initial x-coordinate at the start of the simulation
#if DIM > 1
    double *y0; ///< the initial y-coordinate at the start of the simulation
#if DIM > 2
    double *z0; ///< the initial z-coordinate at the start of the simulation
#endif
#endif
    double *x; ///< the x-coordinate of the sph particle
#if DIM > 1
    double *y; ///< the y-coordinate of the sph particle
#if DIM > 2
    double *z; ///< the z-coordinate of the sph particle
#endif
#endif
    double *vx0; ///< the initial velocity in x-direction at the start of the simulation
#if DIM > 1
    double *vy0; ///< the initial velocity in y-direction at the start of the simulation
#if DIM > 2
    double *vz0; ///< the initial velocity in z-direction at the start of the simulation
#endif
#endif
    double *dxdt; ///< the time derivative of the x-location. note that in sph, dx/dt != vx if XSPH is used
#if DIM > 1 
    double *dydt; ///< the time derivative of the y-location
#if DIM > 2
    double *dzdt; ///< the time derivative of the z-location
#endif
#endif
    double *vx; ///< the velocity in x-direction. if XSPH is 0, dx/dt \equiv vx
#if DIM > 1
    double *vy; ///< the velocity in y-direction
#if DIM > 2
    double *vz; ///< the velocity in z-direction
#endif
#endif
    double *ax; ///< the acceleration in x-direction
#if DIM > 1
    double *ay; ///< the acceleration in y-direction
#if DIM > 2
    double *az; ///< the acceleration in z-direction
#endif
#endif
    double *g_ax; ///< the acceleration due to self-gravity of the particles in x-direction
#if DIM > 1
    double *g_ay; ///< the acceleration due to self-gravity of the particles in y-direction
#if DIM > 2
    double *g_az; ///< the acceleration due to self-gravity of the particles in z-direction
#endif
#endif

// for tree change algorithm
    double *g_local_cellsize; ///< the size of the tree node in which the particle is located
    double *g_x; ///< the gridlength in x-direction of the tree node in which the particle is located
#if DIM > 1
    double *g_y; ///< the gridlength in y-direction of the tree node in which the particle is located
# if DIM > 2
    double *g_z; ///< the gridlength in z-direction of the tree node in which the particle is located
# endif
#endif

    double *m; ///< mass of the sph particle
    double *h; ///< smoothing length of the sph particle
    double *h0; ///< the initial smoothing length of the sph particle at start of the simulation
#if INTEGRATE_SML
    double *dhdt; ///< the time derivative of the smoothing length. only if INTEGRATE_SML is 1
#endif
    double *rho; ///< the density of the sph particle
    double *drhodt; ///< the time derivative of the density of the particle
    double *p; ///< the pressure of the particle
    double *e; ///< the specific internal energy of the particle

#if MORE_OUTPUT
    double *p_min; ///< the smallest pressure that the particle is exerted during the simulation
    double *p_max; ///< the highest pressure that the particle is exerted during the simulation
    double *rho_min; ///< the smallest density that the particle has during the simulation
    double *rho_max; ///< the highest density that the particle has during the simulation
    double *e_min; ///< the smallest specific internal energy of the particle during the simulation
    double *e_max; ///< the highest specific internal energy of the particle during the simulation
    double *cs_min; ///< the slowest sound speed of the particle during the simulation
    double *cs_max; ///< the highest sound speed of the particle during the simulation
#endif
#if INTEGRATE_ENERGY
    double *dedt; ///< the time derivative of the specific internal energy of the particle
#endif

#if NAVIER_STOKES
    double *Tshear; ///< the viscous stress tensor for the Navier-Stokes equation. this is the traceless tensor, given by \sigma = \eta T + \zeta \nabla \cdot \vec{v}, and since we do not store \nabla \cdot \vec{v} for each particle, we store it here
    double *eta; ///< the viscosity coefficient
#endif

#if SOLID
    double *S;  ///< the deviatoric stress tensor
    double *dSdt; ///< the time derivative of the deviatoric stress tensor
    double *local_strain; ///< the local strain of a sph particle as required for the Grady-Kipp fragmentation model
    double *ep; ///< the total strain of a sph particle
    double *edotp; ///< and its time derivative
    double *plastic_f; ///< the plasticity factor (reduce factor of elastic strain to plastic strain
    double *sigma; ///< the stress tensor, \sigma^{\alpha \beta} = -p\delta^{\alpha \beta} + S^{\alpha \beta}
#endif

#if ARTIFICIAL_STRESS
    double *R; ///< the artificial stress o fix the tensile instability following Monaghan 2000
#endif

/// experimental/outdated, do not use.... Johnson-Cook plasticity related
#if JC_PLASTICITY
    double *T;
    double *dTdt;
    double *jc_f;
#endif

    double *xsphvx; ///< the velocity in x-direction if XSPH is used
#if DIM > 1
    double *xsphvy; ///< the velocity in y-direction if XSPH is used
#if DIM > 2
    double *xsphvz; ///< the velocity in z-direction if XSPH is used
#endif
#endif

#if (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH || INTEGRATE_ENERGY)
    double *curlv; ///< \f$\nabla \times \vec{v}\f$
    double *divv; ///< \f$ \nabla \cdot \vec{v} \f$
#endif

#if FRAGMENTATION
    double *d;             ///< DIM-root of tensile damage
    double *damage_total;  ///< tensile damage + porous damage (directly, not DIM-root)
    double *dddt; ///< the time derivative of DIM-root of (tensile) damage
    int *numFlaws; ///< the total number of flaws
    int maxNumFlaws; ///< the maximum number of flaws allowed per particle
    int *numActiveFlaws; ///< the current number of activated flaws
    double *flaws; ///< the values for the strain for each flaw (array of size maxNumFlaws)
# if PALPHA_POROSITY
    double *damage_porjutzi;   ///< DIM-root of porous damage
    double *ddamage_porjutzidt; ///< time derivative of DIM-root of porous damage
# endif
#endif

#if ARTIFICIAL_VISCOSITY
    double *muijmax; ///< value to calculate the time step size for the integrator
#endif
#if INVISCID_SPH
    double *beta; ///< \beta from artificial viscosity, Dehnen ansatz for inviscid sph
    double *beta_old; 
    double *divv_old;
    double *dbetadt; ///< time derivative of artificial viscosity \beta, Dehnen ansatz for inviscid sph
#endif

#if PALPHA_POROSITY
    double *pold; ///< the pressure of the sph particle after the last successful timestep
    double *alpha_jutzi; ///< the current distension of the sph particle
    double *alpha_jutzi_old; ///< the distension of the sph particle after the last successful timestep
    double *dalphadt; ///< the time derivative of the distension
    double *dalphadp; ///< the partial derivative of the distension with respect to the pressure
    double *dp; ///< the difference in pressure from the last timestep to the current one
    double *dalphadrho; ///< the partial derivative of the distension with respect to the density
    double *f; ///< additional factor to reduce the deviatoric stress tensor according to Jutzi
    double *delpdelrho; ///< the partial derivative of the pressure with respect to the density
    double *delpdele; ///< the partial derivative of the pressure with respect to the specific internal energy
	double *cs_old; ///< the sound speed after the last successful timestep
#endif

#if SIRONO_POROSITY
    double *compressive_strength; ///< the current compressive strength
    double *tensile_strength; ///< the current tensile strength
    double *shear_strength; ///< the current shear strength
    double *K;  ///< the current bulk modulus
    double *rho_0prime;
    double *rho_c_plus;
    double *rho_c_minus;
    int *flag_rho_0prime;
    int *flag_plastic;
#endif

#if EPSALPHA_POROSITY
    double *alpha_epspor; ///< distention in the strain-\alpha model
    double *dalpha_epspordt; ///< time derivative of the distension
    double *epsilon_v; ///<  volume change (trace of strain rate tensor)
    double *depsilon_vdt; ///< time derivative of volume change
#endif

#if GHOST_BOUNDARIES
    int *real_partner; ///<  the corresponding real particle index of a ghost particle 
#endif

#if SHEPARD_CORRECTION
    double *shepard_correction; ///< the shepard correction factor for zeroth order consistency
#endif

#if TENSORIAL_CORRECTION
    double *tensorialCorrectionMatrix; ///< correction matrix for linear consistency
    double *tensorialCorrectiondWdrr; ///< correction factors for linear consistency
#endif
#if SML_CORRECTION
    double *sml_omega; 
#endif
    double *cs; ///< sound speed of sph particle
    int *noi; ///< number of interaction partners of sph particle
    int *materialId; ///< the current materialID of the sph particle
    int *materialId0; ///< the initial materialID of the sph particle
    int *depth; ///< the depth in the tree where the particle is located
    int *deactivate_me_flag; ///< flag to organise deactivation of particles after integration step
};  // end 'struct Particle'

/// struct for a gravitating pointmass
struct Pointmass {
    double *x; ///< x-coordinate of pointmass
#if DIM > 1
    double *y; ///< y-coordinate of pointmass
#if DIM > 2
    double *z; ///< z-coordinate of pointmass
#endif
#endif
    double *vx; ///< x-component of velocity of pointmass
#if DIM > 1
    double *vy; ///< y-component of velocity of pointmass
#if DIM > 2
    double *vz; ///< z-component of velocity of pointmass
#endif
#endif
    double *ax; ///< x-component of acceleration of pointmass
    double *feedback_ax; ///< x-component of acceleration due to gravitational interaction with all sph particles
#if DIM > 1
    double *ay; ///< y-component of acceleration of pointmass
    double *feedback_ay; ///< y-component of acceleration due to gravitational interaction with all sph particles
#if DIM > 2
    double *az; ///< z-component of acceleration of pointmass
    double *feedback_az; ///< z-component of acceleration due to gravitational interaction with all sph particles
#endif
#endif
    double *m; ///< mass of pointmass
    double *rmin; ///< minimum distance to pointmass before particle gets accreted
    double *rmax; ///< maximum distance to pointmass before particle gets deactivated

    int *feels_particles; ///< flag to activate feedback from sph particles on pointmass
};


/// the pointers to the arrays on the host
extern struct Pointmass pointmass_host;
/// the pointers to the arrays on the device in constant memory
extern __constant__ struct Pointmass pointmass;
/// the pointers to the arrays on the device residing on the host
extern struct Pointmass pointmass_device;
/// the pointers to the arrays for the runge-kutta integrator
extern struct Pointmass rk_pointmass_device[3];
extern __constant__ struct Pointmass rk_pointmass[3];
/// the pointers to the arrays for the runge-kutta 4 integrator
extern struct Pointmass rk4_pointmass_device[4];
extern __constant__ struct Pointmass rk4_pointmass[4];

extern struct Pointmass predictor_pointmass_device;
extern __constant__ struct Pointmass predictor_pointmass;
extern __constant__ struct Pointmass pointmass_rhs;

/// the pointers to the arrays on the host
extern struct Particle p_host;
/// the pointers to the arrays on the device in constant memory
extern __constant__ struct Particle p;
extern __constant__ struct Particle p_rhs;
/// the pointers to the arrays on the device residing on the host
extern struct Particle p_device;
/// the pointers to the arrays for the runge-kutta integrator
extern struct Particle rk_device[3];
extern __constant__ struct Particle rk[3];

extern struct Particle predictor_device;
extern __constant__ struct Particle predictor;


/// the three (four for rk4) integrator steps
enum {
    RKSTART,
    RKFIRST,
    RKSECOND,
    RKTHIRD
};


/// the implemented integrators
enum {
    EULER, ///< simple 1st order Euler integrator - use only for tests, no production runs, no science!
    RK2_ADAPTIVE, ////< the default embedded Runge Kutta 2/3 integrator with adaptive time step
    MONAGHAN_PC, ////< predictor corrector integrator with initial half step 
    EULER_PC, ///< predictor corrector integrator with initial full step
    HEUN_RK4 ///< fancy coupled Heun/RK4 integrator for use with sims with gravitating point masses. The point masses are integrated using the higher order rk4 and the hydro/solid aka sph part is done with standard Heun (aka Euler PC)
};


#if FRAGMENTATION
extern int maxNumFlaws_host; ///< maximum number of flaws
#endif

extern int *interactions; ///< the array that keeps track and stores all interactions between the particles at a certain timestep
extern int *interactions_host;

extern int *childList_host;
extern int *childListd;
extern __constant__ volatile int *childList;

extern int numberOfParticles;
extern int maxNumberOfParticles;
extern int numberOfRealParticles;
extern int numberOfPointmasses;

extern size_t memorySizeForPointmasses;
extern size_t integermemorySizeForPointmasses;
extern size_t memorySizeForParticles;
extern size_t memorySizeForTree;
extern size_t memorySizeForInteractions;
extern size_t memorySizeForChildren;
extern size_t memorySizeForStress;
#if FRAGMENTATION
extern size_t memorySizeForActivationThreshold;
#endif

extern int numberOfChildren; // 4 for 2D, 8 for 3D
// extern size_t numberOfNodes;
extern int numberOfNodes;

extern int restartedRun;
extern int numberOfMultiprocessors;

extern double treeTheta; ///< the Barnes-Hut tree theta parameter

/// some additional parameters useful to define
typedef struct RunParameter {
    int performanceTest; ///< not used
    int verbose; ///< flag if the user wants to be flooded with information
    int restart; ///< flag if the sim is restarted or new and fresh
    int selfgravity; ///< flag for self-gravity calculation or not
    int directselfgravity; ///< flag to use the N**2 algorithm to calculate self-gravity
    int decouplegravity; ///< flag to decouple the hydrotimestep from the calculation of self-gravity
    int hdf5output; ///< flag to write to HDF5
    int hdf5input; ///< flag to read from HDF5
    int ascii_output; ///< flag to write to ASCII
    int integrator_type; ///< enum of the integrator
    double maxtimestep; ///< max time step size. useful to set an upper limit to the time step
    double firsttimestep; ///< first time step that should be tried
    double angular_momentum_check; ///< check the conservation of angular momentum, quick implementation, do not use
    double rk_epsrel; ///< relative error for RK2_ADAPTIVE integrator, useful values are 1e-4 to 1e-6
    double boundary_ratio;
    char kernel[256];
    char conservedquantitiesfilename[255];
    char binarysystemfilename[256];
    config_t config;
} RunParameter;

extern RunParameter param;

#endif
