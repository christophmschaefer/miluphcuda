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

// particle structure for memory management
// on host and device
struct Particle {
    double *x0;
#if DIM > 1
    double *y0;
#if DIM > 2
    double *z0;
#endif
#endif
    double *x;
#if DIM > 1
    double *y;
#if DIM > 2
    double *z;
#endif
#endif
    double *vx0;
#if DIM > 1
    double *vy0;
#if DIM > 2
    double *vz0;
#endif
#endif
    double *dxdt;
#if DIM > 1
    double *dydt;
#if DIM > 2
    double *dzdt;
#endif
#endif
    double *vx;
#if DIM > 1
    double *vy;
#if DIM > 2
    double *vz;
#endif
#endif
    double *ax;
#if DIM > 1
    double *ay;
#if DIM > 2
    double *az;
#endif
#endif
    double *g_ax;
#if DIM > 1
    double *g_ay;
#if DIM > 2
    double *g_az;
#endif
#endif

// for tree change algorithm
    double *g_local_cellsize;
    double *g_x;
#if DIM > 1
    double *g_y;
# if DIM > 2
    double *g_z;
# endif
#endif

    double *m;
// the smoothing length
    double *h;
// the initial smoothing length
    double *h0;
#if INTEGRATE_SML
    double *dhdt;
#endif
    double *rho;
    double *drhodt;
    double *p;
    double *e;

#if MORE_OUTPUT
    double *p_min;
    double *p_max;
    double *rho_min;
    double *rho_max;
    double *e_min;
    double *e_max;
    double *cs_min;
    double *cs_max;
#endif
#if INTEGRATE_ENERGY
    double *dedt;
#endif

#if NAVIER_STOKES
    // the viscous shear tensor
    // note: this is the traceless tensor
    // the viscous tensor is given by sigma = eta T + zeta div v
    // and since we're storing div v for each particle, we do not store it here
    double *Tshear;
    double *eta;
#endif

#if SOLID
    //in case of soil S will be used as sigma
    double *S;
#if FRAGMENTATION
    double *Sreal;
#endif
    double *dSdt;
    double *local_strain;

    // *the* one and only stress tensor (note: not in the soil case, where S is the stress tensor)
    double *sigma;
#endif

#if ARTIFICIAL_STRESS
    double *R;
#endif

#if JC_PLASTICITY
    double *ep;
    double *edotp;
    double *T;
    double *dTdt;
    double *jc_f;
#endif

    double *xsphvx;
#if DIM > 1
    double *xsphvy;
#if DIM > 2
    double *xsphvz;
#endif
#endif

#if (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH || INTEGRATE_ENERGY)
    double *curlv;
    double *divv;
#endif

#if FRAGMENTATION
    double *d;
    double *damage_total; // tensile damage + porous damage
    double *dddt;
    int *numFlaws;
    int maxNumFlaws;
    int *numActiveFlaws;
    double *flaws;
#if PALPHA_POROSITY
    double *damage_porjutzi;
    double *ddamage_porjutzidt;
#endif
#endif

#if ARTIFICIAL_VISCOSITY
    double *muijmax;
#endif
#if INVISCID_SPH
    double *beta;
    double *beta_old;
    double *divv_old;
    double *dbetadt;
#endif

#if PALPHA_POROSITY
    double *pold;
    double *alpha_jutzi;
    double *alpha_jutzi_old;
    double *dalphadt;
    double *dalphadp;
    double *dp;
    double *dalphadrho;
    double *f;
    double *delpdelrho;
    double *delpdele;
	double *cs_old;
#endif

#if SIRONO_POROSITY
    double *compressive_strength;
    double *tensile_strength;
    double *shear_strength;
    double *K;
    double *rho_0prime;
    double *rho_c_plus;
    double *rho_c_minus;
    int *flag_rho_0prime;
    int *flag_plastic;
#endif

#if EPSALPHA_POROSITY
    double *alpha_epspor;
    double *dalpha_epspordt;
    double *epsilon_v;
    double *depsilon_vdt;
#endif

#if GHOST_BOUNDARIES
    /* the corresponding real particle index of a ghost particle */
    int *real_partner;
#endif

#if SHEPARD_CORRECTION
    double *shepard_correction;
#endif

#if TENSORIAL_CORRECTION
    double *tensorialCorrectionMatrix;
    double *tensorialCorrectiondWdrr;
#endif
#if SML_CORRECTION
    double *sml_omega;
#endif
    double *cs;
    int *noi;
    int *materialId;
    // the initial material id
    int *materialId0;
    int *depth;
};  // end 'struct Particle'


struct Pointmass {
    double *x;
#if DIM > 1
    double *y;
#if DIM > 2
    double *z;
#endif
#endif
    double *vx;
#if DIM > 1
    double *vy;
#if DIM > 2
    double *vz;
#endif
#endif
    double *ax;
    double *feedback_ax;
#if DIM > 1
    double *ay;
    double *feedback_ay;
#if DIM > 2
    double *az;
    double *feedback_az;
#endif
#endif
    double *m;
    double *rmin;
    double *rmax;

    int *feels_particles;
};


// the pointers to the arrays on the host
extern struct Pointmass pointmass_host;
// the pointers to the arrays on the device in constant memory
extern __constant__ struct Pointmass pointmass;
// the pointers to the arrays on the device residing on the host
extern struct Pointmass pointmass_device;
// the pointers to the arrays for the runge-kutta integrator
extern struct Pointmass rk_pointmass_device[3];
extern __constant__ struct Pointmass rk_pointmass[3];
// the pointers to the arrays for the runge-kutta 4 integrator
extern struct Pointmass rk4_pointmass_device[4];
extern __constant__ struct Pointmass rk4_pointmass[4];

extern struct Pointmass predictor_pointmass_device;
extern __constant__ struct Pointmass predictor_pointmass;
extern __constant__ struct Pointmass pointmass_rhs;

// the pointers to the arrays on the host
extern struct Particle p_host;
// the pointers to the arrays on the device in constant memory
extern __constant__ struct Particle p;
extern __constant__ struct Particle p_rhs;
// the pointers to the arrays on the device residing on the host
extern struct Particle p_device;
// the pointers to the arrays for the runge-kutta integrator
extern struct Particle rk_device[3];
extern __constant__ struct Particle rk[3];

extern struct Particle predictor_device;
extern __constant__ struct Particle predictor;


// the three (four for rk4) integrator steps
enum {
    RKSTART,
    RKFIRST,
    RKSECOND,
    RKTHIRD
};


// the implemented integrators
enum {
    EULER,
    RK2_ADAPTIVE,
    MONAGHAN_PC,
    EULER_PC,
    HEUN_RK4
};


#if FRAGMENTATION
extern int maxNumFlaws_host;
#endif

extern int *interactions;
extern int *interactions_host;

extern int *childList_host;
extern int *childListd;
extern __constant__ volatile int *childList;

extern int numberOfParticles;
extern int maxNumberOfParticles;
extern int numberOfRealParticles;

extern int numberOfPointmasses;
extern int memorySizeForPointmasses;
extern int integermemorySizeForPointmasses;

extern int memorySizeForParticles;
extern int memorySizeForTree;
extern int memorySizeForInteractions;
extern int memorySizeForChildren;
extern int memorySizeForStress;
#if FRAGMENTATION
extern int memorySizeForActivationThreshold;
#endif

extern int numberOfChildren; // 4 for 2D, 8 for 3D
extern int numberOfNodes;

extern int restartedRun;
extern int numberOfMultiprocessors;

extern double treeTheta;

typedef struct RunParameter {
    int performanceTest;
    int verbose;
    int restart;
    int selfgravity;
    int directselfgravity;
    int decouplegravity;
    int treeinformation;
    int hdf5output;
    int hdf5input;
    int ascii_output;
    int integrator_type;
    double maxtimestep;
    double firsttimestep;
    double angular_momentum_check;
    double rk_epsrel;
    double boundary_ratio;
    char kernel[256];
    char conservedquantitiesfilename[255];
    char binarysystemfilename[256];
    config_t config;
} RunParameter;

extern RunParameter param;

#endif
