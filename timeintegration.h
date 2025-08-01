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

#ifndef _TIMEINTEGRATION_H
#define _TIMEINTEGRATION_H

#include <stdio.h>
#include <libconfig.h>
#include <pthread.h>
#include <assert.h>
#include "parameter.h"
#include "miluph.h"
#include "io.h"
#include "cuda_utils.h"
#include "kernel.h"
#include "extrema.h"
#include "sinking.h"


// Courant (CFL) number (note that our sml is defined up to the zero of the kernel, not half of it)
#define COURANT_FACT 0.4
// factor for limiting timestep based on local forces/acceleration
#define FORCES_FACT 0.2


extern int startTimestep;
extern int numberOfTimesteps;
extern double startTime;
extern double timePerStep;
extern double dt_host;

extern double currentTime;
extern double h5time;

extern __device__ int numParticles;
extern __device__ int numPointmasses;
extern int *relaxedPerBlock;


extern void (*integrator)();


void timeIntegration(void);
// the available integrators
void euler(void);
void predictor_corrector(void);
void predictor_corrector_euler(void);
void rk2Adaptive(void);
void rk4_nbodies(void); // do we need it here? do we say.. bye,bye?
void heun_rk4(void);

double calculate_angular_momentum(void);

void copyToHostAndWriteToFile(int timestep, int lastTimestep);

__device__ int childListIndex(int nodeIndex, int childNumber);
__global__ void detectVelocityRelaxation(int *relaxedPerBlock);
__device__ int stressIndex(int particleIndex, int row, int col);

void afterIntegrationStep(void);

__global__ void symmetrizeStress(void);


#define NUM_THREADS_512 512
#define NUM_THREADS_256 256
#define NUM_THREADS_128 128
#define NUM_THREADS_64 64
#define NUM_THREADS_1 1

#define NUM_THREADS_COMPUTATIONAL_DOMAIN 128
#define NUM_THREADS_BUILD_TREE 32
#define NUM_THREADS_TREEDEPTH 128
#define NUM_THREADS_TREECHANGE 128
#define NUM_THREADS_CALC_CENTER_OF_MASS 256
#define NUM_THREADS_SELFGRAVITY 128
#define NUM_THREADS_BOUNDARY_CONDITIONS 128
#define NUM_THREADS_NEIGHBOURSEARCH 256
#define NUM_THREADS_SYMMETRIZE_INTERACTIONS 256
#define NUM_THREADS_DENSITY 256
#define NUM_THREADS_PRESSURE 256
#define NUM_THREADS_PALPHA_POROSITY 256
#define NUM_THREADS_REDUCTION 256

#define NUM_THREADS_DETECTRELAX 256
#define NUM_THREADS_LIMITTIMESTEP 256

// RK2
#define NUM_THREADS_RK2_INTEGRATE_STEP 256
#define NUM_THREADS_ERRORCHECK 256
// RK4
#define NUM_THREADS_RK4_INTEGRATE_STEP 256
// EULER
#define NUM_THREADS_EULER_INTEGRATOR 256

// PREDICTOR-CORRECTOR
#define NUM_THREADS_PC_INTEGRATOR 256

#define EMPTY -1
#define LOCKED -2


// 128 seems like a very high default
//#define MAXDEPTH 128
#define MAXDEPTH 64


// Runge-Kutta constants
#define B21 0.5
#define B31 -1.0
#define B32 2.0
#define C1 1.0
#define C2 4.0
#define C3 1.0


#endif
