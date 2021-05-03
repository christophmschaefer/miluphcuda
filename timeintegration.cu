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

#include "boundary.h"
#include "timeintegration.h"
#include "tree.h"
#include "porosity.h"
#include "pressure.h"
#include "plasticity.h"
#include "soundspeed.h"
#include "parameter.h"
#include "io.h"
#include "xsph.h"
#include "miluph.h"
#include "aneos.h"
#include "linalg.h"
#include "density.h"
#include "rhs.h"
#include "viscosity.h"
#include "float.h"
#include "extrema.h"
#include "sinking.h"
#include "config_parameter.h"


pthread_t fileIOthread;

double L_ini = 0.0;

__device__ double maxPosAbsError;
__device__ double maxVelAbsError;
__device__ int movingparticles = 0;
__device__ int reset_movingparticles = 1;
__device__ double dtNewErrorCheck = 0.0;
#if INTEGRATE_DENSITY
__device__ double maxDensityAbsError;
#endif
#if INTEGRATE_ENERGY
__device__ double maxEnergyAbsError;
#endif
#if PALPHA_POROSITY
__device__ double maxPressureAbsChange;
__device__ double maxAlphaDiff;
#endif
#if FRAGMENTATION
__device__ double maxDamageTimeStep;
#endif
__device__ int errorSmallEnough = FALSE;
__constant__ int isRelaxationRun = FALSE;
__constant__ volatile int *childList;
int *childListd;

/* time variables */
void (*integrator)();
int startTimestep = 0;
int numberOfTimesteps = 1;
double timePerStep;
double dt_host;
double dt_grav;
int gravity_index = 0;
int flag_force_gravity_calc = 0;
double currentTime;
double startTime;
double h5time;
__device__ double dt;   // timestep on the device
__device__ double dtmax;    // max allowed timestep (either from cmd-line or output timestep)
__device__ double endTimeD, currentTimeD;
__device__ double substep_currentTimeD;

__device__ int blockCount = 0;
__device__ volatile int maxNodeIndex;
__device__ volatile double radius;

// tree computational domain
double *minxPerBlock, *maxxPerBlock;
__device__ double minx, maxx;
#if DIM > 1
double *minyPerBlock, *maxyPerBlock;
__device__ double miny, maxy;
#endif
#if DIM == 3
double *minzPerBlock, *maxzPerBlock;
__device__ double minz, maxz;
#endif



// map [i][j] to [i*DIM*DIM+j] for the tensors
__device__ int stressIndex(int particleIndex, int row, int col)
{
    return particleIndex*DIM*DIM+row*DIM+col;
}


#if SOLID
__global__ void symmetrizeStress(void)
{
    register int i, j, k, inc;
    register double val;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        for (j = 0; j < DIM; j ++) {
            for (k = 0; k < j; k++) {
                val = 0.5 * (p.S[stressIndex(i,j,k)] + p.S[stressIndex(i,k,j)]);
                p.S[stressIndex(i,j,k)] = val;
                p.S[stressIndex(i,k,j)] = val;
            }
        }
    }
}
#endif


double calculate_angular_momentum(void)
{
    int i;
    double l_i = 0.0;
    double Lx = 0.0;
    double Ly = 0.0;
    double Lz = 0.0;
    double L = 0.0;

#if DIM > 1
    for (i = 0; i < numberOfParticles; i++) {
        l_i = 0;
#if DIM > 2
        l_i = p_host.m[i]*(p_host.y[i]*p_host.vz[i] - p_host.z[i]*p_host.vy[i]);
        Lx += l_i;
        l_i = p_host.m[i]*(p_host.z[i]*p_host.vx[i] - p_host.x[i]*p_host.vz[i]);
        Ly += l_i;
        l_i = p_host.m[i]*(p_host.x[i]*p_host.vy[i] - p_host.y[i]*p_host.vx[i]);
        Lz += l_i;
#else
        l_i = p_host.m[i]*(p_host.x[i]*p_host.vy[i] - p_host.y[i]*p_host.vx[i]);
        Lz += l_i;
#endif
    }
    L = sqrt(Lx*Lx + Ly*Ly + Lz*Lz);
#endif

    return L;
}


void initIntegration()
{
    L_ini = calculate_angular_momentum();
    if (param.verbose) {
        fprintf(stdout, "Initial angular momentum is: %.17e\n", L_ini);
    }

    dt_host = timePerStep;
    // copy constants to device
    cudaVerify(cudaMemcpyToSymbol(dt, &dt_host, sizeof(double)));
    cudaVerify(cudaMemcpyToSymbol(dtmax, &param.maxtimestep, sizeof(double)));
    cudaVerify(cudaMemcpyToSymbol(theta, &treeTheta, sizeof(double)));
    cudaVerify(cudaMemcpyToSymbol(numParticles, &numberOfParticles, sizeof(int)));
    cudaVerify(cudaMemcpyToSymbol(numPointmasses, &numberOfPointmasses, sizeof(int)));
    cudaVerify(cudaMemcpyToSymbol(maxNumParticles, &maxNumberOfParticles, sizeof(int)));
    cudaVerify(cudaMemcpyToSymbol(numRealParticles, &numberOfRealParticles, sizeof(int)));
    cudaVerify(cudaMemcpyToSymbol(numChildren, &numberOfChildren, sizeof(int)));
    cudaVerify(cudaMemcpyToSymbol(numNodes, &numberOfNodes, sizeof(int)));

#if FRAGMENTATION
    cudaVerify(cudaMemcpyToSymbol(maxNumFlaws, &maxNumFlaws_host, sizeof(int)));
#endif
    // memory for tree
    cudaVerify(cudaMalloc((void**)&minxPerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&maxxPerBlock, sizeof(double)*numberOfMultiprocessors));
#if DIM > 1
    cudaVerify(cudaMalloc((void**)&minyPerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&maxyPerBlock, sizeof(double)*numberOfMultiprocessors));
#endif
#if DIM == 3
    cudaVerify(cudaMalloc((void**)&minzPerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&maxzPerBlock, sizeof(double)*numberOfMultiprocessors));
#endif

    // set the pointer on the gpu to p_device
    cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
    cudaVerify(cudaMemcpyToSymbol(p_rhs, &p_device, sizeof(struct Particle)));

    cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
    cudaVerify(cudaMemcpyToSymbol(pointmass_rhs, &pointmass_device, sizeof(struct Pointmass)));

    cudaVerifyKernel((initializeSoundspeed<<<numberOfMultiprocessors*4, NUM_THREADS_512>>>()));
}


//this function is called after every successful integration (not only when ouput is generated)
void afterIntegrationStep(void)
{
#if PARTICLE_ACCRETION
    cudaVerifyKernel((ParticleSinking<<<numberOfMultiprocessors*4, NUM_THREADS_PRESSURE>>>()));
#endif

#if MORE_OUTPUT
	cudaVerifyKernel((get_extrema<<<numberOfMultiprocessors*4, NUM_THREADS_PRESSURE>>>()));
#endif
}


void endIntegration(void)
{
    int rc = pthread_join(fileIOthread, NULL);
    assert(0 == rc);

    // free memory
    cudaVerify(cudaFree(minxPerBlock));
    cudaVerify(cudaFree(maxxPerBlock));
#if DIM > 1
    cudaVerify(cudaFree(minyPerBlock));
    cudaVerify(cudaFree(maxyPerBlock));
#endif
#if DIM == 3
    cudaVerify(cudaFree(minzPerBlock));
    cudaVerify(cudaFree(maxzPerBlock));
#endif

    cleanupMaterials();
}


/* just do it */
void timeIntegration()
{
    initIntegration();
    integrator();
    endIntegration();
}
