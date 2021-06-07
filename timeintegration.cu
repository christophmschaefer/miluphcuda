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
#include "com_correction.h"

pthread_t fileIOthread;


double L_ini = 0.0;


// integration parameters
__constant__ double b21 = 0.5;
__constant__ double b31 = -1.0;
__constant__ double b32 = 2.0;
__constant__ double c1 = 1.0;
__constant__ double c2 = 4.0;
__constant__ double c3 = 1.0;

__constant__ double safety = 0.9;
__device__ double maxPosAbsError;
__device__ double maxVelAbsError;
__device__ int treeMaxDepth = 0;
__device__ int movingparticles = 0;
__device__ int reset_movingparticles = 1;
__device__ double dtNewErrorCheck = 0.0;
__device__ double dtNewAlphaCheck = 0.0;
#if INTEGRATE_DENSITY
__device__ double maxDensityAbsError;
#endif
#if INTEGRATE_ENERGY
__device__ double maxEnergyAbsError;
#endif
__device__ double maxPressureAbsChange;
#if FRAGMENTATION
__device__ double maxDamageTimeStep;
#endif
#if PALPHA_POROSITY
__device__ double maxalphaDiff = 0.0;
#endif
__device__ int errorSmallEnough = FALSE;
__constant__ int isRelaxationRun = FALSE;
__constant__ volatile int *childList;
int *childListd;
#if FIXED_BINARY
__device__ int cnt = -1;
double j_adv;
double M_acc1 = 0.0;
double M_acc2 = 0.0; 
double M_out = 0.0;
double M_in = 0.0;
int N_acc1 = 0;
int N_acc2 = 0;
int N_out = 0;
int N_in = 0;
double J_acc1 = 0.0;
double J_acc2 = 0.0;
double J_out = 0.0;
double J_in = 0.0;
int cnt_dt_acc = 0;
#endif



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
__device__ double dt;
__device__ double dtmax;
__device__ double endTimeD, currentTimeD;
__device__ double substep_currentTimeD;


__device__ int blockCount = 0;
__device__ volatile int maxNodeIndex;
int maxNodeIndex_host;
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
__device__ int stressIndex(int particleIndex, int row, int col) {
    return particleIndex*DIM*DIM+row*DIM+col;
}


#if SOLID
__global__ void symmetrizeStress(void) {
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


#if FIXED_BINARY
__global__ void bufferAccretedParticles(double *buffer_x, double *buffer_y, double *buffer_vx, double *buffer_vy, double *buffer_m, double *buffer_ID)
{
    register int i, inc;
    double r, x_star1, y_star1, x_star2, y_star2, dist_1, dist_2, vx_star1, vx_star2, vy_star1, vy_star2;
    inc = blockDim.x * gridDim.x;
    x_star1 = pointmass.x[0];
    y_star1 = pointmass.y[0];
    x_star2 = pointmass.x[1];
    y_star2 = pointmass.y[1];
    vx_star1 = pointmass.vx[0];
    vy_star1 = pointmass.vy[0];
    vx_star2 = pointmass.vx[1];
    vy_star2 = pointmass.vy[1];
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+=inc) {
        r = sqrt(p.x[i] * p.x[i] + p.y[i] * p.y[i]);
        dist_1 = sqrt( (p.x[i] - x_star1)*(p.x[i] - x_star1) + (p.y[i] - y_star1)*(p.y[i] - y_star1));
        dist_2 = sqrt( (p.x[i] - x_star2)*(p.x[i] - x_star2) + (p.y[i] - y_star2)*(p.y[i] - y_star2));

        if (p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
            continue;
        }


#if defined(FIXED_INNER_BOUNDARY)
        if (r < FIXED_INNER_BOUNDARY && p_rhs.materialId[i] != EOS_TYPE_IGNORE) {                                        //need to fix this boundary, use pointmass.rmin but has to be equal for both stars !
            atomicAdd(&cnt,1);
            p_rhs.materialId[i] = EOS_TYPE_IGNORE;                 // We give the particle the ignored flag, then copy
            buffer_x[cnt] = p.x[i];                                // its parameters to the buffer with atomic cnt
            buffer_y[cnt] = p.y[i];                                // then we "deactivate" the particle
            buffer_vx[cnt] = p.vx[i];
            buffer_vy[cnt] = p.vy[i];
            buffer_m[cnt] = p.m[i];
            buffer_ID[cnt] = EOS_TYPE_ACCRETED_BY_INNER_BOUNDARY;
            p.vx[i] = 0;
            p.vy[i] = 0;
            p.dxdt[i] = 0;
            p.dydt[i] = 0;
            p.ax[i] = 0;
            p.ay[i] = 0;
        }
#else // accretion on individual stars
        if (pointmass.rmin[0] > dist_1 && p_rhs.materialId[i] != EOS_TYPE_IGNORE) {
            atomicAdd(&cnt,1);
            p_rhs.materialId[i] = EOS_TYPE_IGNORE;                 // We give the particle the ignored flag, then copy
            buffer_x[cnt] = p.x[i] - x_star1;                                // its parameters to the buffer with atomic cnt
            buffer_y[cnt] = p.y[i] - y_star1;                                // then we "deactivate" the particle
            buffer_vx[cnt] = p.vx[i] - vx_star1;
            buffer_vy[cnt] = p.vy[i] - vy_star1;
            buffer_m[cnt] = p.m[i];
            buffer_ID[cnt] = EOS_TYPE_ACCRETED_BY_STAR1;
            p.vx[i] = 0;
            p.vy[i] = 0;
            p.dxdt[i] = 0;
            p.dydt[i] = 0;
            p.ax[i] = 0;
            p.ay[i] = 0;
        } else if (pointmass.rmin[1] > dist_2 && p_rhs.materialId[i] != EOS_TYPE_IGNORE) {
            atomicAdd(&cnt,1);
            p_rhs.materialId[i] = EOS_TYPE_IGNORE;                 // We give the particle the ignored flag, then copy
            buffer_x[cnt] = p.x[i] - x_star2;                                // its parameters to the buffer with atomic cnt
            buffer_y[cnt] = p.y[i] - y_star2;                                // then we "deactivate" the particle
            buffer_vx[cnt] = p.vx[i] - vx_star2;
            buffer_vy[cnt] = p.vy[i] - vy_star2;
            buffer_m[cnt] = p.m[i];
            buffer_ID[cnt] = EOS_TYPE_ACCRETED_BY_STAR2;
            p.vx[i] = 0;
            p.vy[i] = 0;
            p.dxdt[i] = 0;
            p.dydt[i] = 0;
            p.ax[i] = 0;
            p.ay[i] = 0;
        }
#endif
        if ( r >= FIXED_OUTER_BOUNDARY && p_rhs.materialId[i] != EOS_TYPE_IGNORE) {                                     // same boundary fix here
            atomicAdd(&cnt, 1);
            p_rhs.materialId[i] = EOS_TYPE_IGNORE;
            buffer_x[cnt] = p.x[i];
            buffer_y[cnt] = p.y[i];
            buffer_vx[cnt] = p.vx[i];
            buffer_vy[cnt] = p.vy[i];
            buffer_m[cnt] = p.m[i];
            buffer_ID[cnt] = EOS_TYPE_ACCRETED_BY_OUTER_BOUNDARY;                            // -3 flag == outer boundary crossed
            p.vx[i] = 0;
            p.vy[i] = 0;
            p.dxdt[i] = 0;
            p.dydt[i] = 0;
            p.ax[i] = 0;
            p.ay[i] = 0;
        }

        if (cnt >= ACCRETION_BUFFER) {
            printf("Warning, the size of the buffer for the accretion is exceeded !");
            assert(!!!"buffer size of accretion is exceeded"); 
        }

    }
    cnt = -1;
}


void fixedParticlesAccretion()
{
    int i;
    double *d_buffer_x, *d_buffer_y, *d_buffer_vx, *d_buffer_vy, *d_buffer_m, *d_buffer_ID;
    double *h_buffer_x, *h_buffer_y, *h_buffer_vx, *h_buffer_vy, *h_buffer_m, *h_buffer_ID;


    // mem for ACCRETION_BUFFER particles
    const size_t buffer_sz = ACCRETION_BUFFER * sizeof(double);

// Allocating memory
    cudaVerify(cudaMalloc((void **)&d_buffer_x, buffer_sz));
    cudaVerify(cudaMalloc((void **)&d_buffer_y, buffer_sz));
    cudaVerify(cudaMalloc((void **)&d_buffer_vx, buffer_sz));
    cudaVerify(cudaMalloc((void **)&d_buffer_vy, buffer_sz));
    cudaVerify(cudaMalloc((void **)&d_buffer_m, buffer_sz));
    cudaVerify(cudaMalloc((void **)&d_buffer_ID, buffer_sz));

    cudaMemset(d_buffer_x, 0, buffer_sz);
    cudaMemset(d_buffer_y, 0, buffer_sz);
    cudaMemset(d_buffer_vx, 0, buffer_sz);
    cudaMemset(d_buffer_vy, 0, buffer_sz);
    cudaMemset(d_buffer_m, 0, buffer_sz);
    cudaMemset(d_buffer_ID, 0, buffer_sz);



    cudaVerify(cudaMallocHost((void **)&h_buffer_x, buffer_sz));
    cudaVerify(cudaMallocHost((void **)&h_buffer_y, buffer_sz));
    cudaVerify(cudaMallocHost((void **)&h_buffer_vx, buffer_sz));
    cudaVerify(cudaMallocHost((void **)&h_buffer_vy, buffer_sz));
    cudaVerify(cudaMallocHost((void **)&h_buffer_m, buffer_sz));
    cudaVerify(cudaMallocHost((void **)&h_buffer_ID, buffer_sz));


//Kernel call
    cudaVerifyKernel((bufferAccretedParticles<<<numberOfMultiprocessors*4, NUM_THREADS_PRESSURE>>>(d_buffer_x, d_buffer_y, d_buffer_vx, d_buffer_vy, d_buffer_m, d_buffer_ID)));
    cudaVerify(cudaDeviceSynchronize());

// copy back from device to host
    cudaVerify(cudaMemcpy(h_buffer_x, d_buffer_x, buffer_sz, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(h_buffer_y, d_buffer_y, buffer_sz, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(h_buffer_vx, d_buffer_vx, buffer_sz, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(h_buffer_vy, d_buffer_vy, buffer_sz, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(h_buffer_m, d_buffer_m, buffer_sz, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(h_buffer_ID, d_buffer_ID, buffer_sz, cudaMemcpyDeviceToHost));


// Free device memory
    cudaVerify(cudaFree(d_buffer_x));
    cudaVerify(cudaFree(d_buffer_y));
    cudaVerify(cudaFree(d_buffer_vx));
    cudaVerify(cudaFree(d_buffer_vy));
    cudaVerify(cudaFree(d_buffer_m));
    cudaVerify(cudaFree(d_buffer_ID));
// WARNING: current version is computing the AM accreted onto the COM. If
// distinction between accreting star is made, NEED for AM computation to be
// done according to distance(and velocity) between particle AND the considered
// star :(x* - xp) and (v* - vp)

    j_adv = 0;
    for (i = 0; i < ACCRETION_BUFFER; i++) {
        j_adv = h_buffer_m[i] * (h_buffer_x[i] * h_buffer_vy[i] - h_buffer_y[i] * h_buffer_vx[i]);
        if (h_buffer_ID[i] == EOS_TYPE_ACCRETED_BY_INNER_BOUNDARY) {
            M_in += h_buffer_m[i];
            J_in += j_adv;
            N_in += 1;
        } else if (h_buffer_ID[i] == EOS_TYPE_ACCRETED_BY_STAR1) {
            M_acc1 += h_buffer_m[i];
            J_acc1 += j_adv;
            N_acc1 +=1;
        } else if (h_buffer_ID[i] == EOS_TYPE_ACCRETED_BY_STAR2) {
            M_acc2 += h_buffer_m[i];
            J_acc2 += j_adv;
            N_acc2 += 1;
        } else if (h_buffer_ID[i] == EOS_TYPE_ACCRETED_BY_OUTER_BOUNDARY) {
            M_out += h_buffer_m[i];
            J_out += j_adv;
            N_out += 1;
        }
    }
    printf("SKIPPING ACCRETION OUTPUT SINCE ACCRETION COUNTER IS %d\n", cnt_dt_acc);
    cnt_dt_acc += 1;
    if (cnt_dt_acc >= 10) {   // Print accretion logs only every 10 calls of the function
        printf("Writing torque data to log file %s and resetting counters ...\n", param.accretionfilename);
        double J_grav1, J_grav2;
        cudaVerify(cudaMemcpy(&J_grav1, &pointmass_device.torque_z[0], sizeof(double), cudaMemcpyDeviceToHost));
        cudaVerify(cudaMemcpy(&J_grav2, &pointmass_device.torque_z[1], sizeof(double), cudaMemcpyDeviceToHost));
        fprintf(param.accretionfile, "%.17le\t %.17le\t %.17le\t %.17le\t %.17le\t %.17le\t %.17le\t %.17le\t %.17le\t %d\t %d\t %d\t %d\t %.17le\t %.17le \n", currentTime, M_acc1, M_acc2, M_in, M_out, J_acc1, J_acc2, J_in, J_out, N_acc1, N_acc2, N_in, N_out, J_grav1, J_grav2);
        printf("at time %e: %d particles left the simulation by accretion on the stars, %d by leaving the inner boundary, %d by leaving the outer boundary\n", currentTime, N_acc1+N_acc2, N_in, N_out);
        M_acc1 = 0.0;
        J_acc1 = 0.0;
        N_acc1 = 0;
        M_acc2 = 0.0;
        J_acc2 = 0.0;
        N_acc2 = 0;
        M_out = 0.0;
        J_out = 0.0;
        N_out = 0;
        M_in = 0.0;
        J_in = 0.0;
        N_in = 0;
        cnt_dt_acc = 0;
        //fflush(param.accretionfile);
    }

// Free host memory
    cudaVerify(cudaFreeHost(h_buffer_x));
    cudaVerify(cudaFreeHost(h_buffer_y));
    cudaVerify(cudaFreeHost(h_buffer_vx));
    cudaVerify(cudaFreeHost(h_buffer_vy));
    cudaVerify(cudaFreeHost(h_buffer_m));
    cudaVerify(cudaFreeHost(h_buffer_ID));

}

#endif

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
#if FIXED_BINARY
    fixedParticlesAccretion();    
#endif

#if PARTICLE_ACCRETION
    cudaVerifyKernel((ParticleSinking<<<numberOfMultiprocessors*4, NUM_THREADS_PRESSURE>>>()));
#endif

#if MORE_OUTPUT
	cudaVerifyKernel((get_extrema<<<numberOfMultiprocessors*4, NUM_THREADS_PRESSURE>>>()));
#endif

#if MOVING_COM_CORRECTION
    cudaVerifyKernel((COMcorrection<<<numberOfMultiprocessors*4, NUM_THREADS_PRESSURE>>>()));    
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
