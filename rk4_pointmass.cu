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

#include "rk4_pointmass.h"
#include "miluph.h"
#include "timeintegration.h"
#include "parameter.h"
#include "memory_handling.h"
#include "rhs.h"
#include "pressure.h"
#include "boundary.h"
#include "config_parameter.h"

extern __device__ double dt;


/* the runge-kutta 4nd order integrator with fixed timestep */
void rk4_nbodies()
{
    int rkstep;

    // alloc mem for multiple rhs and copy immutables
    int allocate_immutables = 0;
    for (rkstep = 0; rkstep < 4; rkstep++) {
        copy_pointmass_immutables_device_to_device(&rk4_pointmass_device[rkstep], &pointmass_device);
    }

    copy_pointmass_variables_device_to_device(&rk4_pointmass_device[RKFIRST], &pointmass_device);
    copy_pointmass_variables_device_to_device(&rk4_pointmass_device[RKSTART], &pointmass_device);
    cudaVerify(cudaDeviceSynchronize());

    // calculate first right hand side with rk[RKFIRST]_device
    cudaVerify(cudaMemcpyToSymbol(pointmass, &rk4_pointmass_device[RKFIRST], sizeof(struct Pointmass)));
    cudaVerifyKernel((rhs_pointmass<<<numberOfMultiprocessors, NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
    cudaVerify(cudaDeviceSynchronize());

    // remember values of first step
    copy_pointmass_variables_device_to_device(&rk4_pointmass_device[RKSTART], &rk4_pointmass_device[RKFIRST]);
    copy_pointmass_derivatives_device_to_device(&rk4_pointmass_device[RKSTART], &rk4_pointmass_device[RKFIRST]);
    cudaVerify(cudaDeviceSynchronize());
    // set rk[RKFIRST] variables
    cudaVerifyKernel((rhs_pointmass<<<numberOfMultiprocessors,NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
    cudaVerifyKernel((rk4_integrateFirstStep<<<numberOfMultiprocessors, NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
    cudaVerify(cudaDeviceSynchronize());

    // get derivatives for second step
    cudaVerify(cudaMemcpyToSymbol(pointmass, &rk4_pointmass_device[RKFIRST], sizeof(struct Pointmass)));
    cudaVerifyKernel((rhs_pointmass<<<numberOfMultiprocessors,NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
    cudaVerify(cudaDeviceSynchronize());
    // rk4_integrate second step
    cudaVerifyKernel((rk4_integrateSecondStep<<<numberOfMultiprocessors, NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
    cudaVerify(cudaDeviceSynchronize());

    // get derivatives for third step
    cudaVerify(cudaMemcpyToSymbol(pointmass, &rk4_pointmass_device[RKSECOND], sizeof(struct Pointmass)));
    cudaVerifyKernel((rhs_pointmass<<<numberOfMultiprocessors,NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
    // rk4_integrate third step
    cudaVerifyKernel((rk4_integrateThirdStep<<<numberOfMultiprocessors, NUM_THREADS_RK4_INTEGRATE_STEP>>>()));

    // get derivatives for the fourth (and last) step
    // this happens at t = t0 + h
    cudaVerify(cudaMemcpyToSymbol(pointmass, &rk4_pointmass_device[RKTHIRD], sizeof(struct Pointmass)));
    cudaVerifyKernel((rhs_pointmass<<<numberOfMultiprocessors,NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
    cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
    // rk4_integrate fourth step
    cudaVerifyKernel((rk4_integrateFourthStep<<<numberOfMultiprocessors, NUM_THREADS_RK4_INTEGRATE_STEP>>>()));
}

// acceleration due to the point masses
__global__ void rhs_pointmass()
{
    int i, inc;
    int j;
    int d;
    double r;
    double rrr;
    double dr[DIM];
    inc = blockDim.x * gridDim.x;
    // loop for point masses
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i += inc) {
        pointmass.ax[i] = 0.0;
#if DIM > 1
        pointmass.ay[i] = 0.0;
#if DIM > 2
        pointmass.az[i] = 0.0;
#endif
#endif
        for (j = 0; j < numPointmasses; j++) {
            if (i == j) continue;
            r = 0.0;
            dr[0] = pointmass.x[j] - pointmass.x[i];
#if DIM > 1
            dr[1] = pointmass.y[j] - pointmass.y[i];
#if DIM > 2
            dr[2] = pointmass.z[j] - pointmass.z[i];
#endif
#endif
            for (d = 0; d < DIM; d++) {
                r += dr[d]*dr[d];
            }
            r = sqrt(r);
            rrr = r*r*r;
            pointmass.ax[i] += gravConst * pointmass.m[j] * dr[0]/(rrr);
#if DIM > 1
            pointmass.ay[i] += gravConst * pointmass.m[j] * dr[1]/(rrr);
#if DIM > 2
            pointmass.az[i] += gravConst * pointmass.m[j] * dr[2]/(rrr);
#endif
#endif
        }
        if (pointmass_rhs.feels_particles[i]) {
            pointmass.ax[i] += pointmass_rhs.feedback_ax[i];
#if DIM > 1
            pointmass.ay[i] += pointmass_rhs.feedback_ay[i];
#if DIM > 2
            pointmass.az[i] += pointmass_rhs.feedback_az[i];
#endif
#endif
        }
    }
}

__global__ void rk4_integrateFirstStep()
{
    int i;
    // loop for the point masses
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i+= blockDim.x * gridDim.x) {
        rk4_pointmass[RKFIRST].x[i] = rk4_pointmass[RKSTART].x[i] + dt * B21 * rk4_pointmass[RKFIRST].vx[i];
#if DIM > 1
        rk4_pointmass[RKFIRST].y[i] = rk4_pointmass[RKSTART].y[i] + dt * B21 * rk4_pointmass[RKFIRST].vy[i];
#endif
#if DIM == 3
        rk4_pointmass[RKFIRST].z[i] = rk4_pointmass[RKSTART].z[i] + dt * B21 * rk4_pointmass[RKFIRST].vz[i];
#endif

        rk4_pointmass[RKFIRST].vx[i] = rk4_pointmass[RKSTART].vx[i] + dt * B21 * rk4_pointmass[RKFIRST].ax[i];
#if DIM > 1
        rk4_pointmass[RKFIRST].vy[i] = rk4_pointmass[RKSTART].vy[i] + dt * B21 * rk4_pointmass[RKFIRST].ay[i];
#endif
#if DIM == 3
        rk4_pointmass[RKFIRST].vz[i] = rk4_pointmass[RKSTART].vz[i] + dt * B21 * rk4_pointmass[RKFIRST].az[i];
#endif
    }
}

__global__ void rk4_integrateSecondStep()
{
    int i;
    // loop for pointmasses
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i+= blockDim.x * gridDim.x) {
        rk4_pointmass[RKSECOND].vx[i] = rk4_pointmass[RKSTART].vx[i] + dt * B21 * rk4_pointmass[RKFIRST].ax[i];
#if DIM > 1
        rk4_pointmass[RKSECOND].vy[i] = rk4_pointmass[RKSTART].vy[i] + dt * B21 * rk4_pointmass[RKFIRST].ay[i];
#endif
#if DIM == 3
        rk4_pointmass[RKSECOND].vz[i] = rk4_pointmass[RKSTART].vz[i] + dt * B21 * rk4_pointmass[RKFIRST].az[i];
#endif

        rk4_pointmass[RKSECOND].x[i] = rk4_pointmass[RKSTART].x[i] + dt * B21 * rk4_pointmass[RKFIRST].vx[i];
#if DIM > 1
        rk4_pointmass[RKSECOND].y[i] = rk4_pointmass[RKSTART].y[i] + dt * B21 * rk4_pointmass[RKFIRST].vy[i];
#endif
#if DIM == 3
        rk4_pointmass[RKSECOND].z[i] = rk4_pointmass[RKSTART].z[i] + dt * B21 * rk4_pointmass[RKFIRST].vz[i];
#endif
    }
}

__global__ void rk4_integrateThirdStep()
{
    int i;
    // loop for pointmasses
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i+= blockDim.x * gridDim.x) {
        rk4_pointmass[RKTHIRD].vx[i] = rk4_pointmass[RKSTART].vx[i] + dt * rk4_pointmass[RKSECOND].ax[i];
#if DIM > 1
        rk4_pointmass[RKTHIRD].vy[i] = rk4_pointmass[RKSTART].vy[i] + dt * rk4_pointmass[RKSECOND].ay[i];
#endif
#if DIM == 3
        rk4_pointmass[RKTHIRD].vz[i] = rk4_pointmass[RKSTART].vz[i] + dt * rk4_pointmass[RKSECOND].az[i];
#endif

        rk4_pointmass[RKTHIRD].x[i] = rk4_pointmass[RKSTART].x[i] + dt * rk4_pointmass[RKSECOND].vx[i];
#if DIM > 1
        rk4_pointmass[RKTHIRD].y[i] = rk4_pointmass[RKSTART].y[i] + dt * rk4_pointmass[RKSECOND].vy[i];
#endif
#if DIM == 3
        rk4_pointmass[RKTHIRD].z[i] = rk4_pointmass[RKSTART].z[i] + dt * rk4_pointmass[RKSECOND].vz[i];
#endif
    }
}

__global__ void rk4_integrateFourthStep()
{
    int i;
    int d;
    // loop pointmasses
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i+= blockDim.x * gridDim.x) {
        pointmass.vx[i] = rk4_pointmass[RKSTART].vx[i] + dt/6.0 * (C1 * rk4_pointmass[RKSTART].ax[i] + B32 * rk4_pointmass[RKFIRST].ax[i] + B32 * rk4_pointmass[RKSECOND].ax[i] + C1 * rk4_pointmass[RKTHIRD].ax[i]);
        pointmass.ax[i] = 1./6.0 *(C1 * rk4_pointmass[RKSTART].ax[i] + B32 * rk4_pointmass[RKFIRST].ax[i] + B32 * rk4_pointmass[RKSECOND].ax[i] + C1 * rk4_pointmass[RKTHIRD].ax[i]);
#if DIM > 1
        pointmass.vy[i] = rk4_pointmass[RKSTART].vy[i] + dt/6.0 * (C1 * rk4_pointmass[RKSTART].ay[i] + B32 * rk4_pointmass[RKFIRST].ay[i] + B32 * rk4_pointmass[RKSECOND].ay[i] + C1 * rk4_pointmass[RKTHIRD].ay[i]);
        pointmass.ay[i] = 1./6.0 *(C1 * rk4_pointmass[RKSTART].ay[i] + B32 * rk4_pointmass[RKFIRST].ay[i] + B32 * rk4_pointmass[RKSECOND].ay[i] + C1 * rk4_pointmass[RKTHIRD].ay[i]);
#endif
#if DIM > 2
        pointmass.vz[i] = rk4_pointmass[RKSTART].vz[i] + dt/6.0 * (C1 * rk4_pointmass[RKSTART].az[i] + B32 * rk4_pointmass[RKFIRST].az[i] + B32 * rk4_pointmass[RKSECOND].az[i] + C1 * rk4_pointmass[RKTHIRD].az[i]);
        pointmass.az[i] = 1./6.0 *(C1 * rk4_pointmass[RKSTART].az[i] + B32 * rk4_pointmass[RKFIRST].az[i] + B32 * rk4_pointmass[RKSECOND].az[i] + C1 * rk4_pointmass[RKFIRST].az[i]);
#endif

        pointmass.x[i] = rk4_pointmass[RKSTART].x[i] + dt/6.0 * (C1 * rk4_pointmass[RKSTART].vx[i] + B32 * rk4_pointmass[RKFIRST].vx[i] + B32 * rk4_pointmass[RKSECOND].vx[i] + C1 * rk4_pointmass[RKTHIRD].vx[i]);
#if DIM > 1
        pointmass.y[i] = rk4_pointmass[RKSTART].y[i] + dt/6.0 * (C1 * rk4_pointmass[RKSTART].vy[i] + B32 * rk4_pointmass[RKFIRST].vy[i] + B32 * rk4_pointmass[RKSECOND].vy[i] + C1 * rk4_pointmass[RKTHIRD].vy[i]);
#endif
#if DIM > 2
        pointmass.z[i] = rk4_pointmass[RKSTART].z[i] + dt/6.0 * (C1 * rk4_pointmass[RKSTART].vz[i] + B32 * rk4_pointmass[RKFIRST].vz[i] + B32 * rk4_pointmass[RKSECOND].vz[i] + C1 * rk4_pointmass[RKTHIRD].vz[i]);
#endif
    }
}
