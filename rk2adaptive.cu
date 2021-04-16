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

#include "rk2adaptive.h"
#include "miluph.h"
#include "timeintegration.h"
#include "config_parameter.h"
#include "parameter.h"
#include "memory_handling.h"
#include "rhs.h"
#include "pressure.h"
#include "boundary.h"
#include "damage.h"

extern __device__ double endTimeD, currentTimeD;
extern __device__ double substep_currentTimeD;
extern __device__ double dt;
extern __device__ int isRelaxed;
extern __device__ int blockCount;
extern __device__ int errorSmallEnough;
extern __device__ double dtNewErrorCheck;
extern __device__ double dtNewAlphaCheck;
extern __device__ double maxPosAbsError;

extern __device__ double maxVelAbsError;
extern __device__ double maxDensityAbsError;
extern __device__ double maxEnergyAbsError;
extern __device__ double maxPressureAbsChange;
extern __device__ double maxDamageTimeStep;
extern __device__ double maxalphaDiff;

__constant__ __device__ double rk_epsrel_d;

extern double L_ini;


/*
   the runge-kutta 2nd order integrator with adaptive timestep
   see cuda-paper for details
 */
void rk2Adaptive()
{
    int rkstep;
    int errorSmallEnough_host;
    double dtmax_host = param.maxtimestep;
    assert(dtmax_host > 0);
    double dtNewErrorCheck_host = 0.0;
#if PALPHA_POROSITY
    double dtNewAlphaCheck_host = -1.0;
    double dt_alphanew = 0;
    double dt_alphaold = 0;
#endif
#if FRAGMENTATION
    double dt_damagenew = 0;
    double dt_damageold = 0;
#endif

    /* first of all copy the rk_epsrel to the device */
    cudaVerify(cudaMemcpyToSymbol(rk_epsrel_d, &param.rk_epsrel, sizeof(double)));

    // allocate memory for runge kutta second order
    double *maxPosAbsErrorPerBlock, *maxVelAbsErrorPerBlock;
    cudaVerify(cudaMalloc((void**)&maxPosAbsErrorPerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&maxVelAbsErrorPerBlock, sizeof(double)*numberOfMultiprocessors));
#if INTEGRATE_DENSITY
    double *maxDensityAbsErrorPerBlock;
    cudaVerify(cudaMalloc((void**)&maxDensityAbsErrorPerBlock , sizeof(double)*numberOfMultiprocessors));
#endif
#if INTEGRATE_ENERGY
    double *maxEnergyAbsErrorPerBlock;
    cudaVerify(cudaMalloc((void**)&maxEnergyAbsErrorPerBlock, sizeof(double)*numberOfMultiprocessors));
#endif
#if FRAGMENTATION
    double *maxDamageTimeStepPerBlock;
    cudaVerify(cudaMalloc((void**)&maxDamageTimeStepPerBlock, sizeof(double)*numberOfMultiprocessors));
#endif
#if PALPHA_POROSITY
    double *maxalphaDiffPerBlock;
    cudaVerify(cudaMalloc((void**)&maxalphaDiffPerBlock, sizeof(double)*numberOfMultiprocessors));
    double *maxPressureAbsChangePerBlock;
    cudaVerify(cudaMalloc((void**)&maxPressureAbsChangePerBlock, sizeof(double)*numberOfMultiprocessors));
#endif

    // alloc mem for multiple rhs and copy immutables
    int allocate_immutables = 0;
    for (rkstep = 0; rkstep < 3; rkstep++) {
        allocate_particles_memory(&rk_device[rkstep], allocate_immutables);
        copy_particles_immutables_device_to_device(&rk_device[rkstep], &p_device);
#if GRAVITATING_POINT_MASSES
        allocate_pointmass_memory(&rk_pointmass_device[rkstep], allocate_immutables);
        copy_pointmass_immutables_device_to_device(&rk_pointmass_device[rkstep], &pointmass_device);
#endif
    }
    // set the symbol pointers
    cudaVerify(cudaMemcpyToSymbol(rk, &rk_device, sizeof(struct Particle) * 3));
#if GRAVITATING_POINT_MASSES
    cudaVerify(cudaMemcpyToSymbol(rk_pointmass, &rk_pointmass_device, sizeof(struct Pointmass) * 3));
#endif

    cudaVerify(cudaDeviceSynchronize());

    int lastTimestep = startTimestep + numberOfTimesteps;
    int timestep;
    int nsteps_cnt = 0;
    double dt_suggested = timePerStep;
    currentTime = startTime;
    double endTime = startTime;
    double substep_currentTime;

    cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));

    // loop over output steps
    for (timestep = startTimestep; timestep < lastTimestep; timestep++) {
        endTime += timePerStep;
        assert(endTime > currentTime);
        cudaVerify(cudaMemcpyToSymbol(endTimeD, &endTime, sizeof(double)));
        fprintf(stdout, "\n\nStart integrating output step %d / %d from time %g to %g...\n",
                timestep+1, lastTimestep, currentTime, endTime);

        // set first dt for this output step
        if (nsteps_cnt == 0) {
            if (param.firsttimestep > 0 && timePerStep > param.firsttimestep) {
                dt_host = dt_suggested = param.firsttimestep;
            } else if (dtmax_host < timePerStep) {
                dt_host = dt_suggested = dtmax_host;
            } else {
                dt_host = dt_suggested = timePerStep;
            }
            if (param.verbose)
                fprintf(stdout, "   starting with timestep: %e\n", dt_host);
        } else {
            dt_host = dt_suggested;   // use previously suggested next timestep as starting point
            if (dt_host > timePerStep)
                dt_host = timePerStep;
            if (param.verbose)
                fprintf(stdout, "   continuing with timestep: %e\n", dt_host);
        }
        assert(dt_host > 0);
        assert(dt_host <= timePerStep);
        cudaVerify(cudaMemcpyToSymbol(dt, &dt_host, sizeof(double)));
        nsteps_cnt++;

        // checking for changes in angular momentum
        if (param.angular_momentum_check > 0) {
            double L_current = calculate_angular_momentum();
            double L_change_relative;
            if (L_ini > 0) {
                L_change_relative = fabs((L_ini - L_current)/L_ini);
            }
            if (param.verbose) {
                fprintf(stdout, "Checking angular momentum conservation.\n");
                fprintf(stdout, "Initial angular momentum: %.17e\n", L_ini);
                fprintf(stdout, "Current angular momentum: %.17e\n", L_current);
                fprintf(stdout, "Relative change: %.17e\n", L_change_relative);
            }
            if (L_change_relative > param.angular_momentum_check) {
                fprintf(stderr, "Conservation of angular momentum violated. Exiting.\n");
                exit(111);
            }
        }

        // loop until end of current (output) timestep
        while (currentTime < endTime) {
            // get the correct time
            substep_currentTime = currentTime;
            cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));

            cudaVerify(cudaDeviceSynchronize());
            // copy particle data to first runge kutta step
            copy_particles_variables_device_to_device(&rk_device[RKFIRST], &p_device);
            cudaVerify(cudaDeviceSynchronize());
#if GRAVITATING_POINT_MASSES
            copy_pointmass_variables_device_to_device(&rk_pointmass_device[RKFIRST], &pointmass_device);
            cudaVerify(cudaDeviceSynchronize());
#endif

            // calculate first right hand side with rk[RKFIRST]_device
            cudaVerify(cudaMemcpyToSymbol(p, &rk_device[RKFIRST], sizeof(struct Particle)));
            cudaVerify(cudaMemcpyToSymbol(pointmass, &rk_pointmass_device[RKFIRST], sizeof(struct Pointmass)));
            rightHandSide();
            cudaVerify(cudaDeviceSynchronize());

#if FRAGMENTATION
            cudaVerify(cudaMemcpyFromSymbol(&dt_damageold, dt, sizeof(double)));
            /* add function for best timestep with fragmentation here */
            cudaVerifyKernel((damageMaxTimeStep<<<numberOfMultiprocessors, NUM_THREADS_ERRORCHECK>>>(
                                maxDamageTimeStepPerBlock)));
            cudaVerify(cudaMemcpyFromSymbol(&dt_damagenew, dt, sizeof(double)));
            if (dt_damagenew < dt_damageold) {
                dt_host = dt_suggested = dt_damagenew;
                if (param.verbose)
                    fprintf(stdout, "reducing timestep due to damage evolution from %g to %g (current time: %e)\n",
                            dt_damageold, dt_damagenew, currentTime);
            }
#endif

            // remember values of first step
            copy_particles_variables_device_to_device(&rk_device[RKSTART], &rk_device[RKFIRST]);
            copy_particles_derivatives_device_to_device(&rk_device[RKSTART], &rk_device[RKFIRST]);
#if GRAVITATING_POINT_MASSES
            copy_pointmass_variables_device_to_device(&rk_pointmass_device[RKSTART], &rk_pointmass_device[RKFIRST]);
            copy_pointmass_derivatives_device_to_device(&rk_pointmass_device[RKSTART], &rk_pointmass_device[RKFIRST]);
#endif

            // remember accels due to gravity
            if (param.selfgravity) {
                copy_gravitational_accels_device_to_device(&rk_device[RKSTART], &rk_device[RKFIRST]);
            }

            // integrate with adaptive timestep and break loop once acceptable
            while (TRUE) {
                cudaVerify(cudaDeviceSynchronize());
                
                // compute
                //    q_n + 0.5*h*k1
                // and store quantities in [RKFIRST]
                cudaVerifyKernel((integrateFirstStep<<<numberOfMultiprocessors, NUM_THREADS_RK2_INTEGRATE_STEP>>>()));
                cudaVerify(cudaDeviceSynchronize());
                
                // check for SMALLEST_DT_ALLOWED
                cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
                if (dt_host < SMALLEST_DT_ALLOWED) {
                    fprintf(stderr, "Timestep %e is below SMALLEST_DT_ALLOWED. Stopping here.\n", dt_host);
                    exit(1);
                }

                // get derivatives for second step (i.e., compute k2)
                // this happens at t = t0 + h/2
                substep_currentTime = currentTime + dt_host*0.5;
                cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));
                cudaVerify(cudaMemcpyToSymbol(p, &rk_device[RKFIRST], sizeof(struct Particle)));
#if GRAVITATING_POINT_MASSES
                cudaVerify(cudaMemcpyToSymbol(pointmass, &rk_pointmass_device[RKFIRST], sizeof(struct Pointmass)));
#endif
                rightHandSide();
                cudaVerify(cudaDeviceSynchronize());

                // compute
                //    q_n - h*k1 + 2*h*k2
                // and store quantities in [RKSECOND]
                cudaVerifyKernel((integrateSecondStep<<<numberOfMultiprocessors, NUM_THREADS_RK2_INTEGRATE_STEP>>>()));
                cudaVerify(cudaDeviceSynchronize());

                if (param.selfgravity) {
                    copy_gravitational_accels_device_to_device(&rk_device[RKSECOND], &rk_device[RKFIRST]);
                }

                // get derivatives for the 3rd (and last) step (i.e., compute k3)
                // this happens at t = t0 + h
                cudaVerify(cudaMemcpyToSymbol(p, &rk_device[RKSECOND], sizeof(struct Particle)));
#if GRAVITATING_POINT_MASSES
                cudaVerify(cudaMemcpyToSymbol(pointmass, &rk_pointmass_device[RKSECOND], sizeof(struct Pointmass)));
#endif
                substep_currentTime = currentTime + dt_host;
                cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));
                rightHandSide();
                cudaVerify(cudaDeviceSynchronize());

                // compute
                //    q_n+1^RK3  from k1, k2, k3 (which are stored in [RKSTART], [RKFIRST], [RKSECOND])
                // and store quantities in p
                cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
#if GRAVITATING_POINT_MASSES
                cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
#endif
                cudaVerifyKernel((integrateThirdStep<<<numberOfMultiprocessors, NUM_THREADS_RK2_INTEGRATE_STEP>>>()));
                cudaVerify(cudaDeviceSynchronize());

                // calculate errors
                // following Stephen Oxley 1999, Modelling the Capture Theory for the Origin of Planetary Systems
                cudaVerifyKernel((checkError<<<numberOfMultiprocessors, NUM_THREADS_ERRORCHECK>>>(
                                maxPosAbsErrorPerBlock, maxVelAbsErrorPerBlock
#if INTEGRATE_DENSITY
                                , maxDensityAbsErrorPerBlock
#endif
#if INTEGRATE_ENERGY
                                , maxEnergyAbsErrorPerBlock
#endif
#if PALPHA_POROSITY
                                , maxPressureAbsChangePerBlock
#endif
                                )));
                /* get info about the quality of the time step: if errorSmallEnough is TRUE, then
                   the integration is successful and the timestep size is raised. if errorSmallEnough
                   is FALSE, the timestep size is lowered and the step is repeated */
                cudaVerify(cudaDeviceSynchronize());
                cudaVerify(cudaMemcpyFromSymbol(&dtNewErrorCheck_host, dtNewErrorCheck, sizeof(double)));
                cudaVerify(cudaMemcpyFromSymbol(&errorSmallEnough_host, errorSmallEnough, sizeof(int)));

#if PALPHA_POROSITY
                /* special checks for the convergence of the p(alpha) crush curve stuff */
                if (errorSmallEnough_host) {
                    dt_alphaold = dt_host;
                    /* checking if the distention change is within the set limit */
                    cudaVerifyKernel((alphaMaxTimeStep<<<numberOfMultiprocessors, NUM_THREADS_ERRORCHECK>>>(
                                    maxalphaDiffPerBlock
                    )));
                    cudaVerify(cudaMemcpyFromSymbol(&dt_alphanew, dtNewAlphaCheck, sizeof(double)));
                    if (dt_alphanew < dt_alphaold && param.verbose && dt_alphanew > 0) {
                        fprintf(stdout, "timestep is too large for distention, reducing from %e to %e\n", dt_alphaold, dt_alphanew);
                    }
                }
                dtNewAlphaCheck_host = -1.0;
                cudaVerify(cudaDeviceSynchronize());
                cudaVerify(cudaMemcpyFromSymbol(&dtNewAlphaCheck_host, dtNewAlphaCheck, sizeof(double)));
                cudaVerify(cudaMemcpyFromSymbol(&errorSmallEnough_host, errorSmallEnough, sizeof(int)));
#endif

                /* last time step was okay, forward time and continue with new time step size */
                if (errorSmallEnough_host) {
                    currentTime += dt_host;
                    if (!param.verbose)
                        fprintf(stdout, "time: %e   last timestep: %e\n", currentTime, dt_host);
                    cudaVerifyKernel((BoundaryConditionsAfterIntegratorStep<<<numberOfMultiprocessors, NUM_THREADS_ERRORCHECK>>>(interactions)));
                    cudaVerify(cudaDeviceSynchronize());
                }

                /* print information about errors */
                if (param.verbose) {
                    double errPos, errVel, errDensity, errEnergy = 0;
                    cudaVerify(cudaMemcpyFromSymbol(&errPos, maxPosAbsError, sizeof(double)));
                    cudaVerify(cudaMemcpyFromSymbol(&errVel, maxVelAbsError, sizeof(double)));
#if INTEGRATE_DENSITY
                    cudaVerify(cudaMemcpyFromSymbol(&errDensity, maxDensityAbsError, sizeof(double)));
#endif
#if INTEGRATE_ENERGY
                    cudaVerify(cudaMemcpyFromSymbol(&errEnergy, maxEnergyAbsError, sizeof(double)));
#endif
                    cudaVerify(cudaDeviceSynchronize());
                    fprintf(stdout, "total relative max error: %g (locations: %e, velocities: %e, density: %e, energy: %e) with timestep: %e\n",
                            max(max(errPos, errVel), errDensity) / param.rk_epsrel, errPos, errVel, errDensity, errEnergy, dt_host);
#if PALPHA_POROSITY
                    fprintf(stdout, "dtNewErrorCheck: %g   dtNewAlphaCheck: %g\n", dtNewErrorCheck_host, dtNewAlphaCheck_host);
#endif
                }

                /* set suggested next timestep */
#if PALPHA_POROSITY
                if (dtNewAlphaCheck_host <= 0) {
                    dt_suggested = dtNewErrorCheck_host;
                } else {
                    dt_suggested = min(dtNewErrorCheck_host, dtNewAlphaCheck_host);
                }
#else
                dt_suggested = dtNewErrorCheck_host;
#endif
                assert(dt_suggested > 0);
#if DEBUG_TIMESTEP
                fprintf(stdout, "suggested next timestep based on errors: %e\n", dt_suggested);
#endif

                /* limit suggested next timestep to max allowed timestep */
                if (dt_suggested > dtmax_host) {
#if DEBUG_TIMESTEP
                    fprintf(stdout, "suggested next timestep is larger than max allowed timestep, reduced from %e to %e\n", dt_suggested, dtmax_host);
#endif
                    dt_suggested = dtmax_host;
                }

                if (currentTime + dt_suggested > endTime) {
                    /* if suggested next timestep would overshoot, reduce it */
                    dt_host = endTime - currentTime;
#if DEBUG_TIMESTEP
                    fprintf(stdout, "next timestep would overshoot output time, reduced from suggested %e to %e\n", dt_suggested, dt_host);
#endif
                } else {
                    /* otherwise use suggested timestep for next step */
                    dt_host = dt_suggested;
                }

                /* tell the gpu the new timestep and the current time */
                cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
                cudaVerify(cudaMemcpyToSymbol(dt, &dt_host, sizeof(double)));

                /* break loop if timestep was successful, otherwise set stage for next adaptive round */
                if (errorSmallEnough_host) {
                    afterIntegrationStep();   // do something after successful step (e.g. look for min/max pressure...)
                    if (param.verbose) {
                        fprintf(stdout, "   error was small enough, last timestep accepted, current time: %e   time to next output: %e   next timestep: %e\n",
                                currentTime, endTime-currentTime, dt_host);
                    }
                    break; // break while(TRUE) and continue with next timestep
                } else {
                    if (param.verbose)
                        fprintf(stdout, "   error was too large, last timestep rejected, current time: %e   timestep lowered to: %e\n", currentTime, dt_host);
                    // copy back the initial values of the particles
                    copy_particles_variables_device_to_device(&rk_device[RKFIRST], &rk_device[RKSTART]);
                    copy_particles_derivatives_device_to_device(&rk_device[RKFIRST], &rk_device[RKSTART]);
#if GRAVITATING_POINT_MASSES
                    copy_pointmass_variables_device_to_device(&rk_pointmass_device[RKFIRST], &rk_pointmass_device[RKSTART]);
                    copy_pointmass_derivatives_device_to_device(&rk_pointmass_device[RKFIRST], &rk_pointmass_device[RKSTART]);
#endif
                    cudaVerify(cudaDeviceSynchronize());
                }
            } // loop until error small enough
        } // current time < end time loop

        // write results
#if FRAGMENTATION
        cudaVerify(cudaDeviceSynchronize());
        cudaVerifyKernel((damageLimit<<<numberOfMultiprocessors*4, NUM_THREADS_PC_INTEGRATOR>>>()));
        cudaVerify(cudaDeviceSynchronize());
#endif
        copyToHostAndWriteToFile(timestep, lastTimestep);
    } // timestep loop


    // free memory
    int free_immutables = 0;
    for (rkstep = 0; rkstep < 3; rkstep++) {
        free_particles_memory(&rk_device[rkstep], free_immutables);
#if GRAVITATING_POINT_MASSES
        free_pointmass_memory(&rk_pointmass_device[rkstep], free_immutables);
#endif
    }
    cudaVerify(cudaFree(maxPosAbsErrorPerBlock));
    cudaVerify(cudaFree(maxVelAbsErrorPerBlock));
#if FRAGMENTATION
    cudaVerify(cudaFree(maxDamageTimeStepPerBlock));
#endif
#if INTEGRATE_ENERGY
    cudaVerify(cudaFree(maxEnergyAbsErrorPerBlock));
#endif
#if INTEGRATE_DENSITY
    cudaVerify(cudaFree(maxDensityAbsErrorPerBlock));
#endif
#if PALPHA_POROSITY
    cudaVerify(cudaFree(maxalphaDiffPerBlock));
#endif
}



__global__ void limitTimestep(double *forcesPerBlock , double *courantPerBlock)
{
    __shared__ double sharedForces[NUM_THREADS_LIMITTIMESTEP];
    __shared__ double sharedCourant[NUM_THREADS_LIMITTIMESTEP];
    int i, j, k, m;
    double forces = 1e100, courant = 1e100;
    double temp;
    double sml;
    double ax;
#if DIM > 1
    double ay;
#endif
#if DIM == 3
    double az;
#endif
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        ax = p.ax[i];
#if DIM > 1
        ay = p.ay[i];
#endif
#if DIM == 3
        az = p.az[i];
#endif
        temp = ax*ax;
#if DIM > 1
        temp += ay*ay;
#endif
#if DIM == 3
         temp += az*az;
#endif
        sml = p.h[i];
        if (temp > 0) {
            temp = sqrt(sml / sqrt(temp));
            forces = min(forces, temp);
        }
        temp = sml / p.cs[i];
        courant = min(courant, temp);
    }

    // reduce shared thread results to one per block
    i = threadIdx.x;
    sharedForces[i] = forces;
    sharedCourant[i] = courant;
    for (j = NUM_THREADS_LIMITTIMESTEP / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;
            sharedForces[i] = forces = min(forces, sharedForces[k]);
            sharedCourant[i] = courant = min(courant, sharedCourant[k]);
        }
    }

    // write block result to global memory
    if (i == 0) {
        k = blockIdx.x;
        forcesPerBlock[k] = forces;
        courantPerBlock[k] = courant;
        m = gridDim.x - 1;
        if (m == atomicInc((unsigned int *)&blockCount, m)) {
            // last block, so combine all block results
            for (j = 0; j <= m; j++) {
                forces = min(forces, forcesPerBlock[j]);
                courant = min(courant, courantPerBlock[j]);
            }
            // set new timestep
            dt = min(COURANT_FACT*courant, FORCES_FACT*forces);
#if DEBUG_TIMESTEP
            printf("<limitTimestep> max allowed timestep due to CFL is %g, due to forces/accels is %g, set timestep to %g\n",
                    COURANT_FACT*courant, FORCES_FACT*forces, dt);
#endif
            // reset block count
            blockCount = 0;
        }
    }
}



__global__ void integrateFirstStep(void)
{
    int i;

#if GRAVITATING_POINT_MASSES
    // loop for point masses
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i+= blockDim.x * gridDim.x) {
        rk_pointmass[RKFIRST].x[i] = rk_pointmass[RKSTART].x[i] + dt * B21 * rk_pointmass[RKSTART].vx[i];
# if DIM > 1
        rk_pointmass[RKFIRST].y[i] = rk_pointmass[RKSTART].y[i] + dt * B21 * rk_pointmass[RKSTART].vy[i];
# endif

# if DIM > 2
        rk_pointmass[RKFIRST].z[i] = rk_pointmass[RKSTART].z[i] + dt * B21 * rk_pointmass[RKSTART].vz[i];
# endif

        rk_pointmass[RKFIRST].vx[i] = rk_pointmass[RKSTART].vx[i] + dt * B21 * rk_pointmass[RKSTART].ax[i];
# if DIM > 1
        rk_pointmass[RKFIRST].vy[i] = rk_pointmass[RKSTART].vy[i] + dt * B21 * rk_pointmass[RKSTART].ay[i];
# endif
# if DIM > 2
        rk_pointmass[RKFIRST].vz[i] = rk_pointmass[RKSTART].vz[i] + dt * B21 * rk_pointmass[RKSTART].az[i];
# endif
    }
#endif

    // loop for particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {

        //printf("START: vx: %g \t %g :dxdt \t\t\t vy: %g \t %g :dydt\n", velxStart[i], dxdtStart[i], velyStart[i], dydtStart[i]);
#if INTEGRATE_DENSITY
        rk[RKFIRST].rho[i] = rk[RKSTART].rho[i] + dt * B21 * rk[RKSTART].drhodt[i];
#endif
#if INTEGRATE_SML
        rk[RKFIRST].h[i] = rk[RKSTART].h[i] + dt * B21 * rk[RKSTART].dhdt[i];
#else
        rk[RKFIRST].h[i] = rk[RKSTART].h[i];
#endif
#if INTEGRATE_ENERGY
        rk[RKFIRST].e[i] = rk[RKSTART].e[i] + dt * B21 * rk[RKSTART].dedt[i];
#endif
#if FRAGMENTATION
        rk[RKFIRST].d[i] = rk[RKSTART].d[i] + dt * B21 * rk[RKSTART].dddt[i];
        rk[RKFIRST].numActiveFlaws[i] = rk[RKSTART].numActiveFlaws[i];
#if PALPHA_POROSITY
        rk[RKFIRST].damage_porjutzi[i] = rk[RKSTART].damage_porjutzi[i] + dt * B21 * rk[RKSTART].ddamage_porjutzidt[i];
#endif
#endif
#if INVISCID_SPH
        rk[RKFIRST].beta[i] = rk[RKSTART].beta[i] + dt * B21 * rk[RKSTART].dbetadt[i];
#endif
#if SOLID
        int j, k;
        for (j = 0; j < DIM; j++) {
            for (k = 0; k < DIM; k++) {
                rk[RKFIRST].S[stressIndex(i,j,k)] = rk[RKSTART].S[stressIndex(i,j,k)] + dt * B21 * rk[RKSTART].dSdt[stressIndex(i,j,k)];
            }
        }
#endif
#if JC_PLASTICITY
        rk[RKFIRST].ep[i] = rk[RKSTART].ep[i] + dt * B21 * rk[RKSTART].edotp[i];
        rk[RKFIRST].T[i] = rk[RKSTART].T[i] + dt * B21 * rk[RKSTART].dTdt[i];
#endif
#if PALPHA_POROSITY
        rk[RKFIRST].alpha_jutzi[i] = rk[RKSTART].alpha_jutzi[i] + dt * B21 * rk[RKSTART].dalphadt[i];
        // rk[RKFIRST].p is the pressure at the begin of the new timestep
        // this pressure has to be compared to the pressure at the end of the timestep
        rk[RKFIRST].pold[i] = rk[RKFIRST].p[i];
#endif
#if SIRONO_POROSITY
        rk[RKFIRST].rho_0prime[i] = rk[RKSTART].rho_0prime[i];
        rk[RKFIRST].rho_c_plus[i] = rk[RKSTART].rho_c_plus[i];
        rk[RKFIRST].rho_c_minus[i] = rk[RKSTART].rho_c_minus[i];
        rk[RKFIRST].compressive_strength[i] = rk[RKSTART].compressive_strength[i];
        rk[RKFIRST].tensile_strength[i] = rk[RKSTART].tensile_strength[i];
        rk[RKFIRST].shear_strength[i] = rk[RKSTART].shear_strength[i];
        rk[RKFIRST].K[i] = rk[RKSTART].K[i];
        rk[RKFIRST].flag_rho_0prime[i] = rk[RKSTART].flag_rho_0prime[i];
        rk[RKFIRST].flag_plastic[i] = rk[RKSTART].flag_plastic[i];
#endif
#if EPSALPHA_POROSITY
        rk[RKFIRST].alpha_epspor[i] = rk[RKSTART].alpha_epspor[i] + dt * B21 * rk[RKSTART].dalpha_epspordt[i];
        rk[RKFIRST].epsilon_v[i] = rk[RKSTART].epsilon_v[i] + dt * B21 * rk[RKSTART].depsilon_vdt[i];
#endif

        rk[RKFIRST].x[i] = rk[RKSTART].x[i] + dt * B21 * rk[RKSTART].dxdt[i];
#if DIM > 1
        rk[RKFIRST].y[i] = rk[RKSTART].y[i] + dt * B21 * rk[RKSTART].dydt[i];
#endif
#if DIM > 2
        rk[RKFIRST].z[i] = rk[RKSTART].z[i] + dt * B21 * rk[RKSTART].dzdt[i];
#endif

        rk[RKFIRST].vx[i] = rk[RKSTART].vx[i] + dt * B21 * rk[RKSTART].ax[i];
#if DIM > 1
        rk[RKFIRST].vy[i] = rk[RKSTART].vy[i] + dt * B21 * rk[RKSTART].ay[i];
#endif
#if DIM > 2
        rk[RKFIRST].vz[i] = rk[RKSTART].vz[i] + dt * B21 * rk[RKSTART].az[i];
#endif
    }
}



__global__ void integrateSecondStep(void)
{
    int i;

#if GRAVITATING_POINT_MASSES
    // loop for pointmasses
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i+= blockDim.x * gridDim.x) {
        rk_pointmass[RKSECOND].vx[i] = rk_pointmass[RKSTART].vx[i] + dt * (B31 * rk_pointmass[RKSTART].ax[i] + B32 * rk_pointmass[RKFIRST].ax[i]);
# if DIM > 1
        rk_pointmass[RKSECOND].vy[i] = rk_pointmass[RKSTART].vy[i] + dt * (B31 * rk_pointmass[RKSTART].ay[i] + B32 * rk_pointmass[RKFIRST].ay[i]);
# endif
# if DIM == 3
        rk_pointmass[RKSECOND].vz[i] = rk_pointmass[RKSTART].vz[i] + dt * (B31 * rk_pointmass[RKSTART].az[i] + B32 * rk_pointmass[RKFIRST].az[i]);
# endif
        rk_pointmass[RKSECOND].x[i] = rk_pointmass[RKSTART].x[i] + dt * (B31 * rk_pointmass[RKSTART].vx[i] + B32 * rk_pointmass[RKFIRST].vx[i]);
# if DIM > 1
        rk_pointmass[RKSECOND].y[i] = rk_pointmass[RKSTART].y[i] + dt * (B31 * rk_pointmass[RKSTART].vy[i] + B32 * rk_pointmass[RKFIRST].vy[i]);
# endif
# if DIM == 3
        rk_pointmass[RKSECOND].z[i] = rk_pointmass[RKSTART].z[i] + dt * (B31 * rk_pointmass[RKSTART].vz[i] + B32 * rk_pointmass[RKFIRST].vz[i]);
# endif
    }
#endif

    // loop for particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
#if INTEGRATE_DENSITY
        rk[RKSECOND].rho[i] = rk[RKSTART].rho[i] + dt * (B31 * rk[RKSTART].drhodt[i] + B32 * rk[RKFIRST].drhodt[i]);
#endif
#if INTEGRATE_SML
        rk[RKSECOND].h[i] = rk[RKSTART].h[i] + dt * (B31 * rk[RKSTART].dhdt[i] + B32 * rk[RKFIRST].dhdt[i]);
#else
        rk[RKSECOND].h[i] = rk[RKSTART].h[i];
#endif
#if INTEGRATE_ENERGY
        rk[RKSECOND].e[i] = rk[RKSTART].e[i] + dt * (B31 * rk[RKSTART].dedt[i] + B32 * rk[RKFIRST].dedt[i]);
#endif
#if FRAGMENTATION
        rk[RKSECOND].d[i] = rk[RKSTART].d[i] + dt * (B31 * rk[RKSTART].dddt[i] + B32 * rk[RKFIRST].dddt[i]);
        rk[RKSECOND].numActiveFlaws[i] = rk[RKFIRST].numActiveFlaws[i];
#if PALPHA_POROSITY
        rk[RKSECOND].damage_porjutzi[i] = rk[RKSTART].damage_porjutzi[i] + dt * (B31 * rk[RKSTART].ddamage_porjutzidt[i] + B32 * rk[RKFIRST].ddamage_porjutzidt[i]);
#endif
#endif
#if JC_PLASTICITY
        rk[RKSECOND].ep[i] = rk[RKSTART].ep[i] + dt * (B31 * rk[RKSTART].edotp[i] + B32 * rk[RKFIRST].edotp[i]);
        rk[RKSECOND].T[i] = rk[RKSTART].T[i] + dt * (B31 * rk[RKSTART].dTdt[i] + B32 * rk[RKFIRST].dTdt[i]);
#endif
#if PALPHA_POROSITY
        rk[RKSECOND].alpha_jutzi[i] = rk[RKSTART].alpha_jutzi[i] + dt * (B31 * rk[RKSTART].dalphadt[i] + B32 * rk[RKFIRST].dalphadt[i]);
        rk[RKSECOND].pold[i] = rk[RKFIRST].pold[i];
#endif
#if SIRONO_POROSITY
        rk[RKSECOND].rho_0prime[i] = rk[RKFIRST].rho_0prime[i];
        rk[RKSECOND].rho_c_plus[i] = rk[RKFIRST].rho_c_plus[i];
        rk[RKSECOND].rho_c_minus[i] = rk[RKFIRST].rho_c_minus[i];
        rk[RKSECOND].compressive_strength[i] = rk[RKFIRST].compressive_strength[i];
        rk[RKSECOND].tensile_strength[i] = rk[RKFIRST].tensile_strength[i];
        rk[RKSECOND].shear_strength[i] = rk[RKFIRST].shear_strength[i];
        rk[RKSECOND].K[i] = rk[RKFIRST].K[i];
        rk[RKSECOND].flag_rho_0prime[i] = rk[RKFIRST].flag_rho_0prime[i];
        rk[RKSECOND].flag_plastic[i] = rk[RKFIRST].flag_plastic[i];
#endif
#if EPSALPHA_POROSITY
        rk[RKSECOND].alpha_epspor[i] = rk[RKSTART].alpha_epspor[i] + dt * (B31 * rk[RKSTART].dalpha_epspordt[i] + B32 * rk[RKFIRST].dalpha_epspordt[i]);
        rk[RKSECOND].epsilon_v[i] = rk[RKSTART].epsilon_v[i] + dt * (B31 * rk[RKSTART].depsilon_vdt[i] + B32 * rk[RKFIRST].depsilon_vdt[i]);
#endif
#if INVISCID_SPH
        rk[RKSECOND].beta[i] = rk[RKSTART].beta[i] + dt * (B31 * rk[RKSTART].dbetadt[i] + B32 * rk[RKFIRST].dbetadt[i]);
#endif
#if SOLID
        int j;
        for (j = 0; j < DIM*DIM; j++) {
            rk[RKSECOND].S[i*DIM*DIM+j] = rk[RKSTART].S[i*DIM*DIM+j] + dt * (B31 * rk[RKSTART].dSdt[i*DIM*DIM+j] + B32 * rk[RKFIRST].dSdt[i*DIM*DIM+j]);
        }
#endif

        rk[RKSECOND].vx[i] = rk[RKSTART].vx[i] + dt * (B31 * rk[RKSTART].ax[i] + B32 * rk[RKFIRST].ax[i]);
#if DIM > 1
        rk[RKSECOND].vy[i] = rk[RKSTART].vy[i] + dt * (B31 * rk[RKSTART].ay[i] + B32 * rk[RKFIRST].ay[i]);
#endif
#if DIM == 3
        rk[RKSECOND].vz[i] = rk[RKSTART].vz[i] + dt * (B31 * rk[RKSTART].az[i] + B32 * rk[RKFIRST].az[i]);
#endif

        rk[RKSECOND].x[i] = rk[RKSTART].x[i] + dt * (B31 * rk[RKSTART].dxdt[i] + B32 * rk[RKFIRST].dxdt[i]);
#if DIM > 1
        rk[RKSECOND].y[i] = rk[RKSTART].y[i] + dt * (B31 * rk[RKSTART].dydt[i] + B32 * rk[RKFIRST].dydt[i]);
#endif
#if DIM == 3
        rk[RKSECOND].z[i] = rk[RKSTART].z[i] + dt * (B31 * rk[RKSTART].dzdt[i] + B32 * rk[RKFIRST].dzdt[i]);
#endif
    }
}



__global__ void integrateThirdStep(void)
{
    int i;
    int d;

#if GRAVITATING_POINT_MASSES
    // loop for pointmasses
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i+= blockDim.x * gridDim.x) {
        pointmass.vx[i] = rk_pointmass[RKSTART].vx[i] + dt/6.0 * (C1 * rk_pointmass[RKSTART].ax[i] + C2 * rk_pointmass[RKFIRST].ax[i] + C3 * rk_pointmass[RKSECOND].ax[i]);
        pointmass.ax[i] = 1./6.0 *(C1 * rk_pointmass[RKSTART].ax[i] + C2 * rk_pointmass[RKFIRST].ax[i] + C3 * rk_pointmass[RKSECOND].ax[i]);
# if DIM > 1
        pointmass.vy[i] = rk_pointmass[RKSTART].vy[i] + dt/6.0 * (C1 * rk_pointmass[RKSTART].ay[i] + C2 * rk_pointmass[RKFIRST].ay[i] + C3 * rk_pointmass[RKSECOND].ay[i]);
        pointmass.ay[i] = 1./6.0 *(C1 * rk_pointmass[RKSTART].ay[i] + C2 * rk_pointmass[RKFIRST].ay[i] + C3 * rk_pointmass[RKSECOND].ay[i]);
# endif
# if DIM > 2
        pointmass.vz[i] = rk_pointmass[RKSTART].vz[i] + dt/6.0 * (C1 * rk_pointmass[RKSTART].az[i] + C2 * rk_pointmass[RKFIRST].az[i] + C3 * rk_pointmass[RKSECOND].az[i]);
        pointmass.az[i] = 1./6.0 *(C1 * rk_pointmass[RKSTART].az[i] + C2 * rk_pointmass[RKFIRST].az[i] + C3 * rk_pointmass[RKSECOND].az[i]);
# endif

        pointmass.x[i] = rk_pointmass[RKSTART].x[i] + dt/6.0 * (C1 * rk_pointmass[RKSTART].vx[i] + C2 * rk_pointmass[RKFIRST].vx[i] + C3 * rk_pointmass[RKSECOND].vx[i]);
# if DIM > 1
        pointmass.y[i] = rk_pointmass[RKSTART].y[i] + dt/6.0 * (C1 * rk_pointmass[RKSTART].vy[i] + C2 * rk_pointmass[RKFIRST].vy[i] + C3 * rk_pointmass[RKSECOND].vy[i]);
# endif
# if DIM > 2
        pointmass.z[i] = rk_pointmass[RKSTART].z[i] + dt/6.0 * (C1 * rk_pointmass[RKSTART].vz[i] + C2 * rk_pointmass[RKFIRST].vz[i] + C3 * rk_pointmass[RKSECOND].vz[i]);
# endif
    }
#endif

    // loop for particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        //printf("THIRD: vx: %g \t %g :dxdt \t\t\t vy: %g \t %g :dydt\n", velxSecond[i], dxdtSecond[i], velySecond[i], dydtSecond[i]);
#if INTEGRATE_DENSITY
        p.rho[i] = rk[RKSTART].rho[i] + dt/6.0 *
            (    C1 * rk[RKSTART].drhodt[i]
               + C2 * rk[RKFIRST].drhodt[i]
               + C3 * rk[RKSECOND].drhodt[i]);
        p.drhodt[i] = 1./6.*(C1 * rk[RKSTART].drhodt[i]
               + C2 * rk[RKFIRST].drhodt[i]
               + C3 * rk[RKSECOND].drhodt[i]);
#else
        p.rho[i] = rk[RKSECOND].rho[i];
#endif

#if INTEGRATE_SML
        p.h[i] = rk[RKSTART].h[i] + dt/6.0 *
            (    C1 * rk[RKSTART].dhdt[i]
               + C2 * rk[RKFIRST].dhdt[i]
               + C3 * rk[RKSECOND].dhdt[i]);
        p.dhdt[i] = 1./6.*(C1 * rk[RKSTART].dhdt[i]
               + C2 * rk[RKFIRST].dhdt[i]
               + C3 * rk[RKSECOND].dhdt[i]);
#else
        p.h[i] = rk[RKSECOND].h[i];
#endif

#if INTEGRATE_ENERGY
        p.e[i] = rk[RKSTART].e[i] + dt/6.0 *
            (    C1 * rk[RKSTART].dedt[i]
               + C2 * rk[RKFIRST].dedt[i]
               + C3 * rk[RKSECOND].dedt[i]);
        p.dedt[i] = 1./6.* (C1 * rk[RKSTART].dedt[i]
               + C2 * rk[RKFIRST].dedt[i]
               + C3 * rk[RKSECOND].dedt[i]);
#endif

#if PALPHA_POROSITY
        double dp = rk[RKSECOND].p[i] - rk[RKSTART].p[i];
#endif

#if FRAGMENTATION
        p.d[i] = rk[RKSTART].d[i] + dt/6.0 *
            (    C1 * rk[RKSTART].dddt[i]
               + C2 * rk[RKFIRST].dddt[i]
               + C3 * rk[RKSECOND].dddt[i]);
        p.dddt[i] = 1./6. * (C1 * rk[RKSTART].dddt[i]
               + C2 * rk[RKFIRST].dddt[i]
               + C3 * rk[RKSECOND].dddt[i]);
# if PALPHA_POROSITY
        if (dp > 0.0) {
            p.damage_porjutzi[i] = rk[RKSTART].damage_porjutzi[i] + dt/6.0 *
                (    C1 * rk[RKSTART].ddamage_porjutzidt[i]
                   + C2 * rk[RKFIRST].ddamage_porjutzidt[i]
                   + C3 * rk[RKSECOND].ddamage_porjutzidt[i]);
            p.ddamage_porjutzidt[i] = 1./6. * (C1 * rk[RKSTART].ddamage_porjutzidt[i]
                   + C2 * rk[RKFIRST].ddamage_porjutzidt[i]
                   + C3 * rk[RKSECOND].ddamage_porjutzidt[i]);
        } else {
            p.d[i] = p.d[i];
            p.damage_porjutzi[i] = rk[RKSTART].damage_porjutzi[i];
        }
# endif
#endif

#if JC_PLASTICITY
        p.ep[i] = rk[RKSTART].ep[i] + dt/6.0 *
            (    C1 * rk[RKSTART].edotp[i]
               + C2 * rk[RKFIRST].edotp[i]
               + C3 * rk[RKSECOND].edotp[i]);
        p.T[i] = rk[RKSTART].T[i] + dt/6.0 *
            (    C1 * rk[RKSTART].dTdt[i]
               + C2 * rk[RKFIRST].dTdt[i]
               + C3 * rk[RKSECOND].dTdt[i]);
        p.edotp[i] = 1./6. * ( C1 * rk[RKSTART].edotp[i]
               + C2 * rk[RKFIRST].edotp[i]
               + C3 * rk[RKSECOND].edotp[i]);
        p.dTdt[i] =  1./6. * ( C1 * rk[RKSTART].dTdt[i]
               + C2 * rk[RKFIRST].dTdt[i]
               + C3 * rk[RKSECOND].dTdt[i]);
#endif

#if PALPHA_POROSITY
        if (dp > 0.0) {
            p.alpha_jutzi[i] = rk[RKSTART].alpha_jutzi[i] + dt/6.0 *
                (    C1 * rk[RKSTART].dalphadt[i]
                   + C2 * rk[RKFIRST].dalphadt[i]
                   + C3 * rk[RKSECOND].dalphadt[i]);
            p.dalphadt[i] = 1./6. * (C1 * rk[RKSTART].dalphadt[i]
                   + C2 * rk[RKFIRST].dalphadt[i]
                   + C3 * rk[RKSECOND].dalphadt[i]);
        } else {
            p.alpha_jutzi[i] = rk[RKSTART].alpha_jutzi[i];
        }
#endif

#if EPSALPHA_POROSITY
        p.alpha_epspor[i] = rk[RKSTART].alpha_epspor[i] + dt/6.0 *
                (     C1 * rk[RKSTART].dalpha_epspordt[i]
                    + C2 * rk[RKFIRST].dalpha_epspordt[i]
                    + C3 * rk[RKSECOND].dalpha_epspordt[i]);
        p.dalpha_epspordt[i] = 1./6. *
                (     C1 * rk[RKSTART].dalpha_epspordt[i]
                    + C2 * rk[RKFIRST].dalpha_epspordt[i]
                    + C3 * rk[RKSECOND].dalpha_epspordt[i]);
        p.epsilon_v[i] = rk[RKSTART].epsilon_v[i] + dt/6.0 *
                (     C1 * rk[RKSTART].depsilon_vdt[i]
                    + C2 * rk[RKFIRST].depsilon_vdt[i]
                    + C3 * rk[RKSECOND].depsilon_vdt[i]);
        p.depsilon_vdt[i] = 1./6. *
                (     C1 * rk[RKSTART].depsilon_vdt[i]
                    + C2 * rk[RKFIRST].depsilon_vdt[i]
                    + C3 * rk[RKSECOND].depsilon_vdt[i]);
#endif

#if INVISCID_SPH
        p.beta[i] = rk[RKSTART].beta[i] + dt/6.0 *
            (     C1 * rk[RKSTART].dbetadt[i]
                + C2 * rk[RKFIRST].dbetadt[i]
                + C3 * rk[RKSECOND].dbetadt[i]);
        p.dbetadt[i] = 1./6. * (C1 * rk[RKSTART].dbetadt[i]
                             +  C2 * rk[RKFIRST].dbetadt[i]
                             +  C3 * rk[RKSECOND].dbetadt[i]);
#endif

#if SOLID
        int j;
        for (j = 0; j < DIM*DIM; j++) {
            p.S[i*DIM*DIM+j] = rk[RKSTART].S[i*DIM*DIM+j] + dt/6.0 *
                (    C1 * rk[RKSTART].dSdt[i*DIM*DIM+j]
                   + C2 * rk[RKFIRST].dSdt[i*DIM*DIM+j]
                   + C3 * rk[RKSECOND].dSdt[i*DIM*DIM+j]);
            p.dSdt[i*DIM*DIM+j] = 1./6. *
                (    C1 * rk[RKSTART].dSdt[i*DIM*DIM+j]
                   + C2 * rk[RKFIRST].dSdt[i*DIM*DIM+j]
                   + C3 * rk[RKSECOND].dSdt[i*DIM*DIM+j]);
        }
#endif

        p.vx[i] = rk[RKSTART].vx[i] + dt/6.0 * (C1 * rk[RKSTART].ax[i] + C2 * rk[RKFIRST].ax[i] + C3 * rk[RKSECOND].ax[i]);
        p.ax[i] = 1./6.0 *(C1 * rk[RKSTART].ax[i] + C2 * rk[RKFIRST].ax[i] + C3 * rk[RKSECOND].ax[i]);
        p.g_ax[i] = 1./6.0 *(C1 * rk[RKSTART].g_ax[i] + C2 * rk[RKFIRST].g_ax[i] + C3 * rk[RKSECOND].g_ax[i]);
#if DIM > 1
        p.vy[i] = rk[RKSTART].vy[i] + dt/6.0 * (C1 * rk[RKSTART].ay[i] + C2 * rk[RKFIRST].ay[i] + C3 * rk[RKSECOND].ay[i]);
        p.ay[i] = 1./6.0 *(C1 * rk[RKSTART].ay[i] + C2 * rk[RKFIRST].ay[i] + C3 * rk[RKSECOND].ay[i]);
        p.g_ay[i] = 1./6.0 *(C1 * rk[RKSTART].g_ay[i] + C2 * rk[RKFIRST].g_ay[i] + C3 * rk[RKSECOND].g_ay[i]);
#endif
#if DIM > 2
        p.vz[i] = rk[RKSTART].vz[i] + dt/6.0 * (C1 * rk[RKSTART].az[i] + C2 * rk[RKFIRST].az[i] + C3 * rk[RKSECOND].az[i]);
        p.az[i] = 1./6.0 *(C1 * rk[RKSTART].az[i] + C2 * rk[RKFIRST].az[i] + C3 * rk[RKSECOND].az[i]);
        p.g_az[i] = 1./6.0 *(C1 * rk[RKSTART].g_az[i] + C2 * rk[RKFIRST].g_az[i] + C3 * rk[RKSECOND].g_az[i]);
#endif

        p.x[i] = rk[RKSTART].x[i] + dt/6.0 * (C1 * rk[RKSTART].dxdt[i] + C2 * rk[RKFIRST].dxdt[i] + C3 * rk[RKSECOND].dxdt[i]);
#if DIM > 1
        p.y[i] = rk[RKSTART].y[i] + dt/6.0 * (C1 * rk[RKSTART].dydt[i] + C2 * rk[RKFIRST].dydt[i] + C3 * rk[RKSECOND].dydt[i]);
#endif
#if DIM > 2
        p.z[i] = rk[RKSTART].z[i] + dt/6.0 * (C1 * rk[RKSTART].dzdt[i] + C2 * rk[RKFIRST].dzdt[i] + C3 * rk[RKSECOND].dzdt[i]);
#endif

        /* remember some more values */
        p.noi[i] = rk[RKSECOND].noi[i];
        p.p[i] = rk[RKSECOND].p[i];
#if PALPHA_POROSITY
        p.pold[i] = rk[RKSECOND].p[i];
#endif
#if SIRONO_POROSITY
        p.rho_0prime[i] = rk[RKSECOND].rho_0prime[i];
        p.rho_c_plus[i] = rk[RKSECOND].rho_c_plus[i];
        p.rho_c_minus[i] = rk[RKSECOND].rho_c_minus[i];
        p.compressive_strength[i] = rk[RKSECOND].compressive_strength[i];
        p.tensile_strength[i] = rk[RKSECOND].tensile_strength[i];
        p.shear_strength[i] = rk[RKSECOND].shear_strength[i];
        p.K[i] = rk[RKSECOND].K[i];
        p.flag_rho_0prime[i] = rk[RKSECOND].flag_rho_0prime[i];
        p.flag_plastic[i] = rk[RKSECOND].flag_plastic[i];
#endif
        p.cs[i] = rk[RKSECOND].cs[i];
#if FRAGMENTATION
        p.numActiveFlaws[i] = rk[RKSECOND].numActiveFlaws[i];
#endif
#if SOLID
        p.local_strain[i] = rk[RKSECOND].local_strain[i];
#endif
#if NAVIER_STOKES
        for (d = 0; d < DIM*DIM; d++) {
            p.Tshear[i*DIM*DIM+d] = rk[RKSECOND].Tshear[i*DIM*DIM+d];
        }
#endif

#if 0
#warning experimental superstuff in rk2adaptive...
        if (p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
            p.x[i] = 1e12+1e6*i;
            p.y[i] = 1e12+1e6*i;
        }
#endif
    }
}



#if FRAGMENTATION
#define MAX_DAMAGE_CHANGE 1e-2
__global__ void damageMaxTimeStep(double *maxDamageTimeStepPerBlock)
{
    __shared__ double sharedMaxDamageTimeStep[NUM_THREADS_ERRORCHECK];
    double localMaxDamageTimeStep = 0;
    double tmp = 0;
    double dtsuggested = 0;
    int i, j, k, m;

    // loop for particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        if (rk[RKFIRST].dddt[i] > 0) {
            tmp = 1./ ( (rk[RKFIRST].d[i] + MAX_DAMAGE_CHANGE) / rk[RKFIRST].dddt[i] );
            localMaxDamageTimeStep = max(tmp, localMaxDamageTimeStep);
        }
    }

    // reduce shared thread results to one per block
    i = threadIdx.x;
    sharedMaxDamageTimeStep[i] = localMaxDamageTimeStep;
    for (j = NUM_THREADS_ERRORCHECK / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;
            sharedMaxDamageTimeStep[i] = localMaxDamageTimeStep = max(localMaxDamageTimeStep, sharedMaxDamageTimeStep[k]);
        }
    }

    // write block result to global memory
    if (i == 0) {
        k = blockIdx.x;
        maxDamageTimeStepPerBlock[k] = localMaxDamageTimeStep;
        m = gridDim.x - 1;
        if (m == atomicInc((unsigned int *)&blockCount, m)) {
            // last block, so combine all block results
            for (j = 0; j <= m; j++) {
                localMaxDamageTimeStep = max(localMaxDamageTimeStep, maxDamageTimeStepPerBlock[j]);
            }
            maxDamageTimeStep = localMaxDamageTimeStep;
            // reset block count
            blockCount = 0;

            if (maxDamageTimeStep > 0) {
                dtsuggested = 1./maxDamageTimeStep;
                if (dtsuggested < dt)
                    dt = dtsuggested;
            }
        }
    }
}
#endif



#if PALPHA_POROSITY
#define MAX_ALPHA_CHANGE 1e-4
__global__ void alphaMaxTimeStep(double *maxalphaDiffPerBlock)
{
    __shared__ double sharedMaxalphaDiff[NUM_THREADS_ERRORCHECK];
    double localMaxalphaDiff = 0.0;
    double tmp = 0.0;
    int i, j, k, m;

    maxalphaDiff = 0.0;
    dtNewAlphaCheck = -1.0;

    // loop for particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        tmp = fabs(rk[RKSTART].alpha_jutzi_old[i] - p.alpha_jutzi[i]);
        localMaxalphaDiff = max(tmp, localMaxalphaDiff);
    }

    // reduce shared thread results to one per block
    i = threadIdx.x;
    sharedMaxalphaDiff[i] = localMaxalphaDiff;
    for (j = NUM_THREADS_ERRORCHECK / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;
            sharedMaxalphaDiff[i] = localMaxalphaDiff = max(localMaxalphaDiff, sharedMaxalphaDiff[k]);
        }
    }

    // write block result to global memory
    if (i == 0) {
        k = blockIdx.x;
        maxalphaDiffPerBlock[k] = localMaxalphaDiff;
        m = gridDim.x - 1;
        if (m == atomicInc((unsigned int *)&blockCount, m)) {
            // last block, so combine all block results
            for (j = 0; j <= m; j++) {
                localMaxalphaDiff = max(localMaxalphaDiff, maxalphaDiffPerBlock[j]);
            }
            maxalphaDiff = localMaxalphaDiff;
            // reset block count
            blockCount = 0;
        }

#define FIXMEDT 1000
        /* maybe needs a smoother implementation - also if timestep gets too small because of alpha
           set it to 5e-14. It's a temporary fix for cases where dtNewAlphaCheck gets too low and crashes */
        dtNewAlphaCheck = FIXMEDT*dt;
//        dtNewAlphaCheck = dt * MAX_ALPHA_CHANGE / (maxalphaDiff);
        if (maxalphaDiff > MAX_ALPHA_CHANGE) {
            dtNewAlphaCheck = dt * MAX_ALPHA_CHANGE / (maxalphaDiff * 1.51);
            errorSmallEnough = FALSE;
//            if (dtNewAlphaCheck < 1e-29) {
//                dtNewAlphaCheck = 1e-29;
//                errorSmallEnough = TRUE;
//                printf("Timestep too small: %g Old Timestep: %g Alpha Change: %g Max Allowed: %g \n", dtNewAlphaCheck, dt, maxalphaDiff, MAX_ALPHA_CHANGE);
//            }
        }
    }
}
#endif



__global__ void checkError(double *maxPosAbsErrorPerBlock, double *maxVelAbsErrorPerBlock
#if INTEGRATE_DENSITY
        , double *maxDensityAbsErrorPerBlock
#endif
#if INTEGRATE_ENERGY
        , double *maxEnergyAbsErrorPerBlock
#endif
#if PALPHA_POROSITY
        , double *maxPressureAbsChangePerBlock
#endif
        )
{
    __shared__ double sharedMaxPosAbsError[NUM_THREADS_ERRORCHECK];
    __shared__ double sharedMaxVelAbsError[NUM_THREADS_ERRORCHECK];
#if INTEGRATE_DENSITY
    __shared__ double sharedMaxDensityAbsError[NUM_THREADS_ERRORCHECK];
    double localMaxDensityAbsError = 0;
#endif
#if INTEGRATE_ENERGY
    __shared__ double sharedMaxEnergyAbsError[NUM_THREADS_ERRORCHECK];
    double localMaxEnergyAbsError = 0;
    int hasEnergy = 0;
#endif
#if PALPHA_POROSITY
    __shared__ double sharedMaxPressureAbsChange[NUM_THREADS_ERRORCHECK];
    double localMaxPressureAbsChange = 0;
#endif
    int i, j, k, m;
    double posAbsErrorTemp = 0, velAbsErrorTemp = 0, temp = 0, dtNew = 0;
    double localMaxPosAbsError = 0, localMaxVelAbsError = 0;
    double tmp_vel = 0.0;
    double tmp_vel2 = 0.0;
    double tmp_pos = 0.0;
    double tmp_pos2 = 0.0;
    double min_pos_change_rk2 = 0.0;


#if GRAVITATING_POINT_MASSES
    // loop for pointmasses
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i+= blockDim.x * gridDim.x) {
        temp = dt * (rk_pointmass[RKFIRST].ax[i]/3.0 - (rk_pointmass[RKSTART].ax[i] + rk_pointmass[RKSECOND].ax[i])/6.0);
        tmp_vel = fabs(rk_pointmass[RKSTART].vx[i]) + fabs(dt*rk_pointmass[RKSTART].ax[i]);
        if (tmp_vel > MIN_VEL_CHANGE_RK2) {
            tmp_vel2 = fabs(temp) / tmp_vel;
            velAbsErrorTemp = tmp_vel2;
        }
# if DIM > 1
        temp = dt * (rk_pointmass[RKFIRST].ay[i]/3.0 - (rk_pointmass[RKSTART].ay[i] + rk_pointmass[RKSECOND].ay[i])/6.0);
        tmp_vel = fabs(rk_pointmass[RKSTART].vy[i]) + fabs(dt*rk_pointmass[RKSTART].ay[i]);
        if (tmp_vel > MIN_VEL_CHANGE_RK2) {
            tmp_vel2 = fabs(temp) / tmp_vel;
            velAbsErrorTemp = max(velAbsErrorTemp, tmp_vel2);
        }
# endif
# if DIM == 3
        temp = dt * (rk_pointmass[RKFIRST].az[i]/3.0 - (rk_pointmass[RKSTART].az[i] + rk_pointmass[RKSECOND].az[i])/6.0);
        tmp_vel = fabs(rk_pointmass[RKSTART].vz[i]) + fabs(dt*rk_pointmass[RKSTART].az[i]);
        if (tmp_vel > MIN_VEL_CHANGE_RK2) {
            tmp_vel2 = fabs(temp) / tmp_vel;
            velAbsErrorTemp = max(velAbsErrorTemp, tmp_vel2);
        }
# endif
        localMaxVelAbsError = max(localMaxVelAbsError, velAbsErrorTemp);
    }
#endif

    // loop for particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        if (p_rhs.materialId[i] == EOS_TYPE_IGNORE) continue;

        temp = dt * (rk[RKFIRST].dxdt[i]/3.0 - (rk[RKSTART].dxdt[i] + rk[RKSECOND].dxdt[i])/6.0);
        tmp_pos = fabs(rk[RKSTART].x[i]) + fabs(dt*rk[RKSTART].dxdt[i]);
        min_pos_change_rk2 = rk[RKSTART].h[i];
        min_pos_change_rk2 *= RK2_LOCATION_SAFETY;
        if (tmp_pos > min_pos_change_rk2) {
            posAbsErrorTemp = fabs(temp) / tmp_pos;
        }
#if DIM > 1
        temp = dt * (rk[RKFIRST].dydt[i]/3.0 - (rk[RKSTART].dydt[i] + rk[RKSECOND].dydt[i])/6.0);
        tmp_pos = fabs(rk[RKSTART].y[i]) + fabs(dt*rk[RKSTART].dydt[i]);
        if (tmp_pos > min_pos_change_rk2) {
            tmp_pos2 = fabs(temp) / tmp_pos;
            posAbsErrorTemp = max(posAbsErrorTemp,tmp_pos2);
        }
#endif
#if DIM > 2
        temp = dt * (rk[RKFIRST].dzdt[i]/3.0 - (rk[RKSTART].dzdt[i] + rk[RKSECOND].dzdt[i])/6.0);
        tmp_pos = fabs(rk[RKSTART].z[i]) + fabs(dt*rk[RKSTART].dzdt[i]);
        if (tmp_pos > min_pos_change_rk2) {
            tmp_pos2 = fabs(temp) / tmp_pos;
            posAbsErrorTemp = max(posAbsErrorTemp,tmp_pos2);
        }
#endif
        localMaxPosAbsError = max(localMaxPosAbsError, posAbsErrorTemp);

        temp = dt * (rk[RKFIRST].ax[i]/3.0 - (rk[RKSTART].ax[i] + rk[RKSECOND].ax[i])/6.0);
        tmp_vel = fabs(rk[RKSTART].vx[i]) + fabs(dt*rk[RKSTART].ax[i]);
        if (tmp_vel > MIN_VEL_CHANGE_RK2) {
            tmp_vel2 = fabs(temp) / tmp_vel;
            velAbsErrorTemp = tmp_vel2;
        }
#if DIM > 1
        temp = dt * (rk[RKFIRST].ay[i]/3.0 - (rk[RKSTART].ay[i] + rk[RKSECOND].ay[i])/6.0);
        tmp_vel = fabs(rk[RKSTART].vy[i]) + fabs(dt*rk[RKSTART].ay[i]);
        if (tmp_vel > MIN_VEL_CHANGE_RK2) {
            tmp_vel2 = fabs(temp) / tmp_vel;
            velAbsErrorTemp = max(velAbsErrorTemp, tmp_vel2);
        }
#endif
#if DIM == 3
        temp = dt * (rk[RKFIRST].az[i]/3.0 - (rk[RKSTART].az[i] + rk[RKSECOND].az[i])/6.0);
        tmp_vel = fabs(rk[RKSTART].vz[i]) + fabs(dt*rk[RKSTART].az[i]);
        if (tmp_vel > MIN_VEL_CHANGE_RK2) {
            tmp_vel2 = fabs(temp) / tmp_vel;
            velAbsErrorTemp = max(velAbsErrorTemp, tmp_vel2);
        }
#endif
        localMaxVelAbsError = max(localMaxVelAbsError, velAbsErrorTemp);

#if INTEGRATE_DENSITY
        temp = dt * (rk[RKFIRST].drhodt[i]/3.0 - (rk[RKSTART].drhodt[i] + rk[RKSECOND].drhodt[i])/6.0);
        temp = fabs(temp) / (fabs(rk[RKSTART].rho[i]) + fabs(dt*rk[RKSTART].drhodt[i]) + TINY_RK2);
        localMaxDensityAbsError = max(localMaxDensityAbsError, temp);
#endif

#if PALPHA_POROSITY
        // check if the pressure changes too much
        temp = fabs(rk[RKFIRST].p[i] - rk[RKSECOND].p[i]);
        localMaxPressureAbsChange = max(localMaxPressureAbsChange, temp);
#endif

#if INTEGRATE_ENERGY
        hasEnergy = 0;

        switch  (matEOS[p_rhs.materialId[i]]) {
            case (EOS_TYPE_TILLOTSON):
                hasEnergy = 1;
                break;
            case (EOS_TYPE_JUTZI):
                hasEnergy = 1;
                break;
			case (EOS_TYPE_JUTZI_ANEOS):
				hasEnergy = 1;
				break;
            case (EOS_TYPE_SIRONO):
                hasEnergy = 1;
                break;
            case (EOS_TYPE_EPSILON):
                hasEnergy = 1;
                break;
            case (EOS_TYPE_ANEOS):
                hasEnergy = 1;
                break;
            case (EOS_TYPE_IDEAL_GAS):
                hasEnergy = 1;
                break;
            default:
                hasEnergy = 0;
                break;
        }
        if (hasEnergy) {
            temp = dt * (rk[RKFIRST].dedt[i]/3.0 - (rk[RKSTART].dedt[i] + rk[RKSECOND].dedt[i])/6.0);
            temp = fabs(temp) / (fabs(rk[RKSTART].e[i]) + fabs(dt*rk[RKSTART].dedt[i]) + TINY_RK2);
            localMaxEnergyAbsError = max(localMaxEnergyAbsError, temp);
        }
#endif
    }   // loop for particles

    // reduce shared thread results to one per block
    i = threadIdx.x;
    sharedMaxPosAbsError[i] = localMaxPosAbsError;
    sharedMaxVelAbsError[i] = localMaxVelAbsError;
#if INTEGRATE_DENSITY
    sharedMaxDensityAbsError[i] = localMaxDensityAbsError;
#endif
#if INTEGRATE_ENERGY
    sharedMaxEnergyAbsError[i] = localMaxEnergyAbsError;
#endif
#if PALPHA_POROSITY
    sharedMaxPressureAbsChange[i] = localMaxPressureAbsChange;
#endif
    for (j = NUM_THREADS_ERRORCHECK / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;
            sharedMaxPosAbsError[i] = localMaxPosAbsError = max(localMaxPosAbsError, sharedMaxPosAbsError[k]);
            sharedMaxVelAbsError[i] = localMaxVelAbsError = max(localMaxVelAbsError, sharedMaxVelAbsError[k]);
#if INTEGRATE_DENSITY
            sharedMaxDensityAbsError[i] = localMaxDensityAbsError = max(localMaxDensityAbsError, sharedMaxDensityAbsError[k]);
#endif
#if INTEGRATE_ENERGY
            sharedMaxEnergyAbsError[i] = localMaxEnergyAbsError = max(localMaxEnergyAbsError, sharedMaxEnergyAbsError[k]);
#endif
#if PALPHA_POROSITY
            sharedMaxPressureAbsChange[i] = localMaxPressureAbsChange = max(localMaxPressureAbsChange, sharedMaxPressureAbsChange[k]);
#endif
        }
    }

    // write block result to global memory
    if (i == 0) {
        k = blockIdx.x;
        maxPosAbsErrorPerBlock[k] = localMaxPosAbsError;
        maxVelAbsErrorPerBlock[k] = localMaxVelAbsError;
#if INTEGRATE_DENSITY
        maxDensityAbsErrorPerBlock[k] = localMaxDensityAbsError;
#endif
#if INTEGRATE_ENERGY
        maxEnergyAbsErrorPerBlock[k] = localMaxEnergyAbsError;
#endif
#if PALPHA_POROSITY
        maxPressureAbsChangePerBlock[k] = localMaxPressureAbsChange;
#endif
        m = gridDim.x - 1;
        if (m == atomicInc((unsigned int *)&blockCount, m)) {
            // last block, so combine all block results
            for (j = 0; j <= m; j++) {
                localMaxPosAbsError = max(localMaxPosAbsError, maxPosAbsErrorPerBlock[j]);
                localMaxVelAbsError = max(localMaxVelAbsError, maxVelAbsErrorPerBlock[j]);
#if INTEGRATE_DENSITY
                localMaxDensityAbsError = max(localMaxDensityAbsError, maxDensityAbsErrorPerBlock[j]);
#endif
#if INTEGRATE_ENERGY
                localMaxEnergyAbsError = max(localMaxEnergyAbsError, maxEnergyAbsErrorPerBlock[j]);
#endif
#if PALPHA_POROSITY
                localMaxPressureAbsChange = max(localMaxPressureAbsChange, maxPressureAbsChangePerBlock[j]);
#endif
            }
            maxPosAbsError = localMaxPosAbsError;
            maxVelAbsError = localMaxVelAbsError;
            temp = max(localMaxPosAbsError, localMaxVelAbsError); // relative total max error
#if INTEGRATE_DENSITY
            maxDensityAbsError = localMaxDensityAbsError;
            temp = max(temp, localMaxDensityAbsError);
#endif
#if INTEGRATE_ENERGY
            maxEnergyAbsError = localMaxEnergyAbsError;
// we neglect the error from the energy integration
//            temp = max(temp, localMaxEnergyAbsError);
#endif
#if PALPHA_POROSITY
            maxPressureAbsChange = localMaxPressureAbsChange;
#endif
            temp /= rk_epsrel_d; // total error
            if (temp > 1 && maxPressureAbsChange > max_abs_pressure_change) {
                printf("pressure changes too much, maximum allowed change is %e, current registered change was %e, reducing time step\n", max_abs_pressure_change, maxPressureAbsChange);
                temp = 1.1;
            }
            if (temp > 1) { // error too large
                errorSmallEnough = FALSE;
                dtNew = max( 0.1*dt, dt*RK2_TIMESTEP_SAFETY*pow(temp,-0.25) );
            } else { // error small enough
                errorSmallEnough = TRUE;
                dtNew = dt * RK2_TIMESTEP_SAFETY * pow(temp, -0.3);
#if PALPHA_POROSITY
				// do not increase more than 1.1 times in the porous case
 				if (dtNew > 1.1 * dt) {
 					dtNew = 1.1 * dt;
 				}
#else
                // do not increase more than 5 times
                if (dtNew > 5.0 * dt) {
                    dtNew = 5.0 * dt;
                }
#endif
                // do not make timestep smaller
                if (dtNew < dt) {
                    dtNew = 1.05 * dt;
                }
            }

            dtNewErrorCheck = dtNew;
            blockCount = 0;   // reset block count
        }
    }
}
