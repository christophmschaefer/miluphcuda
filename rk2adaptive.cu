/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com, Thomas I. Maindl, Christoph Burger
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
#include <float.h>

extern __device__ double endTimeD, currentTimeD;
extern __device__ double substep_currentTimeD;
extern __device__ double dt;
extern __device__ int isRelaxed;
extern __device__ int blockCount;
extern __device__ int errorSmallEnough;
extern __device__ double dtNewErrorCheck;
extern __device__ double maxPosAbsError;

extern __device__ double maxVelAbsError;
extern __device__ double maxDensityAbsError;
extern __device__ double maxEnergyAbsError;
extern __device__ double maxPressureAbsChange;
extern __device__ double maxDamageTimeStep;
extern __device__ double maxAlphaDiff;

__constant__ __device__ double rk_epsrel_d;

extern double L_ini;



void rk2Adaptive()
{
    int rkstep;
    int errorSmallEnough_host;
    double dtmax_host = param.maxtimestep;
    assert(dtmax_host > 0);
    double dt_new;

    // vars for timestep benchmarking
    unsigned int ts_no_total = 0, ts_no_total_acc = 0, ts_no_total_rej = 0;   // total number of timesteps in sim
    unsigned int ts_no_substep = 0, ts_no_substep_acc = 0, ts_no_substep_rej = 0;
    double ts_smallest = DBL_MAX, ts_largest = 0.0;   // smallest, largest accepted timestep in sim
    double ts_smallest_rej = DBL_MAX;
    int approaching_output_time = FALSE;

    cudaVerify(cudaMemcpyToSymbol(rk_epsrel_d, &param.rk_epsrel, sizeof(double)));

    // allocate mem
    double *maxPosAbsErrorPerBlock;
    cudaVerify(cudaMalloc((void**)&maxPosAbsErrorPerBlock, sizeof(double)*numberOfMultiprocessors));
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
    double *maxVelAbsErrorPerBlock;
    cudaVerify(cudaMalloc((void**)&maxVelAbsErrorPerBlock, sizeof(double)*numberOfMultiprocessors));
#endif
#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
    double *maxDensityAbsErrorPerBlock;
    cudaVerify(cudaMalloc((void**)&maxDensityAbsErrorPerBlock , sizeof(double)*numberOfMultiprocessors));
#endif
#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
    double *maxEnergyAbsErrorPerBlock;
    cudaVerify(cudaMalloc((void**)&maxEnergyAbsErrorPerBlock, sizeof(double)*numberOfMultiprocessors));
#endif
#if RK2_USE_COURANT_LIMIT
    double *courantPerBlock;
    cudaVerify(cudaMalloc((void**)&courantPerBlock, sizeof(double)*numberOfMultiprocessors));
#endif
#if RK2_USE_FORCES_LIMIT
    double *forcesPerBlock;
    cudaVerify(cudaMalloc((void**)&forcesPerBlock, sizeof(double)*numberOfMultiprocessors));
#endif
#if RK2_USE_DAMAGE_LIMIT && FRAGMENTATION
    double *maxDamageTimeStepPerBlock;
    cudaVerify(cudaMalloc((void**)&maxDamageTimeStepPerBlock, sizeof(double)*numberOfMultiprocessors));
#endif
#if RK2_LIMIT_PRESSURE_CHANGE && PALPHA_POROSITY
    double *maxPressureAbsChangePerBlock;
    cudaVerify(cudaMalloc((void**)&maxPressureAbsChangePerBlock, sizeof(double)*numberOfMultiprocessors));
#endif
#if RK2_LIMIT_ALPHA_CHANGE && PALPHA_POROSITY
    double *maxAlphaDiffPerBlock;
    cudaVerify(cudaMalloc((void**)&maxAlphaDiffPerBlock, sizeof(double)*numberOfMultiprocessors));
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

        ts_no_substep = ts_no_substep_acc = ts_no_substep_rej = 0;
        approaching_output_time = FALSE;

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
                fprintf(stdout, "    starting with timestep: %g\n", dt_host);
        } else {
            dt_host = dt_suggested;   // use previously suggested next timestep as starting point
            if (dt_host < SMALLEST_DT_ALLOWED)
                dt_host = 1.1 * SMALLEST_DT_ALLOWED;
            if (dt_host > timePerStep)
                dt_host = timePerStep;
            if (param.verbose)
                fprintf(stdout, "    continuing with timestep: %g\n", dt_host);
        }
        assert(dt_host > 0.0);
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

        // loop until end of current output time
        while (currentTime < endTime) {
            // set all deactivation flags to zero
            cudaVerifyKernel((BoundaryConditionsBeforeIntegratorStep<<<numberOfMultiprocessors, NUM_THREADS_ERRORCHECK>>>(interactions))); 
            cudaVerify(cudaDeviceSynchronize());
            // get the correct time
            substep_currentTime = currentTime;
            cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));

            cudaVerify(cudaDeviceSynchronize());

            // copy particle data to first Runge Kutta step
            copy_particles_variables_device_to_device(&rk_device[RKFIRST], &p_device);
            cudaVerify(cudaDeviceSynchronize());
#if GRAVITATING_POINT_MASSES
            copy_pointmass_variables_device_to_device(&rk_pointmass_device[RKFIRST], &pointmass_device);
            cudaVerify(cudaDeviceSynchronize());
#endif

            // calculate first rhs, based on quantities in [RKFIRST]
            cudaVerify(cudaMemcpyToSymbol(p, &rk_device[RKFIRST], sizeof(struct Particle)));
#if GRAVITATING_POINT_MASSES
            cudaVerify(cudaMemcpyToSymbol(pointmass, &rk_pointmass_device[RKFIRST], sizeof(struct Pointmass)));
#endif
            rightHandSide();
            cudaVerify(cudaDeviceSynchronize());

#if RK2_USE_COURANT_LIMIT
            /* limit timestep based on CFL condition, with dt ~ sml/cs */
            cudaVerifyKernel((limitTimestepCourant<<<numberOfMultiprocessors, NUM_THREADS_LIMITTIMESTEP>>>(
                                courantPerBlock)));
            cudaVerify(cudaDeviceSynchronize());
            cudaVerify(cudaMemcpyFromSymbol(&dt_new, dt, sizeof(double)));
            if (param.verbose && dt_new < dt_host)
                fprintf(stdout, "reducing coming timestep due to CFL condition from %g to %g (current time: %e)\n", dt_host, dt_new, currentTime);
            dt_host = dt_suggested = dt_new;
#endif
#if RK2_USE_FORCES_LIMIT
            /* limit timestep based on local forces/acceleration, with dt ~ sqrt(sml/a) */
            cudaVerifyKernel((limitTimestepForces<<<numberOfMultiprocessors, NUM_THREADS_LIMITTIMESTEP>>>(
                                forcesPerBlock)));
            cudaVerify(cudaDeviceSynchronize());
            cudaVerify(cudaMemcpyFromSymbol(&dt_new, dt, sizeof(double)));
            if (param.verbose && dt_new < dt_host)
                fprintf(stdout, "reducing coming timestep due to forces/accels from %g to %g (current time: %e)\n", dt_host, dt_new, currentTime);
            dt_host = dt_suggested = dt_new;
#endif
#if RK2_USE_DAMAGE_LIMIT && FRAGMENTATION
            /* limit timestep based on rate of damage change */
            cudaVerifyKernel((limitTimestepDamage<<<numberOfMultiprocessors, NUM_THREADS_LIMITTIMESTEP>>>(
                                maxDamageTimeStepPerBlock)));
            cudaVerify(cudaDeviceSynchronize());
            cudaVerify(cudaMemcpyFromSymbol(&dt_new, dt, sizeof(double)));
            if (param.verbose && dt_new < dt_host)
                fprintf(stdout, "reducing coming timestep due to damage evolution from %g to %g (current time: %e)\n", dt_host, dt_new, currentTime);
            dt_host = dt_suggested = dt_new;
#endif

            // remember values of first step
            copy_particles_variables_device_to_device(&rk_device[RKSTART], &rk_device[RKFIRST]);
            copy_particles_derivatives_device_to_device(&rk_device[RKSTART], &rk_device[RKFIRST]);
#if GRAVITATING_POINT_MASSES
            copy_pointmass_variables_device_to_device(&rk_pointmass_device[RKSTART], &rk_pointmass_device[RKFIRST]);
            copy_pointmass_derivatives_device_to_device(&rk_pointmass_device[RKSTART], &rk_pointmass_device[RKFIRST]);
#endif
            // remember accels due to gravity
            if (param.selfgravity)
                copy_gravitational_accels_device_to_device(&rk_device[RKSTART], &rk_device[RKFIRST]);

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
                if (dt_host < SMALLEST_DT_ALLOWED && !approaching_output_time) {
                    fprintf(stderr, "Timestep %e is below SMALLEST_DT_ALLOWED. Stopping here.\n", dt_host);
                    exit(1);
                }

                // get derivatives for second step (i.e., compute k2), based on quantities in [RKFIRST]
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

                // get derivatives for the 3rd (and last) step (i.e., compute k3), based on quantities in [RKSECOND]
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
                                  maxPosAbsErrorPerBlock
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
                                , maxVelAbsErrorPerBlock
#endif
#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
                                , maxDensityAbsErrorPerBlock
#endif
#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
                                , maxEnergyAbsErrorPerBlock
#endif
#if RK2_LIMIT_PRESSURE_CHANGE && PALPHA_POROSITY
                                , maxPressureAbsChangePerBlock
#endif
#if RK2_LIMIT_ALPHA_CHANGE && PALPHA_POROSITY
                                , maxAlphaDiffPerBlock
#endif
                                )));
                cudaVerify(cudaDeviceSynchronize());
                cudaVerify(cudaMemcpyFromSymbol(&dt_suggested, dtNewErrorCheck, sizeof(double)));
                cudaVerify(cudaMemcpyFromSymbol(&errorSmallEnough_host, errorSmallEnough, sizeof(int)));
                cudaVerify(cudaDeviceSynchronize());

                /* last timestep was okay, forward time and continue with new timestep */
                if (errorSmallEnough_host) {
                    currentTime += dt_host;
                    if (!param.verbose) {
                        fprintf(stdout, "time: %e   last timestep: %g   time to next output: %e\n", currentTime, dt_host, endTime-currentTime);
                    }
                    cudaVerifyKernel((BoundaryConditionsAfterIntegratorStep<<<numberOfMultiprocessors, NUM_THREADS_ERRORCHECK>>>(interactions)));
                    cudaVerify(cudaDeviceSynchronize());
                }

                /* update timestep statistics */
                ts_no_substep++;
                if (errorSmallEnough_host) {
                    ts_no_substep_acc++;
                    if(!approaching_output_time)
                        ts_smallest = fmin(ts_smallest, dt_host);
                    ts_largest = fmax(ts_largest, dt_host);
                } else {
                    ts_no_substep_rej++;
                    ts_smallest_rej = fmin(ts_smallest_rej, dt_host);
                }

                /* print information about errors */
                if (param.verbose) {
                    double errPos = 0.0, errVel = 0.0, errDensity = 0.0, errEnergy = 0.0;
                    cudaVerify(cudaMemcpyFromSymbol(&errPos, maxPosAbsError, sizeof(double)));
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
                    cudaVerify(cudaMemcpyFromSymbol(&errVel, maxVelAbsError, sizeof(double)));
#endif
#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
                    cudaVerify(cudaMemcpyFromSymbol(&errDensity, maxDensityAbsError, sizeof(double)));
#endif
#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
                    cudaVerify(cudaMemcpyFromSymbol(&errEnergy, maxEnergyAbsError, sizeof(double)));
#endif
                    cudaVerify(cudaDeviceSynchronize());
                    fprintf(stdout, "    with timestep: %g\n", dt_host);
                    fprintf(stdout, "    total max error (relative to eps): %g   (location: %g, velocity: %g, density: %g, energy: %g)\n",
                            max(max(max(errPos, errVel), errDensity), errEnergy) / param.rk_epsrel,
                            errPos / param.rk_epsrel, errVel / param.rk_epsrel, errDensity / param.rk_epsrel, errEnergy / param.rk_epsrel);
#if PALPHA_POROSITY
                    double errPressure = 0.0, errAlpha = 0.0;
# if RK2_LIMIT_PRESSURE_CHANGE
                    cudaVerify(cudaMemcpyFromSymbol(&errPressure, maxPressureAbsChange, sizeof(double)));
# endif
# if RK2_LIMIT_ALPHA_CHANGE
                    cudaVerify(cudaMemcpyFromSymbol(&errAlpha, maxAlphaDiff, sizeof(double)));
# endif
                    cudaVerify(cudaDeviceSynchronize());
                    fprintf(stdout, "    total max change (relative to max allowed): %g   (pressure: %g, alpha: %g)\n",
                            max(errPressure, errAlpha), errPressure, errAlpha);
#endif
                    fprintf(stdout, "    errors suggest next timestep: %g\n", dt_suggested);
                }

                /* limit suggested next timestep to max allowed timestep */
                assert(dt_suggested > 0.0);
                if (dt_suggested > dtmax_host) {
#if DEBUG_TIMESTEP
                    fprintf(stdout, "suggested next timestep is larger than max allowed timestep, reduced from %g to %g\n", dt_suggested, dtmax_host);
#endif
                    dt_suggested = dtmax_host;
                }

                if (currentTime + dt_suggested > endTime) {
                    /* if suggested next timestep would overshoot, reduce it */
                    dt_host = endTime - currentTime;
#if DEBUG_TIMESTEP
                    fprintf(stdout, "next timestep would overshoot output time, reduced from suggested %g to %g\n", dt_suggested, dt_host);
#endif
                    approaching_output_time = TRUE;
                } else {
                    /* otherwise use suggested timestep for next step */
                    dt_host = dt_suggested;
                }

                /* tell the GPU the new timestep and the current time */
                cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
                cudaVerify(cudaMemcpyToSymbol(dt, &dt_host, sizeof(double)));

                /* break loop if timestep was successful, otherwise set stage for next adaptive round */
                if (errorSmallEnough_host) {
                    afterIntegrationStep();   // do something after successful step (e.g. look for min/max pressure...)
                    if (param.verbose) {
                        fprintf(stdout, "Errors were small enough, last timestep accepted, current time: %e   time to next output: %g   suggested next timestep: %g\n\n",
                                currentTime, endTime-currentTime, dt_host);
                    }
                    break; // break while(TRUE) and continue with next timestep
                } else {
                    if (param.verbose)
                        fprintf(stdout, "Errors were too large, last timestep rejected, current time: %e   timestep lowered to: %g\n\n", currentTime, dt_host);
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

        fprintf(stdout, "Finished integrating output step %d / %d. Had to integrate %d timesteps (%d accepted, %d rejected).\n",
                timestep+1, lastTimestep, ts_no_substep, ts_no_substep_acc, ts_no_substep_rej);
        ts_no_total += ts_no_substep;
        ts_no_total_acc += ts_no_substep_acc;
        ts_no_total_rej += ts_no_substep_rej;

        // write results
#if FRAGMENTATION
        // necessary because damage was limited only in rhs calls and not after integrating third step
        cudaVerify(cudaDeviceSynchronize());
        cudaVerifyKernel((damageLimit<<<numberOfMultiprocessors*4, NUM_THREADS_PC_INTEGRATOR>>>()));
        cudaVerify(cudaDeviceSynchronize());
#endif
        copyToHostAndWriteToFile(timestep, lastTimestep);
    } // timestep loop

    fprintf(stdout, "\nTimestep statistics:\n");
    fprintf(stdout, "    total no integrated timesteps: %d\n", ts_no_total);
    fprintf(stdout, "    accepted timesteps: %d\n", ts_no_total_acc);
    fprintf(stdout, "    rejected timesteps: %d\n", ts_no_total_rej);
    fprintf(stdout, "    fraction of rejected timesteps: %g\n\n", (double)ts_no_total_rej/(double)ts_no_total);
    fprintf(stdout, "    smallest accepted timestep: %g\n", ts_smallest);
    fprintf(stdout, "    largest accepted timestep:  %g\n", ts_largest);
    fprintf(stdout, "    smallest rejected timestep: %g\n\n", ts_smallest_rej);

    // free memory
    int free_immutables = 0;
    for (rkstep = 0; rkstep < 3; rkstep++) {
        free_particles_memory(&rk_device[rkstep], free_immutables);
#if GRAVITATING_POINT_MASSES
        free_pointmass_memory(&rk_pointmass_device[rkstep], free_immutables);
#endif
    }
    cudaVerify(cudaFree(maxPosAbsErrorPerBlock));
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
    cudaVerify(cudaFree(maxVelAbsErrorPerBlock));
#endif
#if RK2_USE_COURANT_LIMIT
    cudaVerify(cudaFree(courantPerBlock));
#endif
#if RK2_USE_FORCES_LIMIT
    cudaVerify(cudaFree(forcesPerBlock));
#endif
#if RK2_USE_DAMAGE_LIMIT && FRAGMENTATION
    cudaVerify(cudaFree(maxDamageTimeStepPerBlock));
#endif
#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
    cudaVerify(cudaFree(maxEnergyAbsErrorPerBlock));
#endif
#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
    cudaVerify(cudaFree(maxDensityAbsErrorPerBlock));
#endif
#if RK2_LIMIT_PRESSURE_CHANGE && PALPHA_POROSITY
    cudaVerify(cudaFree(maxPressureAbsChangePerBlock));
#endif
#if RK2_LIMIT_ALPHA_CHANGE && PALPHA_POROSITY
    cudaVerify(cudaFree(maxAlphaDiffPerBlock));
#endif
}



#if RK2_USE_COURANT_LIMIT
__global__ void limitTimestepCourant(double *courantPerBlock)
{
    __shared__ double sharedCourant[NUM_THREADS_LIMITTIMESTEP];
    int i, j, k, m;
    double courant = 1e100;

    // loop for particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        // only consider particles that interact
        if (p.noi[i] > 0) {
            courant = min(courant, p.h[i] / p.cs[i]);
        }
    }

    // reduce shared thread results to one per block
    i = threadIdx.x;
    sharedCourant[i] = courant;
    for (j = NUM_THREADS_LIMITTIMESTEP / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;
            sharedCourant[i] = courant = min(courant, sharedCourant[k]);
        }
    }

    // compute block result
    if (i == 0) {
        k = blockIdx.x;
        courantPerBlock[k] = courant;
        m = gridDim.x - 1;
        if (m == atomicInc((unsigned int *)&blockCount, m)) {
            // last block, so combine all block results
            for (j = 0; j <= m; j++)
                courant = min(courant, courantPerBlock[j]);
            blockCount = 0;  // reset block count

            courant *= COURANT_FACT;
#if DEBUG_TIMESTEP
            printf("<limitTimestepCourant> suggests max timestep: %g\n", courant);
#endif
            // reduce timestep if necessary
            if (courant < dt  &&  courant > 0.0)
                dt = courant;
        }
    }
}
#endif



#if RK2_USE_FORCES_LIMIT
__global__ void limitTimestepForces(double *forcesPerBlock)
{
    __shared__ double sharedForces[NUM_THREADS_LIMITTIMESTEP];
    int i, j, k, m;
    double forces = 1e100;
    double tmp;
    double ax;
#if DIM > 1
    double ay;
#endif
#if DIM == 3
    double az;
#endif

    // loop for particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        ax = p.ax[i];
#if DIM > 1
        ay = p.ay[i];
#endif
#if DIM == 3
        az = p.az[i];
#endif
        tmp = ax*ax;
#if DIM > 1
        tmp += ay*ay;
#endif
#if DIM == 3
         tmp += az*az;
#endif
        if (tmp > 0.0) {
            tmp = sqrt(p.h[i] / sqrt(tmp));
            forces = min(forces, tmp);
        }
    }

    // reduce shared thread results to one per block
    i = threadIdx.x;
    sharedForces[i] = forces;
    for (j = NUM_THREADS_LIMITTIMESTEP / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;
            sharedForces[i] = forces = min(forces, sharedForces[k]);
        }
    }

    // compute block result
    if (i == 0) {
        k = blockIdx.x;
        forcesPerBlock[k] = forces;
        m = gridDim.x - 1;
        if (m == atomicInc((unsigned int *)&blockCount, m)) {
            // last block, so combine all block results
            for (j = 0; j <= m; j++)
                forces = min(forces, forcesPerBlock[j]);
            blockCount = 0;  // reset block count

            forces *= FORCES_FACT;
#if DEBUG_TIMESTEP
            printf("<limitTimestepForces> suggests max timestep: %g\n", forces);
#endif
            // reduce timestep if necessary
            if (forces < dt  &&  forces > 0.0)
                dt = forces;
        }
    }
}
#endif



#if RK2_USE_DAMAGE_LIMIT && FRAGMENTATION
__global__ void limitTimestepDamage(double *maxDamageTimeStepPerBlock)
{
    __shared__ double sharedMaxDamageTimeStep[NUM_THREADS_LIMITTIMESTEP];
    double localMaxDamageTimeStep = 1e100;
    double tmp = 0.0;
    int i, j, k, m;

    // loop for particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        if (p.dddt[i] > 0.0) {
            tmp = 0.7 * (p.d[i] + RK2_MAX_DAMAGE_CHANGE) / p.dddt[i];
            tmp = min(tmp, RK2_MAX_DAMAGE_CHANGE / p.dddt[i]);
            localMaxDamageTimeStep = min(tmp, localMaxDamageTimeStep);
        }
    }

    // reduce shared thread results to one per block
    i = threadIdx.x;
    sharedMaxDamageTimeStep[i] = localMaxDamageTimeStep;
    for (j = NUM_THREADS_LIMITTIMESTEP / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;
            sharedMaxDamageTimeStep[i] = localMaxDamageTimeStep = min(localMaxDamageTimeStep, sharedMaxDamageTimeStep[k]);
        }
    }

    // compute block result
    if (i == 0) {
        k = blockIdx.x;
        maxDamageTimeStepPerBlock[k] = localMaxDamageTimeStep;
        m = gridDim.x - 1;
        if (m == atomicInc((unsigned int *)&blockCount, m)) {
            // last block, so combine all block results
            for (j = 0; j <= m; j++) {
                localMaxDamageTimeStep = min(localMaxDamageTimeStep, maxDamageTimeStepPerBlock[j]);
            }
            blockCount = 0;  // reset block count

#if DEBUG_TIMESTEP
            printf("<limitTimestepDamage> suggests max timestep: %g\n", localMaxDamageTimeStep);
#endif
            // reduce timestep if necessary
            if (localMaxDamageTimeStep < dt  &&  localMaxDamageTimeStep > 0.0)
                dt = localMaxDamageTimeStep;

            // write also to global device mem...
            maxDamageTimeStep = localMaxDamageTimeStep;
        }
    }
}
#endif



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
        rk[RKFIRST].ep[i] = rk[RKSTART].ep[i] + dt * B21 * rk[RKSTART].edotp[i];
#endif
#if JC_PLASTICITY
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
# if PALPHA_POROSITY
        rk[RKSECOND].damage_porjutzi[i] = rk[RKSTART].damage_porjutzi[i] + dt * (B31 * rk[RKSTART].ddamage_porjutzidt[i] + B32 * rk[RKFIRST].ddamage_porjutzidt[i]);
# endif
#endif
#if JC_PLASTICITY
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
        rk[RKSECOND].ep[i] = rk[RKSTART].ep[i] + dt * (B31 * rk[RKSTART].edotp[i] + B32 * rk[RKFIRST].edotp[i]);
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
        p.T[i] = rk[RKSTART].T[i] + dt/6.0 *
            (    C1 * rk[RKSTART].dTdt[i]
               + C2 * rk[RKFIRST].dTdt[i]
               + C3 * rk[RKSECOND].dTdt[i]);
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
        p.ep[i] = rk[RKSTART].ep[i] + dt/6.0 *
                                      (    C1 * rk[RKSTART].edotp[i]
                                           + C2 * rk[RKFIRST].edotp[i]
                                           + C3 * rk[RKSECOND].edotp[i]);
        p.edotp[i] = 1./6. * ( C1 * rk[RKSTART].edotp[i]
                               + C2 * rk[RKFIRST].edotp[i]
                               + C3 * rk[RKSECOND].edotp[i]);
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



__global__ void checkError(double *maxPosAbsErrorPerBlock
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
                        , double *maxVelAbsErrorPerBlock
#endif
#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
                        , double *maxDensityAbsErrorPerBlock
#endif
#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
                        , double *maxEnergyAbsErrorPerBlock
#endif
#if RK2_LIMIT_PRESSURE_CHANGE && PALPHA_POROSITY
                        , double *maxPressureAbsChangePerBlock
#endif
#if RK2_LIMIT_ALPHA_CHANGE && PALPHA_POROSITY
                        , double *maxAlphaDiffPerBlock
#endif
        )
{
    __shared__ double sharedMaxPosAbsError[NUM_THREADS_ERRORCHECK];
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
    __shared__ double sharedMaxVelAbsError[NUM_THREADS_ERRORCHECK];
#endif
#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
    __shared__ double sharedMaxDensityAbsError[NUM_THREADS_ERRORCHECK];
    double localMaxDensityAbsError = 0.0;
#endif
#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
    __shared__ double sharedMaxEnergyAbsError[NUM_THREADS_ERRORCHECK];
    double localMaxEnergyAbsError = 0.0;
    int hasEnergy = 0;
#endif
#if RK2_LIMIT_PRESSURE_CHANGE && PALPHA_POROSITY
    __shared__ double sharedMaxPressureAbsChange[NUM_THREADS_ERRORCHECK];
    double localMaxPressureAbsChange = 0.0;
#endif
#if RK2_LIMIT_ALPHA_CHANGE && PALPHA_POROSITY
    __shared__ double sharedMaxAlphaDiff[NUM_THREADS_ERRORCHECK];
    double localMaxAlphaDiff = 0.0;
#endif
    int i, j, k, m;
    double dtNew = 0.0;
    double localMaxPosAbsError = 0.0;
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
    double localMaxVelAbsError = 0.0;
    double tmp_vel = 0.0, tmp_vel2 = 0.0;
#endif
    double tmp = 0.0;
    double tmp_pos = 0.0, tmp_pos2 = 0.0;
    double min_pos_change_rk2 = 0.0;



#if GRAVITATING_POINT_MASSES && RK2_USE_VELOCITY_ERROR_POINTMASSES
    // loop for pointmasses
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i+= blockDim.x * gridDim.x) {
        tmp = dt * (rk_pointmass[RKFIRST].ax[i]/3.0 - (rk_pointmass[RKSTART].ax[i] + rk_pointmass[RKSECOND].ax[i])/6.0);
        tmp_vel = fabs(rk_pointmass[RKSTART].vx[i]) + fabs(dt*rk_pointmass[RKSTART].ax[i]);
        if (tmp_vel > MIN_VEL_CHANGE_RK2) {
            tmp_vel2 = fabs(tmp) / tmp_vel;
            localMaxVelAbsError = max(localMaxVelAbsError, tmp_vel2);
        }
# if DIM > 1
        tmp = dt * (rk_pointmass[RKFIRST].ay[i]/3.0 - (rk_pointmass[RKSTART].ay[i] + rk_pointmass[RKSECOND].ay[i])/6.0);
        tmp_vel = fabs(rk_pointmass[RKSTART].vy[i]) + fabs(dt*rk_pointmass[RKSTART].ay[i]);
        if (tmp_vel > MIN_VEL_CHANGE_RK2) {
            tmp_vel2 = fabs(tmp) / tmp_vel;
            localMaxVelAbsError = max(localMaxVelAbsError, tmp_vel2);
        }
# endif
# if DIM == 3
        tmp = dt * (rk_pointmass[RKFIRST].az[i]/3.0 - (rk_pointmass[RKSTART].az[i] + rk_pointmass[RKSECOND].az[i])/6.0);
        tmp_vel = fabs(rk_pointmass[RKSTART].vz[i]) + fabs(dt*rk_pointmass[RKSTART].az[i]);
        if (tmp_vel > MIN_VEL_CHANGE_RK2) {
            tmp_vel2 = fabs(tmp) / tmp_vel;
            localMaxVelAbsError = max(localMaxVelAbsError, tmp_vel2);
        }
# endif
    }
#endif

    // loop for particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        if (p_rhs.materialId[i] == EOS_TYPE_IGNORE) continue;

        min_pos_change_rk2 = rk[RKSTART].h[i] * RK2_LOCATION_SAFETY;

        tmp = dt * (rk[RKFIRST].dxdt[i]/3.0 - (rk[RKSTART].dxdt[i] + rk[RKSECOND].dxdt[i])/6.0);
        tmp_pos = fabs(rk[RKSTART].x[i]) + fabs(dt*rk[RKSTART].dxdt[i]);
        if (tmp_pos > min_pos_change_rk2) {
            tmp_pos2 = fabs(tmp) / tmp_pos;
            localMaxPosAbsError = max(localMaxPosAbsError, tmp_pos2);
        }
#if DIM > 1
        tmp = dt * (rk[RKFIRST].dydt[i]/3.0 - (rk[RKSTART].dydt[i] + rk[RKSECOND].dydt[i])/6.0);
        tmp_pos = fabs(rk[RKSTART].y[i]) + fabs(dt*rk[RKSTART].dydt[i]);
        if (tmp_pos > min_pos_change_rk2) {
            tmp_pos2 = fabs(tmp) / tmp_pos;
            localMaxPosAbsError = max(localMaxPosAbsError, tmp_pos2);
        }
#endif
#if DIM > 2
        tmp = dt * (rk[RKFIRST].dzdt[i]/3.0 - (rk[RKSTART].dzdt[i] + rk[RKSECOND].dzdt[i])/6.0);
        tmp_pos = fabs(rk[RKSTART].z[i]) + fabs(dt*rk[RKSTART].dzdt[i]);
        if (tmp_pos > min_pos_change_rk2) {
            tmp_pos2 = fabs(tmp) / tmp_pos;
            localMaxPosAbsError = max(localMaxPosAbsError, tmp_pos2);
        }
#endif

#if RK2_USE_VELOCITY_ERROR
        tmp = dt * (rk[RKFIRST].ax[i]/3.0 - (rk[RKSTART].ax[i] + rk[RKSECOND].ax[i])/6.0);
        tmp_vel = fabs(rk[RKSTART].vx[i]) + fabs(dt*rk[RKSTART].ax[i]);
        if (tmp_vel > MIN_VEL_CHANGE_RK2) {
            tmp_vel2 = fabs(tmp) / tmp_vel;
            localMaxVelAbsError = max(localMaxVelAbsError, tmp_vel2);
        }
# if DIM > 1
        tmp = dt * (rk[RKFIRST].ay[i]/3.0 - (rk[RKSTART].ay[i] + rk[RKSECOND].ay[i])/6.0);
        tmp_vel = fabs(rk[RKSTART].vy[i]) + fabs(dt*rk[RKSTART].ay[i]);
        if (tmp_vel > MIN_VEL_CHANGE_RK2) {
            tmp_vel2 = fabs(tmp) / tmp_vel;
            localMaxVelAbsError = max(localMaxVelAbsError, tmp_vel2);
        }
# endif
# if DIM == 3
        tmp = dt * (rk[RKFIRST].az[i]/3.0 - (rk[RKSTART].az[i] + rk[RKSECOND].az[i])/6.0);
        tmp_vel = fabs(rk[RKSTART].vz[i]) + fabs(dt*rk[RKSTART].az[i]);
        if (tmp_vel > MIN_VEL_CHANGE_RK2) {
            tmp_vel2 = fabs(tmp) / tmp_vel;
            localMaxVelAbsError = max(localMaxVelAbsError, tmp_vel2);
        }
# endif
#endif

#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
        tmp = dt * (rk[RKFIRST].drhodt[i]/3.0 - (rk[RKSTART].drhodt[i] + rk[RKSECOND].drhodt[i])/6.0);
        tmp = fabs(tmp) / (fabs(rk[RKSTART].rho[i]) + fabs(dt*rk[RKSTART].drhodt[i]) + RK2_TINY_DENSITY);
        localMaxDensityAbsError = max(localMaxDensityAbsError, tmp);
#endif

#if RK2_LIMIT_PRESSURE_CHANGE && PALPHA_POROSITY
        // check if the pressure changes too much
        tmp = fabs(rk[RKFIRST].p[i] - rk[RKSECOND].p[i]);
        localMaxPressureAbsChange = max(localMaxPressureAbsChange, tmp);
#endif
#if RK2_LIMIT_ALPHA_CHANGE && PALPHA_POROSITY
        // check if alpha changes too much
        tmp = fabs(rk[RKSTART].alpha_jutzi_old[i] - p.alpha_jutzi[i]);
        localMaxAlphaDiff = max(localMaxAlphaDiff, tmp);
#endif

#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
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
            tmp = dt * (rk[RKFIRST].dedt[i]/3.0 - (rk[RKSTART].dedt[i] + rk[RKSECOND].dedt[i])/6.0);
            tmp = fabs(tmp) / (fabs(rk[RKSTART].e[i]) + fabs(dt*rk[RKSTART].dedt[i]) + RK2_TINY_ENERGY);
            localMaxEnergyAbsError = max(localMaxEnergyAbsError, tmp);
        }
#endif
    }   // loop for particles


    // reduce shared thread results to one per block
    i = threadIdx.x;
    sharedMaxPosAbsError[i] = localMaxPosAbsError;
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
    sharedMaxVelAbsError[i] = localMaxVelAbsError;
#endif
#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
    sharedMaxDensityAbsError[i] = localMaxDensityAbsError;
#endif
#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
    sharedMaxEnergyAbsError[i] = localMaxEnergyAbsError;
#endif
#if RK2_LIMIT_PRESSURE_CHANGE && PALPHA_POROSITY
    sharedMaxPressureAbsChange[i] = localMaxPressureAbsChange;
#endif
#if RK2_LIMIT_ALPHA_CHANGE && PALPHA_POROSITY
    sharedMaxAlphaDiff[i] = localMaxAlphaDiff;
#endif
    for (j = NUM_THREADS_ERRORCHECK / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;
            sharedMaxPosAbsError[i] = localMaxPosAbsError = max(localMaxPosAbsError, sharedMaxPosAbsError[k]);
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
            sharedMaxVelAbsError[i] = localMaxVelAbsError = max(localMaxVelAbsError, sharedMaxVelAbsError[k]);
#endif
#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
            sharedMaxDensityAbsError[i] = localMaxDensityAbsError = max(localMaxDensityAbsError, sharedMaxDensityAbsError[k]);
#endif
#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
            sharedMaxEnergyAbsError[i] = localMaxEnergyAbsError = max(localMaxEnergyAbsError, sharedMaxEnergyAbsError[k]);
#endif
#if RK2_LIMIT_PRESSURE_CHANGE && PALPHA_POROSITY
            sharedMaxPressureAbsChange[i] = localMaxPressureAbsChange = max(localMaxPressureAbsChange, sharedMaxPressureAbsChange[k]);
#endif
#if RK2_LIMIT_ALPHA_CHANGE && PALPHA_POROSITY
            sharedMaxAlphaDiff[i] = localMaxAlphaDiff = max(localMaxAlphaDiff, sharedMaxAlphaDiff[k]);
#endif
        }
    }

    // write block result to global memory
    if (i == 0) {
        k = blockIdx.x;
        maxPosAbsErrorPerBlock[k] = localMaxPosAbsError;
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
        maxVelAbsErrorPerBlock[k] = localMaxVelAbsError;
#endif
#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
        maxDensityAbsErrorPerBlock[k] = localMaxDensityAbsError;
#endif
#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
        maxEnergyAbsErrorPerBlock[k] = localMaxEnergyAbsError;
#endif
#if RK2_LIMIT_PRESSURE_CHANGE && PALPHA_POROSITY
        maxPressureAbsChangePerBlock[k] = localMaxPressureAbsChange;
#endif
#if RK2_LIMIT_ALPHA_CHANGE && PALPHA_POROSITY
        maxAlphaDiffPerBlock[k] = localMaxAlphaDiff;
#endif
        m = gridDim.x - 1;
        if (m == atomicInc((unsigned int *)&blockCount, m)) {
            // last block, so combine all block results
            for (j = 0; j <= m; j++) {
                localMaxPosAbsError = max(localMaxPosAbsError, maxPosAbsErrorPerBlock[j]);
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
                localMaxVelAbsError = max(localMaxVelAbsError, maxVelAbsErrorPerBlock[j]);
#endif
#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
                localMaxDensityAbsError = max(localMaxDensityAbsError, maxDensityAbsErrorPerBlock[j]);
#endif
#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
                localMaxEnergyAbsError = max(localMaxEnergyAbsError, maxEnergyAbsErrorPerBlock[j]);
#endif
#if RK2_LIMIT_PRESSURE_CHANGE && PALPHA_POROSITY
                localMaxPressureAbsChange = max(localMaxPressureAbsChange, maxPressureAbsChangePerBlock[j]);
#endif
#if RK2_LIMIT_ALPHA_CHANGE && PALPHA_POROSITY
                localMaxAlphaDiff = max(localMaxAlphaDiff, maxAlphaDiffPerBlock[j]);
#endif
            }

            // (single) max relative error
            tmp = localMaxPosAbsError;
            maxPosAbsError = localMaxPosAbsError;
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
            tmp = max(tmp, localMaxVelAbsError);
            maxVelAbsError = localMaxVelAbsError;
#endif
#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
            tmp = max(tmp, localMaxDensityAbsError);
            maxDensityAbsError = localMaxDensityAbsError;
#endif
#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
            tmp = max(tmp, localMaxEnergyAbsError);
            maxEnergyAbsError = localMaxEnergyAbsError;
#endif
            // max relative error over Runge-Kutta eps - error too large if > 1, error ok if < 1
            tmp /= rk_epsrel_d;

#if RK2_LIMIT_PRESSURE_CHANGE && PALPHA_POROSITY
            // store change relative to max allowed change
            maxPressureAbsChange = localMaxPressureAbsChange / max_abs_pressure_change;

            tmp = max(tmp, maxPressureAbsChange);
#endif
#if RK2_LIMIT_ALPHA_CHANGE && PALPHA_POROSITY
            // store change relative to max allowed change
            maxAlphaDiff = localMaxAlphaDiff / RK2_MAX_ALPHA_CHANGE;

            tmp = max(tmp, maxAlphaDiff);

// old implemenation:
//        dtNewAlphaCheck = dt * RK2_MAX_ALPHA_CHANGE / maxAlphaDiff;
//        if (maxAlphaDiff > RK2_MAX_ALPHA_CHANGE)
//            dtNewAlphaCheck = dt * RK2_MAX_ALPHA_CHANGE / (maxAlphaDiff * 1.51);
#endif

            if (tmp > 1.0) {
                /* error too large */
                errorSmallEnough = FALSE;
                dtNew = max( 0.1*dt, dt*RK2_TIMESTEP_SAFETY*pow(tmp,-0.25) );
            } else {
                /* error small enough */
                errorSmallEnough = TRUE;
                dtNew = dt * RK2_TIMESTEP_SAFETY * pow(tmp, -0.3);
//#if PALPHA_POROSITY
                // do not increase more than 1.1 times for p-alpha porosity
//                if (dtNew > 1.1 * dt)
//                    dtNew = 1.1 * dt;
//#else
                // do not increase more than 5 times
                if (dtNew > 5.0 * dt)
                    dtNew = 5.0 * dt;
//#endif
                // do not make timestep smaller if error small enough
                if (dtNew < dt)
                    dtNew = dt;
            }

            dtNewErrorCheck = dtNew;
            blockCount = 0;   // reset block count
        }
    }

// another loop to check if one particle got deactivated. if so, the timestep is
// set to the old time step and the last step is repeated
// we do not use reduction or atomic here because if the flag is set for one particle, the whole step
// has to be repeated for all
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        if (p_rhs.deactivate_me_flag[i] > 0) {
            printf("Deactivation flag set for particle %d - deactivation now!! and re-doing the last time step\n.", i);
            p_rhs.materialId[i] = EOS_TYPE_IGNORE;
            p_rhs.deactivate_me_flag[i] = FALSE;
            errorSmallEnough = FALSE;
            dtNewErrorCheck = 0.1*dt;
        }
    }


}



void print_rk2_adaptive_settings()
{
    double tmp;

    fprintf(stdout, "\n\n");
    fprintf(stdout, "Using rk2_adaptive for time-integration with the following settings:\n");
    fprintf(stdout, "    start time: %g\n", startTime);
    fprintf(stdout, "    output index of start time: %d\n", startTimestep);
    fprintf(stdout, "    no output times: %d\n", numberOfTimesteps);
    fprintf(stdout, "    duration between output times: %g\n", timePerStep);
    fprintf(stdout, "\n");
    fprintf(stdout, "    first timestep from cmd-line: %g\n", param.firsttimestep);
    fprintf(stdout, "    max allowed timestep: %g\n", param.maxtimestep);
    fprintf(stdout, "\n");
    fprintf(stdout, "    pre-timestep checks to limit timestep in advance:\n");
#if RK2_USE_COURANT_LIMIT
    fprintf(stdout, "        Courant condition:    yes    (Courant factor: %g)\n", COURANT_FACT);
#else
    fprintf(stdout, "        Courant condition:    no\n");
#endif
#if RK2_USE_FORCES_LIMIT
    fprintf(stdout, "        forces/acceleration:  yes    (forces factor: %g)\n", FORCES_FACT);
#else
    fprintf(stdout, "        forces/acceleration:  no\n");
#endif
#if FRAGMENTATION
# if RK2_USE_DAMAGE_LIMIT
    fprintf(stdout, "        limit damage change:  yes    (MAX_DAMAGE_CHANGE: %g)\n", RK2_MAX_DAMAGE_CHANGE);
# else
    fprintf(stdout, "        limit damage change:  no\n");
# endif
#endif
    fprintf(stdout, "\n");
    fprintf(stdout, "    post-timestep error checks to adapt timestep:\n");
    fprintf(stdout, "        general accuracy (eps): %g\n", param.rk_epsrel);
    fprintf(stdout, "        positions:       yes    (LOCATION_SAFETY: %g)\n", RK2_LOCATION_SAFETY);
#if RK2_USE_VELOCITY_ERROR
    fprintf(stdout, "        velocities:      yes    (MIN_VEL_CHANGE: %g)\n", MIN_VEL_CHANGE_RK2);
#else
    fprintf(stdout, "        velocities:      no\n");
#endif
#if GRAVITATING_POINT_MASSES
# if RK2_USE_VELOCITY_ERROR_POINTMASSES
    fprintf(stdout, "        velocities pointmasses: yes    (MIN_VEL_CHANGE: %g)\n", MIN_VEL_CHANGE_RK2);
# else
    fprintf(stdout, "        velocities pointmasses: no\n");
# endif
#endif
#if INTEGRATE_DENSITY
# if RK2_USE_DENSITY_ERROR
    fprintf(stdout, "        density:         yes    (TINY_DENSITY: %g)\n", RK2_TINY_DENSITY);
# else
    fprintf(stdout, "        density:         no\n");
# endif
#endif
#if INTEGRATE_ENERGY
# if RK2_USE_ENERGY_ERROR
    fprintf(stdout, "        energy:          yes    (TINY_ENERGY: %g)\n", RK2_TINY_ENERGY);
# else
    fprintf(stdout, "        energy:          no\n");
# endif
#endif
#if PALPHA_POROSITY
# if RK2_LIMIT_PRESSURE_CHANGE
    cudaVerify(cudaMemcpyFromSymbol(&tmp, max_abs_pressure_change, sizeof(double)));
    fprintf(stdout, "        pressure change: yes    (max allowed change: %g)\n", tmp);
# else
    fprintf(stdout, "        pressure change: no\n");
# endif
# if RK2_LIMIT_ALPHA_CHANGE
    fprintf(stdout, "        alpha change:    yes    (max allowed change: %g)\n", RK2_MAX_ALPHA_CHANGE);
# else
    fprintf(stdout, "        alpha change:    no\n");
# endif
#endif
    fprintf(stdout, "\n");
}
