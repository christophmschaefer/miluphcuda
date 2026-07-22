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

#include "euler.h"
#include "timeintegration.h"
#include "parameter.h"
#include "rhs.h"


extern __device__ double dt;
extern __device__ double endTimeD, currentTimeD;

extern double L_ini;



__global__ void integrateEuler(void)
{
        register int i, inc;
        inc = blockDim.x * gridDim.x;
#if GRAVITATING_POINT_MASSES
        for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i += inc) {
                pointmass.x[i] += dt * pointmass.vx[i];
#if DIM > 1
                pointmass.y[i] += dt * pointmass.vy[i];
                pointmass.vy[i] += dt * pointmass.ay[i];
#if DIM == 3
                pointmass.z[i] += dt * pointmass.vz[i];
#endif
#endif
                pointmass.vx[i] += dt * pointmass.ax[i];
#if DIM == 3
                pointmass.vz[i] += dt * pointmass.az[i];
#endif

        }
#endif
        for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
#if INTEGRATE_DENSITY
            p.rho[i] += dt * p.drhodt[i];
#endif
#if INTEGRATE_ENERGY
            p.e[i] += dt * p.dedt[i];
#endif

#if PALPHA_POROSITY
            p.alpha_jutzi[i] += dt * p.dalphadt[i];
#endif

#if SIRONO_POROSITY
            p.rho_0prime[i] = p.rho_0prime[i];
            p.rho_c_plus[i] = p.rho_c_plus[i];
            p.rho_c_minus[i] = p.rho_c_minus[i];
            p.compressive_strength[i] = p.compressive_strength[i];
            p.tensile_strength[i] = p.tensile_strength[i];
            p.shear_strength[i] = p.shear_strength[i];
            p.K[i] = p.K[i];
            p.flag_rho_0prime[i] = p.flag_rho_0prime[i];
            p.flag_plastic[i] = p.flag_plastic[i];
#endif

#if INTEGRATE_SML
            p.h[i] += dt * p.dhdt[i];
#endif
#if JC_PLASTICITY
            p.T[i] += dt * p.dTdt[i];
#endif
#if INVISCID_SPH
            p.beta[i] += dt * p.dbetadt[i];
#endif
#if SOLID
#if FRAGMENTATION
            p.d[i] += dt * p.dddt[i];
# if PALPHA_POROSITY
            p.damage_porjutzi[i] += dt * p.ddamage_porjutzidt[i];
            p.pold[i] = p.p[i];
# endif
#endif
            int k;
            for (k = 0; k < DIM*DIM; k++) {
                    p.S[i*DIM*DIM+k] += dt * p.dSdt[i*DIM*DIM+k];
            }
            p.ep[i] += dt * p.edotp[i];
#endif
            p.x[i] += dt * p.dxdt[i];
#if DIM > 1
            p.y[i] += dt * p.dydt[i];
            p.vy[i] += dt * p.ay[i];
#if DIM == 3
            p.z[i] += dt * p.dzdt[i];
#endif
#endif
            p.vx[i] += dt * p.ax[i];
#if DIM == 3
            p.vz[i] += dt * p.az[i];
#endif
        }
}




void euler()
{
        // integrate
        int lastTimestep = startTimestep + numberOfTimesteps;
        int timestep;
        int eulerstep;
        double tmptimestep = param.maxtimestep;
        double endTime = startTime;
        currentTime = startTime;
        cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
        cudaVerify(cudaMemcpyToSymbol(dt, &tmptimestep, sizeof(double)));

        for (timestep = startTimestep; timestep < lastTimestep; timestep++) {
                eulerstep = 0;
                endTime += timePerStep;
                cudaVerify(cudaMemcpyToSymbol(endTimeD, &endTime, sizeof(double)));
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
                while (currentTime < endTime) {
                        fprintf(stdout, "Euler Step # %d\n", ++eulerstep);
                        fprintf(stdout, " currenttime: %e \t endtime: %e, integrating with euler dt: %g\n", currentTime, endTime, param.maxtimestep);
                        rightHandSide();
                        if (currentTime + param.maxtimestep > endTime) {
                            tmptimestep = endTime - currentTime;
                            cudaVerify(cudaMemcpyToSymbol(dt, &tmptimestep, sizeof(double)));
                            currentTime += tmptimestep;
                        } else {
                            cudaVerify(cudaMemcpyToSymbol(dt, &param.maxtimestep, sizeof(double)));
                            currentTime += param.maxtimestep;
                        }
                        cudaVerifyKernel((integrateEuler<<<numberOfMultiprocessors, NUM_THREADS_EULER_INTEGRATOR>>>()));
			//step was successful --> do something (e.g. look for min/max pressure...)
                    	afterIntegrationStep();
                }

                copyToHostAndWriteToFile(timestep, lastTimestep);
        }
}
