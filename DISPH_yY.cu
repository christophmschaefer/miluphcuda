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

#include "DISPH_yY.h"
#include "miluph.h"
#include "config_parameter.h"
#include "timeintegration.h"
#include "parameter.h"
#include "tree.h"
#include "pressure.h"
#include "aneos.h"

#if DISPH

extern __device__ SPH_kernel kernel;
extern __device__ SPH_kernel wendlandc2_p;


__global__ void calculate_DISPH_y_DISPH_rho(int *interactions) {


    int i, inc;
    int j;
    int ip;
    int d;
    double W;
    double dx[DIM];
    double dWdx[DIM];
    double dWdr;
    double sml;
    register double y;
    register double rho;
    int matId;


            // Start loop over all particles
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
            
    	matId = p_rhs.materialId[i];
            sml = p.h[i];
            
            // self-contribution of particle i
            for (d = 0; d < DIM; d++) {
                dx[d] = 0;
            }
            kernel(&W, dWdx, &dWdr, dx, sml);
            y = p.DISPH_Y[i] * W;


            // sph sum for particle i over neighbour particles
            for (j = 0; j < p.noi[i]; j++) {
                ip = interactions[i * MAX_NUM_INTERACTIONS + j];
		if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[ip]] || p_rhs.materialId[ip] == EOS_TYPE_IGNORE) {
                    continue;
                }


                                dx[0] = p.x[i] - p.x[ip];
                #if DIM > 1
                                dx[1] = p.y[i] - p.y[ip];
                #if DIM > 2
                                dx[2] = p.z[i] - p.z[ip];
                #endif
                #endif  
                kernel(&W, dWdx, &dWdr, dx, sml);
                y += p.DISPH_Y[ip] * W;
	    }
	    // write to global memory
	    p.DISPH_y[i] = y;

            rho = p.m[i]*p.DISPH_y[i]/p.DISPH_Y[i];
            if (rho <  matRho0[matId]*matRhoLimit[matId]) {
		    rho =matRho0[matId]*matRhoLimit[matId];
	    }
	    p.DISPH_rho[i] = rho;
            } // end loop over all particles
	    
}


__global__ void calculate_DISPH_Y() {
    register int i, inc;
    double DISPH_alpha = 0.1;
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
                p.DISPH_Y[i] = (p.m[i]*pow(p.p[i], DISPH_alpha))/p.DISPH_rho[i];
            }

}
__global__ void DISPH_Y_to_zero() {
    register int i, inc;
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
                p.DISPH_Y[i] =0.0;
 	    }

}

__global__ void set_initial_DISPH_Y_if_its_zero() {

    double eta, e, rho, mu, p1, p2, pressure;
    int i, inc;
    double DISPH_alpha = 0.1;
    int matId;
    extern __device__ int DISPH_initial_Y;

            // Start loop over all particles
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
		    if (p.DISPH_Y[i] == 0.0){
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        	matId = p_rhs.materialId[i];
		if (p.rho[i] < matRho0[matId]*matRhoLimit[matId]){ //matRho0[matId]*1.05) {
		    p.DISPH_rho[i] =  matRho0[matId]*matRhoLimit[matId]; //matRho0[matId]*1.05;
		}
		else{
		    p.DISPH_rho[i] = p.rho[i];
		}
	    }
DISPH_initial_Y = 1;
	break;
	    }else{
		    DISPH_initial_Y = 0;
	    }
	    }

}



__global__ void determine_max_dp(double *maxDISPH_PressureAbsErrorPerBlock)
{
	__shared__ double sharedMaxDISPH_PressureAbsError[NUM_THREADS_ERRORCHECK];
	double localMaxDISPH_PressureAbsError = 0.0;
	extern __device__ double maxDISPH_PressureAbsError;
	extern __device__ int blockCount;
    int i, j, k, m;

    double tmp = 0.0;


    // loop for particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        if (p_rhs.materialId[i] == EOS_TYPE_IGNORE) continue;

        tmp =  pow(p.DISPH_y[i], 10)/p.p[i] - 1;
	localMaxDISPH_PressureAbsError = max(localMaxDISPH_PressureAbsError, tmp);

    }   // loop for particles


    // reduce shared thread results to one per block
    i = threadIdx.x;
    sharedMaxDISPH_PressureAbsError[i] = localMaxDISPH_PressureAbsError;

    for (j = NUM_THREADS_ERRORCHECK / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;

        sharedMaxDISPH_PressureAbsError[i] = localMaxDISPH_PressureAbsError = max(localMaxDISPH_PressureAbsError, sharedMaxDISPH_PressureAbsError[k]);


        }
    }

    // write block result to global memory
    if (i == 0) {
        k = blockIdx.x;
        maxDISPH_PressureAbsErrorPerBlock[k] = localMaxDISPH_PressureAbsError;

        m = gridDim.x - 1;
        if (m == atomicInc((unsigned int *)&blockCount, m)) {
            // last block, so combine all block results
            for (j = 0; j <= m; j++) {
                localMaxDISPH_PressureAbsError = max(localMaxDISPH_PressureAbsError, maxDISPH_PressureAbsErrorPerBlock[j]);

            }
            // (single) max relative error
            maxDISPH_PressureAbsError = localMaxDISPH_PressureAbsError;

            blockCount = 0;   // reset block count
        }
    }
}


#endif


