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

#if DISPH
extern __device__ SPH_kernel kernel;
extern __device__ SPH_kernel wendlandc2_p;


// Calculates p_i = sum_j Y_j W_ij
__global__ void calculate_DISPH_y_DISPH_rho(int *interactions) {


    register int i, inc, matId;
    int j;

    int ip;
    int d;
    double W;
    double Wj;
    double dx[DIM];
    double dWdx[DIM];
    double dWdr;
    double sml;
    int cnt = 0;


            // Start loop over all particles
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
            
            sml = p.h[i];
            
            // self-contribution of particle i
            for (d = 0; d < DIM; d++) {
                dx[d] = 0;
            }
            kernel(&W, dWdx, &dWdr, dx, sml);

            p.DISPH_y[i] = p.DISPH_Y[i] * W;


            // sph sum for particle i over neighbour particles
            for (j = 0; j < p.noi[i]; j++) {
                ip = interactions[i * MAX_NUM_INTERACTIONS + j];



                                dx[0] = p.x[i] - p.x[ip];
                #if DIM > 1
                                dx[1] = p.y[i] - p.y[ip];
                #if DIM > 2
                                dx[2] = p.z[i] - p.z[ip];
                #endif
                #endif  
                kernel(&W, dWdx, &dWdr, dx, sml);
                p.DISPH_y[i] += p.DISPH_Y[ip] * W;


            }


            p.DISPH_rho[i] = p.m[i]*p.DISPH_y[i]/p.DISPH_Y[i];

            } // end loop over all particles 

}


__global__ void calculate_DISPH_Y() {
    register int i, inc;
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
                p.DISPH_Y[i] = p.m[i]*p.DISPH_y[i]/p.DISPH_rho[i];
            }

}



__global__ void calculate_DISPH_dp() {
    double DISPH_alpha = 0.1;
	register int i, inc;
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
                p.DISPH_dp[i] = fabs((pow(p.DISPH_y[i], 1/DISPH_alpha)-p.p[i])/p.p[i]);
	    }
}


__global__ void calculate_DISPH_Y_initial() {
    double DISPH_alpha = 0.1;
	register int i, inc;
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
                p.DISPH_Y[i] = p.m[i]*pow(p.p[i], DISPH_alpha)/p.rho[i];
	    }
}


#endif


