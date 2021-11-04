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


    int i, inc;
    int j;
    int test_index = 1;
    double DISPH_alpha = 0.1;
    int ip;
    int d;
    double W;
    double Wj;
    double dx[DIM];
    double dWdx[DIM];
    double dWdr;
    double sml;
    int cnt = 0;
    register double y;
    register double rho;
    int matId;


            // Start loop over all particles
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
            
    	matId = p_rhs.materialId[i];
//        if (matId == BOUNDARY_PARTICLE_ID) {
//		p.DISPH_y[i] = 0.98818;
//		p.DISPH_rho[i] = 1.2;
//		continue;
//	}
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
	    	//printf("y is %f W: %e \n", y, W);
                y += p.DISPH_Y[ip] * W;

	    if (p.DISPH_Y[ip] < 0.00001) {
   //            	printf("In calc_y_and_rho: Y =  %f W: %e \n", p.DISPH_Y[ip], W);
            }

            }
	    if (y < 0.0001) {
     //           printf("In calc_y_and_rho: y is %f W: %e \n", y, W);
            }
	    // write to global memory
	    p.DISPH_y[i] = y;

		//printf("i = %e  p.DISPH_y[i] = %e \n", i, p.DISPH_y[i]);
            rho = p.m[i]*p.DISPH_y[i]/p.DISPH_Y[i];
            if (rho < matRho0[matId]*1.1) {
		    rho = matRho0[matId]*1.1;
            //   printf("rho is %f W: %e \n", rho, W);
            }
		//printf("i = %e  p.DISPH_rho[i] = %e \n", i, p.DISPH_rho[i]);
	    p.DISPH_rho[i] = rho;
            } // end loop over all particles
	  //  printf("i = %i \n p.DISPH_y[i] = %e \n  p.DISPH_rho[i] = %e \n", i, y, rho);
	    
}


__global__ void calculate_DISPH_Y() {
    register int i, inc;
    // int matId;
    double DISPH_alpha = 0.1;
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
  //  matId = p_rhs.materialId[i];
//	if (matId == BOUNDARY_PARTICLE_ID) {
//		p.DISPH_Y[i] = 0.000617613;
//		continue;
//	}
                p.DISPH_Y[i] = (p.m[i]*pow(p.p[i], DISPH_alpha))/p.DISPH_rho[i];

	    if (p.DISPH_y[i] < 0.0001) {
 //               printf("In calc_Y: y =: %e \n", p.DISPH_y[i]);
            }
            }

}



//__global__ void calculate_DISPH_dp() {
//    double DISPH_alpha = 0.1;
//    int test_index = 1;
//	register int i, inc;
//            inc = blockDim.x * gridDim.x;
//            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
//                p.DISPH_dp[i] = fabs((pow(p.DISPH_y[i], 1/DISPH_alpha)-p.p[i])/p.p[i]);
//	    }
//}


__global__ void determine_max_dp(double *maxDISPH_PressureAbsErrorPerBlock)
{
    __shared__ double sharedMaxDISPH_PressureAbsError[NUM_THREADS_ERRORCHECK];
    double localMaxDISPH_PressureAbsError = 0.0;
extern __device__ double maxDISPH_PressureAbsError;
extern __device__ int blockCount;
    double DISPH_alpha = 0.1;
    int i, j, k, m;

    double tmp = 0.0;


	   // printf("number Particles: %i \n", numParticles);
    // loop for particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        if (p_rhs.materialId[i] == EOS_TYPE_IGNORE) continue;

        tmp =  pow(p.DISPH_y[i], 10)/p.p[i] - 1;
	//printf("in determine_max_dp: tmp = %e , p = %e \n", tmp, p.p[i]);
	localMaxDISPH_PressureAbsError = max(localMaxDISPH_PressureAbsError, tmp);
	//printf("localMaxDISPH_PressureAbsError = %e \n", localMaxDISPH_PressureAbsError); 

    }   // loop for particles


    // reduce shared thread results to one per block
    i = threadIdx.x;
    sharedMaxDISPH_PressureAbsError[i] = localMaxDISPH_PressureAbsError;
	//printf("i = %i, shared ...= %e \n", i, sharedMaxDISPH_PressureAbsError[i]);

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


