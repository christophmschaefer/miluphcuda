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

#include "density.h"
#include "miluph.h"
#include "config_parameter.h"
#include "timeintegration.h"
#include "parameter.h"
#include "pressure.h"
#include "tree.h"

extern __device__ SPH_kernel kernel;
extern __device__ SPH_kernel wendlandc2_p;
#if SML_CORRECTION
extern __device__ void redo_NeighbourSearch(int particle_id, int *interactions);
#endif // SML_CORRECTION

// calculates the density of all particles via the kernel sum
// is also called for INTEGRATE_DENSITY to determine the densities of particles
// of materials with density_via_kernel_sum = 1 in material.cfg
__global__ void calculateDensity(int *interactions) {
    register int64_t interactions_index;
    int i;
    int j;
    int inc;
    int ip;
    int d;
    double W;
    double Wj;
    double dx[DIM];
    double dWdx[DIM];
    double dWdr;
    double rho;
    double sml;
    double tolerance;
#if SML_CORRECTION
    double dhdrho, sml_omega,sml_omega_sum, r;
    double f, df, h_new, h_init, rho_h;
    //the proportionality constant (h_fact = 4.0) defines the average number of neighbours: [2D] noi = pi * h_fact^2, [3D] noi = 4/3 * pi * h_fact^3
    double h_fact = 4.0;
#endif // SML_CORRECTION
    
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
#if INTEGRATE_DENSITY
        if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[i]] || p_rhs.materialId[i] == EOS_TYPE_IGNORE || matdensity_via_kernel_sum[p_rhs.materialId[i]] < 1) {
                continue;
        }
#else
        if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[i]] || p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
                continue;
        }
#endif // INTEGRATE_DENSITY
        tolerance = 0.0;
        int cnt = 0;
        
#if SML_CORRECTION
        h_init = p.h[i];
        h_new = 0.0;
        /* // if Bisection method is used
        double a = 0.0, b = 0.0, c = 0.0;
	    int bis_cnt = 0;
        int bisection = 0; */
#endif // SML_CORRECTION

        do {
#if SML_CORRECTION
            sml_omega_sum = 0.0;
#endif // SML_CORRECTION
            sml = p.h[i];

            // self density is m_i W_ii
            for (d = 0; d < DIM; d++) {
                dx[d] = 0;
            }
            kernel(&W, dWdx, &dWdr, dx, sml);
#if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
#endif
            rho = p.m[i] * W;
            if (rho == 0.0) {
                printf("rho is %f W: %e \n", rho, W);
            }
            // sph sum for particle i
            for (j = 0; j < p.noi[i]; j++) {
                interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + j;
                ip = interactions[interactions_index];
                if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[ip]] || p_rhs.materialId[ip] == EOS_TYPE_IGNORE) {
                    continue;
                }
#if (VARIABLE_SML || INTEGRATE_SML || DEAL_WITH_TOO_MANY_INTERACTIONS)
                sml = 0.5*(p.h[i] + p.h[ip]);
#endif

                dx[0] = p.x[i] - p.x[ip];
#if DIM > 1
                dx[1] = p.y[i] - p.y[ip];
#if DIM > 2
                dx[2] = p.z[i] - p.z[ip];
#endif
#endif

#if SML_CORRECTION
                r = 0;
                for (d = 0; d < DIM; d++) {
                    r += dx[d]*dx[d];
                }
                r = sqrt(r);
#endif // SML_CORRECTION

#if AVERAGE_KERNELS
                kernel(&W, dWdx, &dWdr, dx, p.h[i]);
                Wj = 0;
                kernel(&Wj, dWdx, &dWdr, dx, p.h[j]);
# if SHEPARD_CORRECTION
                W /= p_rhs.shepard_correction[i];
                Wj /= p_rhs.shepard_correction[j];
# endif
                W = 0.5 * (W + Wj);
#else
                kernel(&W, dWdx, &dWdr, dx, sml);
# if SHEPARD_CORRECTION
                W /= p_rhs.shepard_correction[i];
# endif
            // contribution of interaction
#endif // AVERAGE_KERNELS

#if SML_CORRECTION
                sml_omega_sum += p.m[ip] * (-1) * (DIM * W/sml + (r / sml) * dWdr);
#endif // SML_CORRECTION
                rho += p.m[ip] * W;
            }
#if SML_CORRECTION
            rho_h = p.m[i] * pow(double(h_fact / p.h[i]), DIM);
            dhdrho = -p.h[i] / (DIM * rho);
            sml_omega = 1 - dhdrho * sml_omega_sum;

            // Newton-Raphson method tolerance e-3 (Phantom)
            f = rho_h - rho;
            df = -DIM * rho / p.h[i] * sml_omega;
            h_new = p.h[i] - f / df;

            // arbitrary set limit for sml change
            if (h_new > 1.2 * p.h[i]) {
                h_new = 1.2 * p.h[i];
            } else if (h_new < 0.8 * p.h[i]) {
                h_new = 0.8 * p.h[i];
            }
/*
            //Bisection method (alternative to NR method)
	        if (cnt == 0 && h_new < 0) {
	            bisection = 1;
	        }
	        if (bisection == 1) {
	    	    if ((f/df) > 0) {
		            if(bis_cnt == 0) {
		    	        b = p.h[i];
	                } else {
		    	        b = c;
		            }
	    	    } else if((f/df) < 0) {
                    if(bis_cnt == 0) {
                    	a = p.h[i];
                    	b = 2.0 * a;
                    } else {
                    	a = c;
                    }
		        }
		        c = 0.5 * (a + b);
                h_init = p.h[i];
                h_new = c;
                bis_cnt++;
	        }
*/
           	tolerance = abs(h_new - p.h[i]) / h_init;
            if (tolerance > 1e-3) {
                if (h_new < 0){
	       	        printf("SML_CORRECTION: NEGATIVE SML!");
                }
                p.h[i] = h_new;
                p.sml_omega[i] = sml_omega;
                redo_NeighbourSearch(i, interactions);
                cnt++;
            }
#endif // SML_CORRECTION
        } while (tolerance > 1e-3 && cnt < 10);       
        // write to global memory
        p.rho[i] = rho;
    }
}
