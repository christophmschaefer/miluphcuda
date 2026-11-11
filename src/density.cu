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
#if AVERAGE_KERNELS
    double dWdrj;
#endif
    double rho;
    double sml;
    int iterate;
#if SML_CORRECTION
    double tolerance;
    int cnt;
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
#endif // INTEGRATE_DENSITY
#if SML_CORRECTION
        tolerance = 0.0;
        cnt = 0;
        h_init = p.h[i];
        h_new = 0.0;
        // in case the loop below is left before Omega is computed
        sml_omega = 1.0;
        /* // if Bisection method is used
        double a = 0.0, b = 0.0, c = 0.0;
	    int bis_cnt = 0;
        int bisection = 0; */
#endif // SML_CORRECTION

        do {
            // set to 1 below if another iteration with an updated smoothing length is needed (SML_CORRECTION)
            iterate = 0;
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
#if SML_CORRECTION
            // self-contribution to d rho_i/d h_i: d(m_i W(0, h_i))/d h_i = -DIM m_i W(0, h_i)/h_i
            sml_omega_sum -= p.m[i] * DIM * W / p.h[i];
#endif // SML_CORRECTION
#if !INTEGRATE_DENSITY
            if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[i]] || p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
		p.rho[i] = rho;
                continue;
            }
#endif
            // sph sum for particle i
            for (j = 0; j < p.noi[i]; j++) {
                interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + j;
                ip = interactions[interactions_index];
                if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[ip]] || p_rhs.materialId[ip] == EOS_TYPE_IGNORE) {
                    continue;
                }
                // symmetric in i and j
                sml = 0.5*(p.h[i] + p.h[ip]);

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

            // contribution of interaction
            // for SML_CORRECTION, sml_omega_sum collects d rho_i/d h_i (h_j fixed), with dW/dh = -(DIM W + r dW/dr)/h
            // for W = h^-DIM f(r/h); the h-dependence of the Shepard factor is neglected
#if AVERAGE_KERNELS
                kernel(&W, dWdx, &dWdr, dx, p.h[i]);
                Wj = 0;
                // separate dWdrj, dWdr of h_i is still needed for SML_CORRECTION
                kernel(&Wj, dWdx, &dWdrj, dx, p.h[ip]);
# if SHEPARD_CORRECTION
                W /= p_rhs.shepard_correction[i];
                dWdr /= p_rhs.shepard_correction[i];
                Wj /= p_rhs.shepard_correction[ip];
# endif
# if SML_CORRECTION
                // W_ij = (W(r, h_i) + W(r, h_j))/2, only W(r, h_i) depends on h_i
                sml_omega_sum -= 0.5 * p.m[ip] * (DIM * W + r * dWdr) / p.h[i];
# endif // SML_CORRECTION
                W = 0.5 * (W + Wj);
#else
                kernel(&W, dWdx, &dWdr, dx, sml);
# if SHEPARD_CORRECTION
                W /= p_rhs.shepard_correction[i];
                dWdr /= p_rhs.shepard_correction[i];
# endif
# if SML_CORRECTION
                // W_ij = W(r, h_ij) with h_ij = (h_i + h_j)/2, hence d W_ij/d h_i = 1/2 dW/dh at h = h_ij
                sml_omega_sum -= 0.5 * p.m[ip] * (DIM * W + r * dWdr) / sml;
# endif // SML_CORRECTION
#endif // AVERAGE_KERNELS

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
            tolerance = fabs(h_new - p.h[i]) / h_init;
            // update h only if it is followed by another evaluation of rho and Omega (at most 10 evaluations, i.e.,
            // 9 updates), so that h_i, rho_i, Omega_i and the interaction list of i belong together
            if (tolerance > 1e-3 && cnt < 9) {
                if (h_new < 0){
	       	        printf("SML_CORRECTION: NEGATIVE SML!");
                }
                p.h[i] = h_new;
                redo_NeighbourSearch(i, interactions);
                cnt++;
                iterate = 1;
            }
#endif // SML_CORRECTION
        } while (iterate);
        // write to global memory
        p.rho[i] = rho;
#if SML_CORRECTION
        // store Omega always, also if h was already converged in the first iteration (p.sml_omega is reset to 1
        // in zero_all_derivatives() for every rhs evaluation); Omega belongs to the same h as rho
        p.sml_omega[i] = sml_omega;
#endif // SML_CORRECTION
    }
}
