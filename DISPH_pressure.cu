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

#include "DISPH_pressure.h"
#include "miluph.h"
#include "config_parameter.h"
#include "timeintegration.h"
#include "parameter.h"
#include "tree.h"
#include "pressure.h"
extern __device__ SPH_kernel kernel;
extern __device__ SPH_kernel wendlandc2_p;

__global__ void calculate_pressure_and_DISPH_Y(int *interactions) {


    register int i, inc, matId;
    register double eta, e, rho, mu, p1, p2;

    int j;

    int ip;
    int d;
    double W;
    double Wj;
    double dx[DIM];
    double dWdx[DIM];
    double dWdr;
    double p_stored[numParticles];
    double sml;
    double tolerance;


	
    int cnt = 0;

// Zero step: Calculate p_i = sum_j Y_j W_ij, where the Y are initial guesses obtained by the corresponding differential eq. 
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
            sml = p.h[i];
            for (d = 0; d < DIM; d++) {
                dx[d] = 0;
            }
            kernel(&W, dWdx, &dWdr, dx, sml);

            p.p[i] = p.DISPH_Y[i] * W;

            if (p == 0.0) {
                printf("p is %f W: %e \n", p, W);
            }
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



                p.p[i] += p.DISPH_Y[ip] * W;


            }
            }


// start iteration procedure to solve implicit p-Y-relation
do {

cnt += 1;
tolerance = 0.0;






            
            // Start loop over all particles (the first 3 steps can be done in one loop)
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

            // First step: Calculate rho = m*p/Y
            p.DISPH_rho[i] = p.m[i]*p.p[i]/p.DISPH_Y[i];
            







            // Second step: Calculate pressure by using eos

            //pressure = 0.0;
        matId = p_rhs.materialId[i];
        if (EOS_TYPE_IGNORE == matEOS[matId] || matId == EOS_TYPE_IGNORE) {
            continue;
        }
        if (EOS_TYPE_POLYTROPIC_GAS == matEOS[matId]) {
            p.p[i] = matPolytropicK[matId] * pow(p.DISPH_rho[i], matPolytropicGamma[matId]);
        } else if (EOS_TYPE_IDEAL_GAS == matEOS[matId]) {
            p.p[i] = (matPolytropicGamma[matId] - 1) * p.DISPH_rho[i] * p.e[i];
        } else if (EOS_TYPE_LOCALLY_ISOTHERMAL_GAS == matEOS[matId]) {
            p.p[i] = p.cs[i]*p.cs[i] * p._DISPH_rho[i];
        } else if (EOS_TYPE_ISOTHERMAL_GAS == matEOS[matId]) {
        /* this is pure molecular hydrogen at 10 K */
            p.p[i] = 41255.407 * p._DISPH_rho[i];
        } else if (EOS_TYPE_MURNAGHAN == matEOS[matId] || EOS_TYPE_VISCOUS_REGOLITH == matEOS[matId]) {
            eta = p._DISPH_rho[i] / matRho0[matId];
            if (eta < matRhoLimit[matId]) {
                p.p[i] = 0.0;
            } else {
                p.p[i] = (matBulkmodulus[matId]/matN[matId])*(pow(eta, matN[matId]) - 1.0);
            }
            
        } else if (EOS_TYPE_TILLOTSON == matEOS[matId]) {
            rho = p._DISPH_rho[i];
            e = p.e[i];
            eta = rho / matTillRho0[matId];
            mu = eta - 1.0;
            if (eta < matRhoLimit[matId] && e < matTillEcv[matId]) {
                p.p[i] = 0.0;
            } else {
                if (e <= matTillEiv[matId] || eta >= 1.0) {
                    p.p[i] = (matTilla[matId] + matTillb[matId]/(e/(eta*eta*matTillE0[matId])+1.0))
                        * rho * e + matTillA[matId]*mu + matTillB[matId]*mu*mu;
                } else if (e >= matTillEcv[matId] && eta >= 0.0) {
                    p.p[i] = matTilla[matId]*rho*e + (matTillb[matId]*rho*e/(e/(eta*eta*matTillE0[matId])+1.0)
                        + matTillA[matId] * mu * exp(-matTillBeta[matId]*(matTillRho0[matId]/rho - 1.0)))
                        * exp(-matTillAlpha[matId] * (pow(matTillRho0[matId]/rho-1.0, 2)));
                } else if (e > matTillEiv[matId] && e < matTillEcv[matId]) {
                    // for intermediate states:
                    // weighted average of pressures calculated by expanded
                    // and compressed versions of Tillotson (both evaluated at e)
                    p1 = (matTilla[matId]+matTillb[matId]/(e/(eta*eta*matTillE0[matId])+1.0)) * rho*e
                        + matTillA[matId]*mu + matTillB[matId]*mu*mu;
                    p2 = matTilla[matId]*rho*e + (matTillb[matId]*rho*e/(e/(eta*eta*matTillE0[matId])+1.0)
                        + matTillA[matId] * mu * exp(-matTillBeta[matId]*(matTillRho0[matId]/rho -1.0)))
                        * exp(-matTillAlpha[matId] * (pow(matTillRho0[matId]/rho-1.0, 2)));
                    p.p[i] = ( p1*(matTillEcv[matId]-e) + p2*(e-matTillEiv[matId]) ) / (matTillEcv[matId]-matTillEiv[matId]);
                } else {
                    printf("\n\nDeep trouble in pressure.\nenergy[%d] = %e\nE_iv = %e, E_cv = %e\n\n", i, e, matTillEiv[matId], matTillEcv[matId]);
                    p.p[i] = 0.0;
                }
            
            }
            }
	// Store pressure values for comparison -> tolerance
	p_stored[i] = p.p[i];



            // Third step: Calculate Y = m*p/DISPH_rho

                p.DISPH_Y[i] = p.m[i]*p.p[i]/p.DISPH_rho[i];


            } // end loop over all particles 



            // another loop over all particles
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

            // Fourth step: Calculate p_i = sum_j Y_j W_ij

            sml = p.h[i];
            for (d = 0; d < DIM; d++) {
                dx[d] = 0;
            }
            kernel(&W, dWdx, &dWdr, dx, sml);

            p.p[i] = p.DISPH_Y[i] * W;

            if (p == 0.0) {
                printf("p is %f W: %e \n", p, W);
            }
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



                p.p[i] += p.DISPH_Y[ip] * W;


            }



            // calculate pressure differences
            
            tolerance += fabs((p_stored[i] - p.p[i])/p.p[i]);

            
          
            } // end loop over all particles 


            printf("In Pressure calculation: average pressure deviation is %e", tolerance/numParticles);
    } while (tolerance/numParticles > 1e-2 && cnt < 10);     


    // calculate the DISPH-density from the true values for p and Y
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
            p.DISPH_rho[i] = p.m[i]*p.p[i]/p.DISPH_Y[i];
        }
}
