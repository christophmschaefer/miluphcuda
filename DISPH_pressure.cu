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
    double p_initial;
    double dp;
    double max_dp;
    double sml;
    int cnt = 0;

// start iteration procedure to solve implicit p-Y-relation
do {

            
            max_dp = 0.0;

            // Start loop over all particles
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
            


            // First step: Calculate p_i = sum_j Y_j W_ij


		// Store pressure values for comparison -> tolerance
				p_initial = p.p[i];
                


            sml = p.h[i];
            
            // self-contribution of particle i
            for (d = 0; d < DIM; d++) {
                dx[d] = 0;
            }
            kernel(&W, dWdx, &dWdr, dx, sml);

            p.p[i] = p.DISPH_Y[i] * W;


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
            
            dp = fabs((p_initial - p.p[i])/p_initial);

            if (dp > max_dp) {
                max_dp = dp;
                }



            // Second step: Calculate Y(p) by using eos

        matId = p_rhs.materialId[i];
        if (EOS_TYPE_IGNORE == matEOS[matId] || matId == EOS_TYPE_IGNORE) {
            continue;
        }
        // For Polytropic gas: Y(p) = K^(1/gamma) * m * p^(1-1/gamma)
        if (EOS_TYPE_POLYTROPIC_GAS == matEOS[matId]) {
            p.DISPH_Y[i] = pow(matPolytropicK[matId], 1/matPolytropicGamma[matId]) *p.m[i]* pow(p.p[i], 1 - 1/matPolytropicGamma[matId]);

        } else if (EOS_TYPE_IDEAL_GAS == matEOS[matId]) {
            // nothing has to be done here

        } else if (EOS_TYPE_LOCALLY_ISOTHERMAL_GAS == matEOS[matId]) {
           // nothing has to be done here

        } else if (EOS_TYPE_ISOTHERMAL_GAS == matEOS[matId]) {
        /* this is pure molecular hydrogen at 10 K */
            // nothing has to be done here

        // For Murnaghan: Y(p) = m*p/rho0 * (K/(np+K))^(1/n)
        } else if (EOS_TYPE_MURNAGHAN == matEOS[matId] || EOS_TYPE_VISCOUS_REGOLITH == matEOS[matId]) {
                p.DISPH_Y[i] = p.m[i]*p.p[i]/matRho0[matId] * 1/pow(1 + matN[matId]*p.p[i]/matBulkmodulus[matId], 1/matN[matId]);
        
        // For Tillotson Y(p) has to be calculated numerically
        } else if (EOS_TYPE_TILLOTSON == matEOS[matId]) {

        double rho;
        double rho_guess;
        double T_eta;
        double T_mu;
        double ex_a;
        double ex_b;
        double eos_minus_p;
        double eos_minus_p_der;
        double accuracy_newton;
        double k;
            
            
            	// Newton method for calculating DISPH_rho first, then Y = m*p/rho
            	accuracy_newton = 1.0e-6;
            	
            	// Y comes from the solution of dY/dt = ..., using factor 2 in rho to ensure that the higher rho will be determined
            	rho = 2*p.m[i]*p.p[i]/p.DISPH_Y[i];
        	rho_guess = 1.0e8*rho;
        	int it = 0;
        	
            // Newton loop
        	do {

		    it += 1;
		    rho_guess = rho;

		    // calc p_eos(rho_guess, u) - p_i
				T_eta = rho_guess/matTillRho0[matId];
				T_mu = T_eta - 1;
				if  (T_eta >= 1.0 || p.e[i] <= matTillEiv[matId]){
				    eos_minus_p = (matTilla[matId] + matTillb[matId]/(p.e[i]/(matTillE0[matId]*pow(T_eta, 2)) + 1))*rho_guess * p.e[i] + matTillA[matId]*T_mu + matTillB[matId]* pow(T_mu,2) - p.p[i];
				    }
				else if (T_eta >=0.0 && p.e[i] >= matTillEcv[matId]){
				    eos_minus_p = matTilla[matId]*rho_guess*p.e[i] + (matTillb[matId]*rho_guess*p.e[i]/(p.e[i]/(matTillE0[matId]*pow(T_eta,2)) + 1) + matTillA[matId]*T_mu *exp(-matTillAlpha[matId]*(1/T_eta - 1)))*exp(-matTillBeta[matId]*pow((1/T_eta - 1),2)) - p.p[i];
				    }
				else if (matTillEiv[matId] < p.e[i] && p.e[i] < matTillEcv[matId]){
				    eos_minus_p = ((p.e[i] - matTillEiv[matId])*(matTilla[matId]*rho_guess*p.e[i] + (matTillb[matId]*rho_guess*p.e[i]/(p.e[i]/(matTillE0[matId]*pow(T_eta,2)) + 1) + matTillA[matId]*T_mu *exp(-matTillAlpha[matId]*(1/T_eta - 1)))*exp(-matTillBeta[matId]*pow((1/T_eta - 1),2))) + (matTillEcv[matId] - p.e[i])* ((matTilla[matId] + matTillb[matId]/(p.e[i]/(matTillE0[matId]*pow(T_eta,2)) + 1))*rho_guess * p.e[i] + matTillA[matId]*T_mu + matTillB[matId]* pow(T_mu,2)))/(matTillEcv[matId] - matTillEiv[matId]) - p.p[i];
				}
			    
			    
		// calc derivative of calc p_eos(rho_guess, u) - p_i
			    k = p.e[i]*pow(matTillRho0[matId],2)/matTillE0[matId];
			    ex_a = exp(-matTillAlpha[matId]*(1/T_eta - 1));
			    ex_b = exp(-matTillBeta[matId]*pow((1/T_eta - 1),2));

			    if  (T_eta >= 1 || p.e[i] <= matTillEiv[matId]){
				double der1;
				der1 = (2*matTillB[matId]*T_mu + matTillA[matId])/matTillRho0[matId] + (2*matTillb[matId]*k*p.e[i])/(pow(rho_guess,2)*pow((k/pow(rho_guess,2) + 1),2)) + p.e[i]*(matTillb[matId]/(k/pow(rho_guess,2) + 1) + matTilla[matId]);
				eos_minus_p_der = der1;
				}

			    else if (T_eta >=0.0 && p.e[i] >= matTillEcv[matId]){
				double der2;    
				der2 = ex_b*(matTillA[matId]*((matTillAlpha[matId]*matTillRho0[matId]*p.e[i])/pow(rho_guess,2) + 1/matTillRho0[matId])*ex_a + (matTillb[matId]*p.e[i]*(3*k/pow(rho_guess,2) + 1))/pow((k/pow(rho_guess,2) + 1),2)) + (2*matTillBeta[matId]*matTillRho0[matId]*T_mu)/pow(rho_guess,2) * ex_b * (matTillA[matId]*T_mu*ex_a + (matTillb[matId]*p.e[i]*rho_guess)/(k/pow(rho_guess,2) + 1)) + matTilla[matId]*p.e[i];
				eos_minus_p_der = der2;
				}

			    else if (matTillEiv[matId] < p.e[i] && p.e[i] < matTillEcv[matId]){
				double der1;
				double der2;
				der1 = (2*matTillB[matId]*T_mu + matTillA[matId])/matTillRho0[matId] + (2*matTillb[matId]*k*p.e[i])/(pow(rho_guess,2)*pow((k/pow(rho_guess,2) + 1),2)) + p.e[i]*(matTillb[matId]/(k/pow(rho_guess,2) + 1) + matTilla[matId]);
				der2 = ex_b*(matTillA[matId]*((matTillAlpha[matId]*matTillRho0[matId]*p.e[i])/pow(rho_guess,2) + 1/matTillRho0[matId])*ex_a + (matTillb[matId]*p.e[i]*(3*k/pow(rho_guess,2) + 1))/pow((k/pow(rho_guess,2) + 1),2)) + (2*matTillBeta[matId]*matTillRho0[matId]*T_mu)/pow(rho_guess,2) * ex_b * (matTillA[matId]*T_mu*ex_a + (matTillb[matId]*p.e[i]*rho_guess)/(k/pow(rho_guess,2) + 1)) + matTilla[matId]*p.e[i];
				eos_minus_p_der = ((p.e[i]-matTillEiv[matId])*der2 + (matTillEcv[matId] - p.e[i])*der1)/(matTillEcv[matId] - matTillEiv[matId]);

		        } else {
                    printf("\n\nDeep trouble in pressure.\nenergy[%d] = %e\nE_iv = %e, E_cv = %e\n\n", i, e, matTillEiv[matId], matTillEcv[matId]);
                }
		    
		    
		    
		    rho = rho_guess - eos_minus_p/eos_minus_p_der;
		        
            	} while (fabs((rho - rho_guess)/rho)>accuracy_newton && it < 50);
            	
   		p.DISPH_Y[i] = p.m[i]*p.p[i]/rho;
            
            } // end if tillotson
	


            } // end loop over all particles 

	
	cnt += 1;
    } while (max_dp > 1e-3 && cnt < 100);     
    printf("\n\n max_dp is \n %e \n", max_dp);
}
