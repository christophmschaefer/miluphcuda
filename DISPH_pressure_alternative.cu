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

__global__ void calculate_pressure_and_DISPH_Y_alternative(int *interactions) {


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
    double tolerance_max;
    double tolerance_array[numParticles];
    int cnt = 0;

// start iteration procedure to solve implicit p-Y-relation
do {

            



            
            
            // Start loop over all particles (the first 3 steps can be done in one loop)
            inc = blockDim.x * gridDim.x;
            for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
            


            // First step: Calculate p_i = sum_j Y_j W_ij


		// Store pressure values for comparison -> tolerance
				p_stored[i] = p.p[i];

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
            
            tolerance_array[i] = fabs((p_stored[i] - p.p[i])/p_stored[i])
            
            
          
            }








            // Second step: Calculate Y(p) by using eos
		inc = blockDim.x * gridDim.x;
		for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
			    
            //pressure = 0.0;
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
                p.DISPH_Y[i] = p.m[i]*p.p[i]/matRho0[matId] * 1/(pow(1 + matN[matId]*p.p[i]/matBulkmodulus[matId], 1/matN[matId]);
        
        // For Tillotson Y(p) has to be calculated numerically
        } else if (EOS_TYPE_TILLOTSON == matEOS[matId]) {
        double ui;
        double u_iv;
        double u_cv;
        double u0;
        double rho0_i;
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
        
            ui = p.e[i];
            u_iv = matTillEiv[matId];
            u_cv = matTillEcv[matId];
            u0 = matTillE0[matId];
            rho0_i = matTillRho0[matId];
            
            if (eta < matRhoLimit[matId] && ui < matTillEcv[matId]) {
                p.DISPH_Y[i] = 0.0;
            } else {
            
            	// Newton method for calculating DISPH_rho first, then Y = m*p/rho
            	accuracy_newton = 1.0e-6;
            	
            	// Y comes from the solution of dY/dt = ..., using factor 2 in rho to ensure that the higher rho will be determined
            	rho = 2*p.m[i]*p.p[i]/p.DISPH_Y[i];
        	rho_guess = 1.0e8*rho;
        	int it = 0
        	
            // Newton loop
        	do {
		    it += 1;
		    rho_guess = rho;

		    // calc p_eos(rho_guess, u) - p_i
		    
				T_eta = rho_guess/rho0_i;
				T_mu = T_eta - 1;
				if  (T_eta >= 1.0 || ui <= u_iv){
				    eos_minus_p = (matTilla[matId] + matTillb[matId]/(ui/(u0*T_eta**2) + 1))*rho_guess * ui + matTillA[matId]*T_mu + matTillB[matId]* T_mu**2 - p.p[i];
				    }
				else if (T_eta >=0.0 && ui >= u_cv){
				    eos_minus_p = matTilla[matId]*rho_guess*ui + (matTillb[matId]*rho_guess*ui/(ui/(u0*T_eta**2) + 1) + matTillA[matId]*T_mu *exp(-matTillAlpha[matId]*(1/T_eta - 1)))*exp(-matTillBeta[matId]*(1/T_eta - 1)**2) - p.p[i];
				    }
				else if (u_iv < ui && ui < u_cv){
				    eos_minus_p = ((ui - u_iv)*(matTilla[matId]*rho_guess*ui + (matTillb[matId]*rho_guess*ui/(ui/(u0*T_eta**2) + 1) + matTillA[matId]*T_mu *exp(-matTillAlpha[matId]*(1/T_eta - 1)))*exp(-matTillBeta[matId]*(1/T_eta - 1)**2)) + (u_cv - ui)* ((matTilla[matId] + matTillb[matId]/(ui/(u0*T_eta**2) + 1))*rho_guess * ui + matTillA[matId]*T_mu + matTillB[matId]* T_mu**2))/(u_cv - u_iv) - p.p[i];
				}
			    
			    
		// calc derivative of calc p_eos(rho_guess, u) - p_i
			    k = ui*rho0_i**2/u0;
			    ex_a = exp(-matTillAlpha[matId]*(1/T_eta - 1));
			    ex_b = exp(-matTillBeta[matId]*(1/T_eta - 1)**2);
			    if  (T_eta >= 1 || ui <= u_iv){
				der1 = (2*matTillB[matId]*T_mu + matTillA[matId])/rho0_i + (2*matTillb[matId]*k*ui)/(rho_guess**2*(k/rho_guess**2 + 1)**2) + ui*(matTillb[matId]/(k/rho_guess**2 + 1) + matTilla[matId]);
				eos_minus_p_der = der1;
				}
			    else if (T_eta >=0.0 && ui >= u_cv){
				der2 = ex_b*(matTillA[matId]*((matTillAlpha[matId]*rho0_i*ui)/rho_guess**2 + 1/rho0_i)*ex_a + (matTillb[matId]*ui*(3*k/rho_guess**2 + 1))/(k/rho_guess**2 + 1)**2) + (2*matTillBeta[matId]*rho0_i*T_mu)/rho_guess**2 * ex_b * (matTillA[matId]*T_mu*ex_a + (matTillb[matId]*ui*rho_guess)/(k/rho_guess**2 + 1)) + matTilla[matId]*ui;
				eos_minus_p_der = der2;
				}
			    else if (u_iv < ui && ui < u_cv){
				der1 = (2*matTillB[matId]*T_mu + matTillA[matId])/rho0_i + (2*b*k*ui)/(rho_guess**2*(k/rho_guess**2 + 1)**2) + ui*(matTillb[matId]/(k/rho_guess**2 + 1) + matTilla[matId]);
				der2 = ex_b*(matTillA[matId]*((matTillAlpha[matId]*rho0_i*ui)/rho_guess**2 + 1/rho0_i)*ex_a + (matTillb[matId]*ui*(3*k/rho_guess**2 + 1))/(k/rho_guess**2 + 1)**2) + (2*matTillBeta[matId]*rho0_i*T_mu)/rho_guess**2 * ex_b * (matTillA[matId]*T_mu*ex_a + (matTillb[matId]*ui*rho_guess)/(k/rho_guess**2 + 1)) + matTilla[matId]*ui;
				eos_minus_p_der = ((ui-u_iv)*der2 + (u_cv - ui)*der1)/(u_cv - u_iv);
		    } else {
                    printf("\n\nDeep trouble in pressure.\nenergy[%d] = %e\nE_iv = %e, E_cv = %e\n\n", i, e, matTillEiv[matId], matTillEcv[matId]);
                    p.DISPH_Y[i] = 0.0;
                }
		    
		    
		    
		    
		    
		    
		    
		    rho = rho_guess - eos_minus_p/eos_minus_p_der;
		        
            	} while (fabs((rho - rho_guess)/rho)>accuracy_newton && it < 50);
            	
   		p.DISPH_Y[i] = p.m[i]*p.p[i]/rho
            
            } 
            } // end if tillotson
	


            } // end loop over all particles 






            

    // find maximum of tolerance_array
    tolerance_max = tolerance_array[0]

    for(i=1; i<numParticles; i++)
    {
         if(tolerance_max<tolerance_array[i])
            {
                tolerance_max=tolerance_array[i];    
            }   
    }

	
	cnt += 1
    } while (tolerance_max > 1e-2 && cnt < 50);     

}
