/**
 * @author      Marius Morlock and Christoph Schaefer
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
#include "viscosity.h"
#include "miluph.h"
#include "soundspeed.h"
#include "timeintegration.h"
#include "config_parameter.h"
#include "kernel.h"
#include "parameter.h"

extern __device__ SPH_kernel kernel;


#if NAVIER_STOKES
__global__ void calculate_kinematic_viscosity(void)
{

	int i, inc;
	inc = blockDim.x * gridDim.x;
    //Particle Loop
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {
#if SHAKURA_SUNYAEV_ALPHA
	    double R = sqrt(p.x[i]*p.x[i] + p.y[i]*p.y[i]);
	    p_rhs.eta[i] = matalpha_shakura[p_rhs.materialId[i]] * p.cs[i] * p.rho[i] * scale_height * R ;
#elif CONSTANT_KINEMATIC_VISCOSITY
	    p_rhs.eta[i] = matnu[p_rhs.materialId[i]] * p.rho[i];
#else
	    printf("aaaaaah\n");
	    assert(0);
#endif
    }
}
#endif

#if NAVIER_STOKES
__global__ void calculate_shear_stress_tensor(int *interactions)
{
    register int64_t interactions_index;
	int i, inc;
    int e, f, g;
    int j, k;
	inc = blockDim.x * gridDim.x;
    //Particle Loop
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {
        double dv[DIM];
        double dr[DIM];
        double r;
        double sml;
        double dWdr, dWdrj, W, Wj;
        double dWdx[DIM], dWdxj[DIM];

        for (k = 0; k < DIM*DIM; k++) {
            p.Tshear[i*DIM*DIM+k] = 0.0;
        }

        for (k = 0; k < p.noi[i]; k++) {
            interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + k;
            j = interactions[interactions_index];

            dv[0] = p.vx[i] - p.vx[j];
#if DIM > 1
            dv[1] = p.vy[i] - p.vy[j];
#if DIM > 2
            dv[2] = p.vz[i] - p.vz[j];
#endif
#endif

            // relative vector
            dr[0] = p.x[i] - p.x[j];
#if DIM > 1
            dr[1] = p.y[i] - p.y[j];
#if DIM > 2
            dr[2] = p.z[i] - p.z[j];
#endif
#endif

            r = 0;
            for (e = 0; e < DIM; e++) {
                r += dr[e]*dr[e];
                dWdx[e] = 0.0;
            }
            W = 0.0;
            dWdr = 0.0;
            r = sqrt(r);

            sml = p.h[i];
#if (VARIABLE_SML || INTEGRATE_SML || DEAL_WITH_TOO_MANY_INTERACTIONS)
            sml = 0.5*(p.h[i] + p.h[j]);
#endif


#if AVERAGE_KERNELS
            // get kernel values for this interaction
            kernel(&W, dWdx, &dWdr, dr, p.h[i]);
            kernel(&Wj, dWdxj, &dWdrj, dr, p.h[j]);
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            Wj /= p_rhs.shepard_correction[j];
            for (e = 0; e < DIM; e++) {
                dWdx[e] /= p_rhs.shepard_correction[i];
                dWdxj[e] /= p_rhs.shepard_correction[j];
            }
            dWdr /= p_rhs.shepard_correction[i];
            dWdrj /= p_rhs.shepard_correction[j];
# endif
            W = 0.5 * (W + Wj);
            dWdr = 0.5 * (dWdr + dWdrj);
            for (e = 0; e < DIM; e++) {
                dWdx[e] = 0.5 * (dWdx[e] + dWdxj[e]);
            }
#else
            // get kernel values for this interaction
            kernel(&W, dWdx, &dWdr, dr, sml);
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            for (e = 0; e < DIM; e++) {
                dWdx[e] /= p_rhs.shepard_correction[i];
            }
            dWdr /= p_rhs.shepard_correction[i];
# endif

#endif


#if TENSORIAL_CORRECTION
            for (e = 0; e < DIM; e++) {
                for (f = 0; f < DIM; f++) {
                    for (g = 0; g < DIM; g++) {
                        p.Tshear[i*DIM*DIM+e*DIM+f] += p.m[j]/p.rho[j] * (p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+f*DIM+g] * (-dv[e]) * dr[g] * dWdr/r + p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+e*DIM+g] * (-dv[f]) * dr[g] * dWdr/r);
                    }
                    // traceless
                    if (e == f) {
                        for (g = 0; g < DIM; g++) {
                            p.Tshear[i*DIM*DIM+e*DIM+f] -= 2./3 * p.m[j]/p.rho[j] * (p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+e*DIM+g] * (-dv[e]) * dr[g] * dWdr/r);
                        }

                    }
                }
            }
#else
            double trace = 0;
            for (e = 0; e < DIM; e++) {
# if (SPH_EQU_VERSION == 1)
                trace +=  p.m[j]/p.rho[i] * (-dv[e])*dWdx[e] ;
# elif (SPH_EQU_VERSION == 2)
                trace +=  p.m[j]/p.rho[j] * (-dv[e])*dWdx[e] ;
#endif
            }

            for (e = 0; e < DIM; e++) {
                for (f = 0; f < DIM; f++) {
# if (SPH_EQU_VERSION == 1)
                    p.Tshear[i*DIM*DIM+e*DIM+f] += p.m[j]/p.rho[i] * (-dv[e]*dWdx[f] - dv[f]*dWdx[e]);
# elif (SPH_EQU_VERSION == 2)
                    p.Tshear[i*DIM*DIM+e*DIM+f] += p.m[j]/p.rho[j] * (-dv[e]*dWdx[f] - dv[f]*dWdx[e]);
#endif
                    // traceless
                    if (e == f) {
# if (SPH_EQU_VERSION == 1)
                        p.Tshear[i*DIM*DIM+e*DIM+f] -= 2./3 * trace;
# elif (SPH_EQU_VERSION == 2)
                        p.Tshear[i*DIM*DIM+e*DIM+f] -= 2./3 * trace;
#endif
                    }
                }
            }
#endif
        }
    }
}
#endif


/*normally known as beta viscosity, here we use beta due to naming concerns of other variables in this code and to avoid confusion */
#if INVISCID_SPH
__global__ void betaviscosity(int *interactions)
{
    register int64_t interactions_index;
	register int d, i, j, k, m, inc, numInteractions;
	inc = blockDim.x * gridDim.x;

	double dv[DIM], dx[DIM];
	double vi[DIM], vj[DIM];
	double mj, rhoi;
	double W, dWdx[DIM], dWdr;
	double divv[DIM], divV, divvdt;
	double tau, sml, csound;
	double Ai, Ri, Xii;
	double curlj[DIM][DIM] = {0};
	double transpose[DIM][DIM] = {0};
	double limiterMatrix[DIM][DIM] = {0};
	double traceLimiter = 0;
	double beta_loc;
	double beta_max = 4;

    //Particle Loop
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {
    	numInteractions = p.noi[i];
    	sml = p.h[i];
    	csound = p.cs[i];
    	rhoi = p.rho[i];
    	W = 0.0;
    	dWdr = 0.0;
    	Ai = 0;
    	Ri = 0;
    	Xii = 0;

        //Interaction Partner Loop
		for(k = 0; k < numInteractions; k++) {
            interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + k;
    		j = interactions[interactions_index];
            dx[0] = p.x[i] - p.x[j];
            dv[0] = p.vx[i] - p.vx[j];
#if DIM > 1
            dx[1] = p.y[i] - p.y[j];
            dv[1] = p.vy[i] - p.vy[j];
#if DIM > 2
            dx[2] = p.z[i] - p.z[j];
            dv[2] = p.vz[i] - p.vz[j];
#endif
#endif
            vi[0] = p.vx[i];
            vj[0] = p.vx[j];
#if DIM > 1
            vi[1] = p.vy[i];
            vj[1] = p.vy[j];
#if DIM > 2
            vi[2] = p.vz[i];
            vj[2] = p.vz[j];
#endif
#endif

	       kernel(&W, dWdx, &dWdr, dx, sml);
#if SHEPARD_CORRECTION
           W /= p_rhs.shepard_correction[i];
           for (d = 0; d < DIM; d++) {
               dWdx[d] /= p_rhs.shepard_correction[i];
           }
           dWdr /= p_rhs.shepard_correction[i];
#endif

	       mj = p.m[j];

	        /* divv */
            for (d = 0; d < DIM; d++) {
            	divv[d] = mj/rhoi * (vj[d] - vi[d]) * dWdx[d];
            /* Limiter Matrix */
    		    for (m = 0; m < DIM; m++) {
                    curlj[d][m] = 0;
    		    }
    	    }


       	    for (d = 0; d < DIM; d++) {
        	    curlj[d][d] = -1.0/DIM * divv[d];
    			for (m = 0; m < DIM; m++) {
        			curlj[d][m] += (mj/rhoi) *(dv[d]*dWdx[d]/(dx[m]*dWdx[m]) + dv[m]*dWdx[m]/(dx[d]*dWdx[d]))/2;
        			transpose[m][d] = curlj[d][m];
    			}
    	   	}

	    	multiply(curlj, transpose, limiterMatrix);
    		for (d = 0; d < DIM; d++) {
    			traceLimiter += limiterMatrix[d][d];
    			Ri += 1.0/rhoi * sign(dv[d]) * mj * dWdx[d];
    		}
    	} /* Ending Interaction Partner Loop */




        /* Calculating d/dt(divv) */
		divV = p_rhs.divv[i];
		divvdt = divV - p.divv_old[i];


		/* decay time */
		tau = sml / (2.0 * 0.05 * csound);

		/* Limiter for strong shear forces */
		Xii = pow(fabs(2 * pow((1.0 - Ri),4) * divV),2) / (pow(fabs(2 * pow((1.0 - Ri),4) * divV),2) + pow(1.0+fabs(traceLimiter),2));
		//fprintf(stdout, "Ri: %e\n", Ri);
		//fprintf(stdout, "vvnablaW: %e\n", vvnablaW);
		//fprintf(stdout, "traceLimiter: %e\n", traceLimiter);
		//fprintf(stdout, "Xii: %e\n", Xii);

		/* Calc shock indicator */
		if ((-1.0*divvdt) > 0) {
			Ai = Xii * -1.0 * divvdt;
		}
		else {
			Ai = 0;
		}
		//fprintf(stdout, "p[%d].divvdt: %e\n", i, p[i].divvdt);
		beta_loc = beta_max * pow(sml, 2) * Ai / (pow(csound, 2) + pow(sml, 2) * Ai);
		//fprintf(stdout, "p[%d].cs: %e\n", i, p[i].cs);
		//fprintf(stdout, "p[%d].sml: %e\n", i, p[i].sml);
		//fprintf(stdout, "Ai: %e\n", Ai);
		//fprintf(stdout, "beta_max: %e\n", beta_max);
		//fprintf(stdout, "beta_loc: %e\n", beta_loc);

		if (beta_loc > p.beta_old[i]) {
			p.beta[i] = beta_loc;
			p.dbetadt[i] = 0;
		} else {
			p.beta[i] = p.beta_old[i];
			p.dbetadt[i] = (p.beta_old[i] - beta_loc) / tau;
		}

		/* getting ready for the next timestep by giving the "new" old values the current ones */
		p.beta_old[i] = p.beta[i];
		p.divv_old[i] = divV;
	} // Ending Particle Loop
}
#endif

__device__ int sign(double x)
{
	if (x < 0) {
		return -1;
	} else if (x > 0) {
		return  1;
	} else {
		return 0;
	}
}

/* function for multiplying two matrices */
__device__ void multiply(double mat1[][DIM], double mat2[][DIM], double res[][DIM])
{
    int i, j, k;
    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            for (k = 0; k < DIM; k++) {
                res[i][j] += mat1[i][k]*mat2[k][j];
            }
    	}
     }
}
