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

#include "kernel.h"
#include "timeintegration.h"
#include "config_parameter.h"
#include "tree.h"
#include "parameter.h"
#include "miluph.h"
#include "linalg.h"
#include "pressure.h"

// for interaction partners less than this value, the tensorial correction matrix
// will be set to the identity matrix (-> disabling the correction factors)
#define MIN_NUMBER_OF_INTERACTIONS_FOR_TENSORIAL_CORRECTION_TO_WORK 0


// pointers for the kernel function
__device__ SPH_kernel kernel;
__device__ SPH_kernel wendlandc2_p = wendlandc2;
__device__ SPH_kernel wendlandc4_p = wendlandc4;
__device__ SPH_kernel wendlandc6_p = wendlandc6;
__device__ SPH_kernel cubic_spline_p = cubic_spline;
__device__ SPH_kernel spiky_p = spiky;



// spiky kernel taken from Desbrun & Cani 1996
__device__ void spiky(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double sml)
{
    int d;
    double r;
    double q;

    r = 0;
    for (d = 0; d < DIM; d++) {
        r += dx[d]*dx[d];
        dWdx[d] = 0;
    }
    r = sqrt(r);
    *dWdr = 0;
    *W = 0;
    q = r/sml;

#if DIM == 1
    printf("Error, this kernel can only be used with DIM == 2,3\n");
    assert(0);
#endif

#if DIM == 2
    if (q > 1) {
        *W = 0;
    } else if (q >= 0.0) {
        *W = 10./(M_PI*sml*sml)*(1-q)*(1-q)*(1-q);
        *dWdr = -30./(M_PI*sml*sml*sml)*(1-q)*(1-q);
    }
#elif DIM == 3
    if (q > 1) {
        *W = 0;
    } else if (q >= 0.0) {
        *W = 15./(M_PI*sml*sml*sml)*(1-q)*(1-q)*(1-q);
        *dWdr = -45/(M_PI*sml*sml*sml*sml)*(1-q)*(1-q);
    }
#endif

    for (d = 0; d < DIM; d++) {
        dWdx[d] = *dWdr/r * dx[d];
    }
}



// *THE* cubic bspline
__device__ void cubic_spline(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double sml)
{
    int d;
    double r;
    double q;
    double f;

    r = 0;
    for (d = 0; d < DIM; d++) {
        r += dx[d]*dx[d];
        dWdx[d] = 0;
    }
    r = sqrt(r);
    *dWdr = 0;
    *W = 0;
    q = r/sml;

    f = 4./3. * 1./sml;
#if DIM > 1
    f = 40./(7*M_PI) * 1./(sml*sml);
#if DIM > 2
    f = 8./M_PI * 1./(sml*sml*sml);
#endif
#endif

    if (q > 1) {
        *W = 0;
        *dWdr = 0.0;
#if !AVERAGE_KERNELS
       // printf("This should never happen, actually.\n");
#endif
    } else if (q > 0.5) {
        *W = 2.*f * (1.-q)*(1.-q)*(1-q);
        *dWdr = -6.*f*1./sml * (1.-q)*(1.-q);
    } else if (q <= 0.5) {
        *W = f * (6.*q*q*q - 6.*q*q + 1.);
        *dWdr = 6.*f/sml * (3*q*q - 2*q);
    }
    for (d = 0; d < DIM; d++) {
        dWdx[d] = *dWdr/r * dx[d];
    }
}

// Wendland C2 from Dehnen & Aly 2012
__device__ void wendlandc2(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double sml)
{
    int d;
    double r;
    double q;

    r = 0;
    for (d = 0; d < DIM; d++) {
        r += dx[d]*dx[d];
        dWdx[d] = 0;
    }
    r = sqrt(r);
    *dWdr = 0;
    *W = 0;

    if (r > sml) {
        *W = 0;
    } else {
        q = r/sml;
#if (DIM == 2)
        *W = 7./(M_PI*sml*sml) * (1-q)*(1-q)*(1-q)*(1-q)  *(1+4*q) * (q < 1);
        *dWdr = -140./(M_PI*sml*sml*sml) * q * (1-q)*(1-q)*(1-q) * (q < 1);
#elif (DIM == 3)
        *W = 21./(2*M_PI*sml*sml*sml) * (1-q)*(1-q)*(1-q)*(1-q)  *(1+4*q) * (q < 1);
        *dWdr = -210./(M_PI*sml*sml*sml*sml) * q * (1-q)*(1-q)*(1-q) * (q < 1);
#elif (DIM == 1)
        *W = 5./(4.*sml) * (1-q)*(1-q)*(1-q)*(1+3*q) * (q < 1);
        *dWdr = -15/(sml*sml) * q * (1-q)*(1-q) * (q < 1);
#endif
        for (d = 0; d < DIM; d++) {
            dWdx[d] = *dWdr/r * dx[d];
        }
    }
}

// Wendland C4 from Dehnen & Aly 2012
__device__ void wendlandc4(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double sml)
{
    int d;
    double r;
    double q;

    r = 0;
    for (d = 0; d < DIM; d++) {
        r += dx[d]*dx[d];
        dWdx[d] = 0;
    }
    r = sqrt(r);
    *dWdr = 0;
    *W = 0;

    if (r > sml) {
        *W = 0;
    } else {
        q = r/sml;
#if (DIM == 2)
        *W = 9./(M_PI*sml*sml) * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1.+6*q+35./3.*q*q) * (q < 1);
        *dWdr = -54./(M_PI*sml*sml*sml) * (1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1.-35.*q*q+105.*q*q*q) * (q< 1);
#elif (DIM == 3)
        *W = 495./(32.*M_PI*sml*sml*sml) * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1.+6.*q+35./3.*q*q) * (q < 1);
        *dWdr = -1485./(16.*M_PI*sml*sml*sml*sml) * (1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1.-35.*q*q+105.*q*q*q) * (q< 1);
#elif (DIM == 1)
        *W = 3./(2.*sml) * (1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1+5*q+8*q*q) * (q < 1);
        *dWdr = -21./(sml*sml) * q * (1-q)*(1-q)*(1-q)*(1-q) * (1+4*q) * (q < 1);
#endif
        for (d = 0; d < DIM; d++) {
            dWdx[d] = *dWdr/r * dx[d];
        }
    }
}


// Wendland C6 from Dehnen & Aly 2012
__device__ void wendlandc6(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double sml)
{
    int d;
    double r;
    double q;

    r = 0;
    for (d = 0; d < DIM; d++) {
        r += dx[d]*dx[d];
        dWdx[d] = 0;
    }
    r = sqrt(r);
    *dWdr = 0;
    *W = 0;

    if (r > sml) {
        *W = 0;
    } else {
        q = r/sml;
#if (DIM == 2)
        *W =  78./(7.*M_PI*sml*sml) * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1.+8.*q+25.*q*q+32*q*q*q) * (q < 1);

        *dWdr = -1716./(7.*M_PI*sml*sml*sml) * q * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1.+7*q+16*q*q) * (q < 1);
#elif (DIM == 3)
        *W = 1365./(64.*M_PI*sml*sml*sml) * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1.+8.*q+25.*q*q+32*q*q*q) * (q < 1);
        *dWdr = -15015./(32.*M_PI*sml*sml*sml*sml) * q * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) *
            (1.+7*q+16*q*q) * (q < 1);
#elif (DIM == 1)
        *W = 55./(32.*sml) * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (1+7*q+19*q*q+21*q*q*q) * (q < 1);
        *dWdr = -165./(16*sml*sml) * q * (1-q)*(1-q)*(1-q)*(1-q)*(1-q)*(1-q) * (3+18*q+35*q*q) * (q < 1);
#endif
        for (d = 0; d < DIM; d++) {
            dWdx[d] = *dWdr/r * dx[d];
        }
    }
}



#if ARTIFICIAL_STRESS
// prepares particles for the tensile instability fix
// see monaghan jcp 159 (2000)
__device__ double fixTensileInstability(int a, int b)
{
    int d;
    double hbar;
    double dx[DIM];
    double W;
    double W2;
    double dWdr;
    double dWdx[DIM];

    W = 0;
    W2 = 0;
    dWdr = 0;
    for (d = 0; d < DIM; d++) {
        dx[d] = 0.0;
        dWdx[d] = 0;
    }
    dx[0] = p.x[a] - p.x[b];
#if DIM > 1
    dx[1] = p.y[a] - p.y[b];
#if DIM > 2
    dx[2] = p.z[a] - p.z[b];
#endif
#endif

    hbar = 0.5 * (p.h[a] + p.h[b]);
    // calculate kernel for r and particle_distance
    //kernel(distance, hbar);
    kernel(&W, dWdx, &dWdr, dx, hbar);
    dx[0] = matmean_particle_distance[p_rhs.materialId[a]];
    for (d = 1; d < DIM; d++) {
        dx[d] = 0;
    }
    kernel(&W2, dWdx, &dWdr, dx, hbar);
    //printf("++++++++++++++ %.17lf\n", W/W2);
    return W/W2;
}
#endif // ARTIFICIAL_STRESS


#if (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH || INTEGRATE_ENERGY)
__global__ void CalcDivvandCurlv(int *interactions)
{
    register int64_t interactions_index;
    int i, inc, j, k, m, d, dd;
    /* absolute values of div v and curl v */
    double divv;
    double curlv[DIM];
    double W, dWdr;
    double Wj, dWdrj, dWdxj[DIM];
    double dWdx[DIM], dx[DIM];
    double sml;
    double vi[DIM], vj[DIM];
    double r;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[i]] || p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
               continue;
        }
        k = p.noi[i];
        divv = 0;
        for (m = 0; m < DIM; m++) {
            curlv[m] = 0;
            dWdx[m] = 0;
        }
        sml = p.h[i];
        /* interaction partner loop */
        for (m = 0; m < k; m++) {
            interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + m;
            j = interactions[interactions_index];
            /* get the kernel values */
#if VARIABLE_SML
            sml = 0.5 *(p.h[i] + p.h[j]);
#endif
            dx[0] = p.x[i] - p.x[j];
#if DIM > 1
            dx[1] = p.y[i] - p.y[j];
#if DIM > 2
            dx[2] = p.z[i] - p.z[j];
#endif
#endif


#if AVERAGE_KERNELS
            kernel(&W, dWdx, &dWdr, dx, p.h[i]);
            kernel(&Wj, dWdxj, &dWdrj, dx, p.h[j]);
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            Wj /= p_rhs.shepard_correction[j];
            for (d = 0; d < DIM; d++) {
                dWdx[d] /= p_rhs.shepard_correction[i];
                dWdxj[d] /= p_rhs.shepard_correction[j];
            }
# endif
            W = 0.5 * (W + Wj);
            for (d = 0; d < DIM; d++) {
                dWdx[d] = 0.5 * (dWdx[d] + dWdxj[d]);
            }
#else
            kernel(&W, dWdx, &dWdr, dx, sml);
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            for (d = 0; d < DIM; d++) {
                dWdx[d] /= p_rhs.shepard_correction[i];
            }
# endif
#endif // AVERAGE_KERNELS

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
            r = 0;
            for (d = 0; d < DIM; d++) {
                r += dx[d]*dx[d];
            }
            r = sqrt(r);
            /* divv */
            for (d = 0; d < DIM; d++) {
#if TENSORIAL_CORRECTION
                for (dd = 0; dd < DIM; dd++) {
                    divv += p.m[j]/p.rho[j] * (vj[d] - vi[d]) * p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+d*DIM+dd] * dWdx[dd];
                }
#else
                divv += p.m[j]/p.rho[j] * (vj[d] - vi[d]) * dWdx[d];
#endif

            }
            /* curlv */
#if (DIM == 1 && BALSARA_SWITCH)
#error unset BALSARA SWITCH in 1D
#elif DIM == 2
            // only one component in 2D
            curlv[0] += p.m[j]/p.rho[i] * ((vi[0] - vj[0]) * dWdx[1]
                        - (vi[1] - vj[1]) * dWdx[0]);
            curlv[1] = 0;
#elif DIM == 3
            curlv[0] += p.m[j]/p.rho[i] * ((vi[1] - vj[1]) * dWdx[2]
                        - (vi[2] - vj[2]) * dWdx[1]);
            curlv[1] += p.m[j]/p.rho[i] * ((vi[2] - vj[2]) * dWdx[0]
                        - (vi[0] - vj[0]) * dWdx[2]);
            curlv[2] += p.m[j]/p.rho[i] * ((vi[0] - vj[0]) * dWdx[1]
                        - (vi[1] - vj[1]) * dWdx[0]);
#endif
        }
        for (d = 0; d < DIM; d++) {
            p_rhs.curlv[i*DIM+d] = curlv[d];
        }
            p_rhs.divv[i] = divv;
    }
}
#endif //  (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH)

#if SHEPARD_CORRECTION
// this adds zeroth order consistency but needs one more loop over all neighbours
__global__ void shepardCorrection(int *interactions) {

    register int64_t interactions_index;
    register int i, inc, j, m;
    register double dr[DIM], h, dWdr;
    inc = blockDim.x * gridDim.x;
    double W, dWdx[DIM], Wj;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        double shepard_correction;
        W = 0;
        for (m = 0; m < DIM; m++) {
            dr[m] = 0.0;
        }
        kernel(&W, dWdx, &dWdr, dr, p.h[i]);
        shepard_correction = p.m[i]/p.rho[i]*W;

        for (m = 0; m < p.noi[i]; m++) {
            W = 0;
            interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + m;
            j = interactions[interactions_index];
            if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[j]] || p_rhs.materialId[j] == EOS_TYPE_IGNORE) {
                continue;
            }
            dr[0] = p.x[i] - p.x[j];
#if DIM > 1
            dr[1] = p.y[i] - p.y[j];
#if DIM > 2
            dr[2] = p.z[i] - p.z[j];
#endif
#endif

#if AVERAGE_KERNELS
            kernel(&W, dWdx, &dWdr, dr, p.h[i]);
            Wj = 0;
            kernel(&Wj, dWdx, &dWdr, dr, p.h[j]);
            W = 0.5*(W + Wj);
#else
            h = 0.5*(p.h[i] + p.h[j]);
            kernel(&W, dWdx, &dWdr, dr, h);
#endif

            shepard_correction += p.m[j]/p.rho[j]*W;
        }
        p_rhs.shepard_correction[i] = shepard_correction;
        //printf("%g\n", shepard_correction);
    }
}
#endif






#if TENSORIAL_CORRECTION
// this adds first order consistency but needs one more loop over all neighbours
__global__ void tensorialCorrection(int *interactions)
{
    register int64_t interactions_index;
    register int i, inc, j, k, m;
    register int d, dd;
    int rv = 0;
    inc = blockDim.x * gridDim.x;
    register double r, dr[DIM], h, dWdr, tmp, f1, f2;
    double W, dWdx[DIM];
    double Wj, dWdxj[DIM];
    double wend_f, wend_sml, q, distance;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        register double corrmatrix[DIM*DIM];
        register double matrix[DIM*DIM];
        for (d = 0; d < DIM*DIM; d++) {
            corrmatrix[d] = 0;
            matrix[d] = 0;
        }
        if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[i]] || p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
               continue;
        }

        k = p.noi[i];

        // loop over all interaction partner
        for (m = 0; m < k; m++) {
            interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + m;
            j = interactions[interactions_index];
            if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[j]] || p_rhs.materialId[j] == EOS_TYPE_IGNORE) {
                continue;
            }
            dr[0] = p.x[i] - p.x[j];
#if DIM > 1
            dr[1] = p.y[i] - p.y[j];
#if DIM == 3
            dr[2] = p.z[i] - p.z[j];
            r = sqrt(dr[0]*dr[0]+dr[1]*dr[1]+dr[2]*dr[2]);
#elif DIM == 2
            r = sqrt(dr[0]*dr[0]+dr[1]*dr[1]);
#endif
#endif

#if AVERAGE_KERNELS
            kernel(&W, dWdx, &dWdr, dr, p.h[i]);
            kernel(&Wj, dWdxj, &dWdr, dr, p.h[j]);
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            Wj /= p_rhs.shepard_correction[j];
            for (d = 0; d < DIM; d++) {
                dWdx[d] /= p_rhs.shepard_correction[i];
                dWdxj[d] /= p_rhs.shepard_correction[j];
            }
            for (d = 0; d < DIM; d++) {
                dWdx[d] = 0.5 * (dWdx[d] + dWdxj[d]);
            }
            W = 0.5 * (W + Wj);
# endif


#else
            h = 0.5*(p.h[i] + p.h[j]);
            kernel(&W, dWdx, &dWdr, dr, h);
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            for (d = 0; d < DIM; d++) {
                dWdx[d] /= p_rhs.shepard_correction[i];
            }
# endif
#endif // AVERAGE_KERNELS

            for (d = 0; d < DIM; d++) {
                for (dd = 0; dd < DIM; dd++) {
                    corrmatrix[d*DIM+dd] -= p.m[j]/p.rho[j] * dr[d] * dWdx[dd];
                }
            }
        } // end loop over interaction partners

        rv = invertMatrix(corrmatrix, matrix);
        // if something went wrong during inversion, use identity matrix
        if (rv < 0 || k < MIN_NUMBER_OF_INTERACTIONS_FOR_TENSORIAL_CORRECTION_TO_WORK) {
            #if DEBUG_LINALG
            if (threadIdx.x == 0) {
                printf("could not invert matrix: rv: %d and k: %d\n", rv, k);
                for (d = 0; d < DIM; d++) {
                    for (dd = 0; dd < DIM; dd++) {
                        printf("%e\t", corrmatrix[d*DIM+dd]);
                    }
                        printf("\n");
                }
            }
            #endif
            #if 0 //  deactivation is turned off, cms 2023-10-19. implement munroe
            printf("Deactivating particle %d due to matrix inversion problems\n", i);
            p_rhs.deactivate_me_flag[i] = TRUE; // particle is deactivated and the whole rhs step is redone with a shorter timestep
            #endif
            for (d = 0; d < DIM; d++) {
                for (dd = 0; dd < DIM; dd++) {
                    matrix[d*DIM+dd] = 0.0;
                    if (d == dd)
                        matrix[d*DIM+dd] = 1.0;
                }
            }
        }
        for (d = 0; d < DIM*DIM; d++) {
            p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+d] = matrix[d];

        }
    }
}
#endif
