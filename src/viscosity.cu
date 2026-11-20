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
#include "pressure.h"

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

            // symmetric in i and j
            sml = 0.5*(p.h[i] + p.h[j]);


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
            // contribution of particle j to div v at particle i, with the corrected kernel gradient
            double trace = 0;
            for (e = 0; e < DIM; e++) {
                for (g = 0; g < DIM; g++) {
                    trace += p.m[j]/TC_RHO(i, j) * p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+e*DIM+g]
                        * (-dv[e]) * dr[g] * dWdr/r;
                }
            }

            for (e = 0; e < DIM; e++) {
                for (f = 0; f < DIM; f++) {
                    for (g = 0; g < DIM; g++) {
                        p.Tshear[i*DIM*DIM+e*DIM+f] += p.m[j]/TC_RHO(i, j)
                            * (p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+f*DIM+g] * (-dv[e]) * dr[g] * dWdr/r
                               + p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+e*DIM+g] * (-dv[f]) * dr[g] * dWdr/r);
                    }
                    // traceless: subtract 2/3 div v on the diagonal (not 2/3 d_e v_e)
                    if (e == f) {
                        p.Tshear[i*DIM*DIM+e*DIM+f] -= 2./3 * trace;
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


#if INVISCID_SPH
/*
 * Time-dependent artificial viscosity of Cullen & Dehnen (2010, MNRAS 408, 669), "inviscid SPH".
 *
 * Every particle carries its own viscosity coefficient alpha_i (p.av_alpha, integrated like the other variables),
 * used in the standard viscosity of internalForces,
 *     Pi_ij = (beta_ij mu_ij - alpha_ij cbar_ij) mu_ij / rhobar_ij,  alpha_ij = (alpha_i + alpha_j)/2,
 *     beta_ij = alpha_ij beta/alpha,
 * with alpha and beta of the material (artificial_viscosity in the material config): alpha is the maximum
 * alpha_max, and the ratio beta/alpha is kept (beta_ij = 2 alpha_ij for the default alpha = 1, beta = 2).
 *
 * inviscidSPHSwitch runs at the end of every right-hand side and computes per particle, from its neighbours
 * (boundary and ignored particles excluded; h_i is the smoothing length, i.e. the support radius of the kernel):
 *     V_ab = d_a v_b                       velocity gradient (with the correction matrix if TENSORIAL_CORRECTION)
 *     d(div v)/dt = div a - V_ab V_ba      a: the accelerations of this rhs from internalForces and the point
 *                                          masses (not self-gravity, not the boundary conditions; ghost
 *                                          particles do not contribute to div a)
 *     R_i = 1/rho_i sum_j sign(div v_j) m_j W_ij              (j = i included)
 *     xi_i = |2 (1 - R_i)^4 div v_i|^2 / (|2 (1 - R_i)^4 div v_i|^2 + S_ab S_ab),  S: traceless symmetric part of V
 *     A_i = xi_i max(-d(div v)/dt, 0)                          shock indicator, xi_i suppresses it in shear flows
 *     vsig_i = max_j (cbar_ij - min(0, v_ij.x_ij/|x_ij|))      signal velocity
 *     alpha_loc = max(alpha_min, alpha_max h_i^2 A_i / (vsig_i^2 + h_i^2 A_i))
 *     dalpha_i/dt = (alpha_loc - alpha_i) / tau_i if alpha_i > alpha_loc, else 0;  tau_i = h_i / (2 l vsig_i)
 * alpha_min = INVISCID_SPH_ALPHA_MIN (at most alpha_max), l = INVISCID_SPH_DECAY_L.
 * alpha_loc is stored in p_rhs.av_alpha_loc; after every accepted time step, inviscidSPHRaiseAlpha sets
 * alpha_i = max(alpha_i, alpha_loc): alpha rises at once where a shock approaches and decays smoothly behind it.
 * At the start of the simulation (also when restarting), initInviscidSPH sets alpha_i = alpha_max.
 */
__global__ void inviscidSPHSwitch(int *interactions)
{
    int64_t interactions_index;
    int i, inc, j, k, d, e, matId, matIdj;
    double dx[DIM], dWdx[DIM], dWdr, W, g[DIM], w, r, vij_xij;
    double V[DIM][DIM], diva, divv, trVV, SS, sab, R, xi, A, vsig, cbar, h, tau;
    double alpha_max, alpha_min, alpha_loc, num;
    double vi[DIM], vj[DIM], ai[DIM], aj[DIM];
#if AVERAGE_KERNELS
    double Wj, dWdrj, dWdxj[DIM];
#endif

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {
        p.dav_alphadt[i] = 0.0;
        matId = p_rhs.materialId[i];
        if (matId < 0 || matEOS[matId] == EOS_TYPE_IGNORE) {
            p_rhs.av_alpha_loc[i] = 0.0;
            continue;
        }
        alpha_max = matAlpha[matId];
        alpha_min = fmin(INVISCID_SPH_ALPHA_MIN, alpha_max);
        h = p.h[i];

        vi[0] = p.vx[i];
        ai[0] = p.ax[i];
#if DIM > 1
        vi[1] = p.vy[i];
        ai[1] = p.ay[i];
# if DIM > 2
        vi[2] = p.vz[i];
        ai[2] = p.az[i];
# endif
#endif
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++)
                V[d][e] = 0.0;
            dx[d] = 0.0;
        }
        diva = 0.0;
        vsig = p.cs[i];
        /* self term of R_i */
        W = 0.0;
        dWdr = 0.0;
        kernel(&W, dWdx, &dWdr, dx, h);
#if SHEPARD_CORRECTION
        W /= p_rhs.shepard_correction[i];
#endif
        R = (p_rhs.divv[i] > 0.0 ? 1.0 : (p_rhs.divv[i] < 0.0 ? -1.0 : 0.0)) * p.m[i] * W;

        for (k = 0; k < p.noi[i]; k++) {
            interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + k;
            j = interactions[interactions_index];
            matIdj = p_rhs.materialId[j];
            if (matIdj < 0 || matEOS[matIdj] == EOS_TYPE_IGNORE)
                continue;
            dx[0] = p.x[i] - p.x[j];
            vj[0] = p.vx[j];
            aj[0] = p.ax[j];
#if DIM > 1
            dx[1] = p.y[i] - p.y[j];
            vj[1] = p.vy[j];
            aj[1] = p.ay[j];
# if DIM > 2
            dx[2] = p.z[i] - p.z[j];
            vj[2] = p.vz[j];
            aj[2] = p.az[j];
# endif
#endif
#if GHOST_BOUNDARIES
            /* ghost particles have no accelerations of their own: no contribution to div a */
            if (j >= numRealParticles) {
                for (d = 0; d < DIM; d++)
                    aj[d] = ai[d];
            }
#endif
            /* the same kernel as in internalForces */
            W = 0.0;
            dWdr = 0.0;
#if AVERAGE_KERNELS
            kernel(&W, dWdx, &dWdr, dx, p.h[i]);
            Wj = 0.0;
            dWdrj = 0.0;
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
            for (d = 0; d < DIM; d++)
                dWdx[d] = 0.5 * (dWdx[d] + dWdxj[d]);
#else
            kernel(&W, dWdx, &dWdr, dx, 0.5 * (p.h[i] + p.h[j]));
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            for (d = 0; d < DIM; d++)
                dWdx[d] /= p_rhs.shepard_correction[i];
# endif
#endif
#if TENSORIAL_CORRECTION
            /* gradient as for the strain rate in internalForces: m_j/TC_RHO (q_j - q_i) (C_i grad W) */
            for (d = 0; d < DIM; d++) {
                g[d] = 0.0;
                for (e = 0; e < DIM; e++)
                    g[d] += p_rhs.tensorialCorrectionMatrix[i*DIM*DIM + d*DIM + e] * dWdx[e];
            }
            w = p.m[j] / TC_RHO(i, j);
#else
            for (d = 0; d < DIM; d++)
                g[d] = dWdx[d];
            w = p.m[j] / p.rho[j];
#endif
            r = 0.0;
            vij_xij = 0.0;
            for (d = 0; d < DIM; d++) {
                diva += w * (aj[d] - ai[d]) * g[d];
                for (e = 0; e < DIM; e++)
                    V[d][e] += w * (vj[e] - vi[e]) * g[d];
                r += dx[d] * dx[d];
                vij_xij += (vi[d] - vj[d]) * dx[d];
            }
            R += (p_rhs.divv[j] > 0.0 ? 1.0 : (p_rhs.divv[j] < 0.0 ? -1.0 : 0.0)) * p.m[j] * W;
            r = sqrt(r);
            cbar = 0.5 * (p.cs[i] + p.cs[j]);
            if (r > 0.0)
                cbar -= fmin(0.0, vij_xij / r);
            vsig = fmax(vsig, cbar);
        }
        R /= p.rho[i];
        R = fmax(-1.0, fmin(1.0, R));

        divv = 0.0;
        trVV = 0.0;
        for (d = 0; d < DIM; d++) {
            divv += V[d][d];
            for (e = 0; e < DIM; e++)
                trVV += V[d][e] * V[e][d];
        }
        SS = 0.0;
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                sab = 0.5 * (V[d][e] + V[e][d]) - (d == e ? divv / DIM : 0.0);
                SS += sab * sab;
            }
        }
        num = 2.0 * pow(1.0 - R, 4) * divv;
        num *= num;
        xi = (num + SS > 0.0) ? num / (num + SS) : 1.0;
        A = xi * fmax(-(diva - trVV), 0.0);
        num = vsig * vsig + h * h * A;
        alpha_loc = fmax(alpha_min, num > 0.0 ? alpha_max * h * h * A / num : 0.0);
        p_rhs.av_alpha_loc[i] = alpha_loc;
        if (p.av_alpha[i] > alpha_loc && vsig > 0.0) {
            tau = h / (2.0 * INVISCID_SPH_DECAY_L * vsig);
            p.dav_alphadt[i] = (alpha_loc - p.av_alpha[i]) / tau;
        }
    }
}


/* after an accepted time step (afterIntegrationStep): alpha_i = max(alpha_i, alpha_loc) on the new state p_device,
 * alpha_loc from the last rhs of the step */
__global__ void inviscidSPHRaiseAlpha(void)
{
    int i, inc;

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {
        if (p_rhs.materialId[i] < 0)
            continue;
        if (p_rhs.av_alpha_loc[i] > p_rhs.av_alpha[i])
            p_rhs.av_alpha[i] = p_rhs.av_alpha_loc[i];
    }
}


/* start value alpha_i = alpha_max (also when restarting), 0 for boundary and ignored particles */
__global__ void initInviscidSPH(void)
{
    int i, inc, matId;

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];
        p_rhs.av_alpha[i] = (matId < 0 || matEOS[matId] == EOS_TYPE_IGNORE) ? 0.0 : matAlpha[matId];
        p_rhs.dav_alphadt[i] = 0.0;
        p_rhs.av_alpha_loc[i] = p_rhs.av_alpha[i];
    }
}
#endif // INVISCID_SPH
