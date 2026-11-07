/**
 * @author      Oliver Wandel and Christoph Schaefer cm.schaefer@gmail.com
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

#include "miluph.h"
#include "timeintegration.h"
#include "porosity.h"
#include "pressure.h"
#include "parameter.h"
#include "math.h"
#include "float.h"

// we need these material globals in scope
#if SIRONO_POROSITY || (PALPHA_POROSITY && PALPHA_RETURN_MAPPING)
#ifndef _CONFIG_PARAMETER_H
#include "config_parameter.h"
#endif
#endif

#if PALPHA_POROSITY

__global__ void calculateDistensionChange()
{
/*
    register int i, inc, matId;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];
        if (matEOS[matId] == EOS_TYPE_JUTZI || matEOS[matId] == EOS_TYPE_JUTZI_MURNAGHAN || matEOS[matId] == EOS_TYPE_JUTZI_ANEOS) {
            if (p.alpha_jutzi[i] <= 1.0) {
                p.dalphadt[i] = 0.0;
                p.alpha_jutzi[i] = 1.0;
            } else {
                p.dalphadt[i] = ((p.dedt[i] * p.delpdele[i] + p.alpha_jutzi[i] * p.drhodt[i] * p.delpdelrho[i])
                            * p.dalphadp[i]) / (p.alpha_jutzi[i] + p.dalphadp[i] * (p.p[i] - p.rho[i] * p.delpdelrho[i]));
                if (p.dalphadt[i] > 0.0) {
                    p.dalphadt[i] = 0.0;
                }
	        }
        } else {
            p.dalphadt[i] = 0.0;
        }
    }
*/
}
#endif

#if PALPHA_POROSITY && PALPHA_RETURN_MAPPING
/*
 * Return mapping for the distention of the p-alpha model, see PALPHA_RETURN_MAPPING in parameter.h.
 *
 * The integrators keep alpha constant during a step (dalphadt = 0, see internal_forces.cu). For each state that is
 * evaluated, i.e. at the start of each rhs and before the pressure checks of the integrators, alpha is instead
 * determined from rho and e with the history alpha_hist, the distention after the last accepted step: alpha is the
 * root of g(alpha) = alpha - alpha_cc(P(alpha)) with P(alpha) = P_s(alpha*rho, e)/alpha in [1, alpha_hist] if
 * g(alpha_hist) > 0 (compressed beyond the crush curve), and alpha = alpha_hist otherwise (elastic). The first rhs
 * after an accepted step evaluates the accepted state and commits its alpha as new alpha_hist (lazy commit, so that
 * a summation density is already up to date).
 *
 * Since alpha * dg/dalpha is the denominator of dalphadt in internal_forces.cu, g is increasing and the root unique
 * wherever the time integration of alpha is well defined. Where it is not (rho_s dP_s/drho_s < P_s, e.g. in a
 * two-phase region of ANEOS), g can have several roots in the bracket, and the iteration returns one of them.
 * Only real particles are mapped, ghost particles carry no p-alpha state.
 *
 * The changes of alpha are applied to p.S = S/alpha (the matrix deviator S itself does not change with alpha, cf.
 * the alpha-dot term of dSdt in internal_forces.cu) and to the DIM-root of the porous damage with its closed form,
 * the exact integrals of the terms that the time integration of alpha would add.
 */

/* convergence: |g| <= PALPHA_RM_TOL_G or a bracket of the root narrower than PALPHA_RM_TOL_ALPHA. PALPHA_RM_TOL_G is
   far below PALPHA_ELASTIC_RELOADING_TOL, so that calculatePressure() treats a mapped particle as on the crush curve */
#define PALPHA_RM_TOL_G 1e-12
#define PALPHA_RM_TOL_ALPHA 1e-14
#define PALPHA_RM_MAX_ITER 100

int palpha_commit_pending = TRUE;

/* g(alpha) = alpha - alpha_cc(P(alpha)) and dg/dalpha = 1 + dalpha_cc/dP * (P - rho * dP_s/drho_s)/alpha */
__device__ static double palpha_rm_g(double alpha, double rho, double e, int matId, int i, double *dgdalpha)
{
    double pressure_solid, dpdrho_s, dpde, pressure;

    palpha_matrix_eos(rho, alpha, e, matId, &pressure_solid, &dpdrho_s, &dpde, NULL, i);
    pressure = pressure_solid / alpha;
    *dgdalpha = 1.0 + crushcurve_dalphadp(pressure, matId, i) * (pressure - rho * dpdrho_s) / alpha;
    return alpha - crushcurve_alpha(pressure, matId);
}

#if SOLID && FRAGMENTATION
/* DIM-root of the porous damage as function of alpha, Jutzi et al. (2008), eq. (57) for DIM = 3. Its time
   derivative is ddamage_porjutzidt in internal_forces.cu, keep both in sync. */
__device__ static double palpha_damage_root(double alpha, double alpha_0)
{
    const double deld = 0.01;

    return (pow(1.0 - (alpha - 1.0) / (alpha_0 - 1.0) + deld, 1.0/DIM) - pow(deld, 1.0/DIM))
        / (pow(1.0 + deld, 1.0/DIM) - pow(deld, 1.0/DIM));
}
#endif

__global__ void palphaReturnMapping(int commit, int *numCompacted)
{
    int i, matId, iter;
#if SOLID && STRESS_PALPHA_POROSITY
    int k;
#endif
    double rho, e, alpha_in, alpha_hist, alpha, alpha_next, lo, hi, g, dg, dg_lo, dx, dx_old;

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += blockDim.x * gridDim.x) {
        matId = p_rhs.materialId[i];
        if (matId < 0)
            continue;
        if (matEOS[matId] != EOS_TYPE_JUTZI && matEOS[matId] != EOS_TYPE_JUTZI_MURNAGHAN
                && matEOS[matId] != EOS_TYPE_JUTZI_ANEOS)
            continue;
        rho = p.rho[i];
        e = p.e[i];
        alpha_in = p.alpha_jutzi[i];
        alpha_hist = p_rhs.alpha_jutzi_hist[i];

        /* elastic, below the crush curve */
        alpha = alpha_hist;
        if (alpha_hist <= 1.0) {
            alpha = 1.0;
        } else if (rho > 0.0) {
            g = palpha_rm_g(alpha_hist, rho, e, matId, i, &dg);
            if (g > PALPHA_RM_TOL_G) {
                /* compressed beyond the crush curve, root of g in [1, alpha_hist]. g <= PALPHA_RM_TOL_G counts as
                   on the crush curve, so that the mapping of an already mapped state does not change it. */
                if (numCompacted != NULL && g > PALPHA_ELASTIC_RELOADING_TOL)
                    atomicAdd(numCompacted, 1);
                lo = 1.0;
                hi = alpha_hist;
                if (palpha_rm_g(lo, rho, e, matId, i, &dg_lo) >= 0.0) {
                    /* fully compacted */
                    alpha = 1.0;
                } else {
                    /* Newton iteration from alpha_hist, with a bisection step whenever the Newton step leaves the
                       bracket or does not halve the step before last (rtsafe in Press et al., Numerical Recipes) */
                    dx_old = hi - lo;
                    dx = dx_old;
                    for (iter = 0; iter < PALPHA_RM_MAX_ITER; iter++) {
                        alpha_next = alpha - g / dg;
                        if (!(dg > 0.0) || !(alpha_next > lo && alpha_next < hi)
                                || fabs(2.0 * g) > fabs(dx_old * dg)) {
                            alpha_next = 0.5 * (lo + hi);
                        }
                        dx_old = dx;
                        dx = fabs(alpha_next - alpha);
                        alpha = alpha_next;
                        g = palpha_rm_g(alpha, rho, e, matId, i, &dg);
                        if (fabs(g) <= PALPHA_RM_TOL_G)
                            break;
                        if (g > 0.0)
                            hi = alpha;
                        else
                            lo = alpha;
                        if (hi - lo <= PALPHA_RM_TOL_ALPHA)
                            break;
                    }
                    /* without convergence of g, which happens at a discontinuity of P(alpha) (e.g. the Tillotson
                       cutoff at rho_limit), use the end of the bracket with g < 0, on or below the crush curve */
                    if (fabs(g) > PALPHA_RM_TOL_G)
                        alpha = lo;
                    if (iter == PALPHA_RM_MAX_ITER) {
                        printf("WARNING: return mapping for particle %d not converged: g = %e, "
                                "bracket [%.17e, %.17e]\n", i, g, lo, hi);
                    }
                }
            }
        }

        if (alpha != alpha_in) {
#if SOLID && STRESS_PALPHA_POROSITY
            /* p.S = S/alpha with the matrix deviator S */
            for (k = 0; k < DIM*DIM; k++) {
                p.S[i*DIM*DIM + k] *= alpha_in / alpha;
            }
#endif
#if SOLID && FRAGMENTATION
            if (matporjutzi_alpha_0[matId] > 1.0) {
                p.damage_porjutzi[i] += palpha_damage_root(alpha, matporjutzi_alpha_0[matId])
                    - palpha_damage_root(alpha_in, matporjutzi_alpha_0[matId]);
            }
#endif
            p.alpha_jutzi[i] = alpha;
        }
        if (commit) {
            p_rhs.alpha_jutzi_hist[i] = alpha;
        }
    }
}

void palpha_return_mapping(int commit)
{
    static int first_commit = TRUE;
    int *numCompacted_d = NULL;
    int numCompacted = 0;

    if (commit && first_commit) {
        cudaVerify(cudaMalloc((void**)&numCompacted_d, sizeof(int)));
        cudaVerify(cudaMemset(numCompacted_d, 0, sizeof(int)));
    }
    cudaVerifyKernel((palphaReturnMapping<<<numberOfMultiprocessors * 4, NUM_THREADS_PALPHA_POROSITY>>>(
                    commit, numCompacted_d)));
    cudaVerify(cudaDeviceSynchronize());
    if (numCompacted_d != NULL) {
        cudaVerify(cudaMemcpy(&numCompacted, numCompacted_d, sizeof(int), cudaMemcpyDeviceToHost));
        cudaVerify(cudaFree(numCompacted_d));
        if (numCompacted > 0) {
            fprintf(stderr, "WARNING: %d particles start above the crush curve, alpha > alpha_cc(P). "
                    "PALPHA_RETURN_MAPPING compacts them onto the crush curve in the first step.\n", numCompacted);
        }
        first_commit = FALSE;
    }
}
#endif

#if SIRONO_POROSITY
#define MAXFLOAT DBL_MAX

__global__ void calculateCompressiveStrength()
{
    register int i, inc, matId;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];
        if (matEOS[matId] == EOS_TYPE_SIRONO) {
            double alpha = matporsirono_alpha[matId];
            double pm = matporsirono_pm[matId];
            double phimax = matporsirono_phimax[matId];
            double phi0 = matporsirono_phi0[matId];
            double delta = matporsirono_delta[matId];
            double rho_s = matporsirono_rho_s[matId];
            double phi = p.rho[i] / rho_s;
            /* Using omni-sided_compression curve for compressive strength */
            if (phi <= 0.125)
                p.compressive_strength[i] = alpha * 31.45166;
            if ((phi > 0.125) && (phi < 0.58))
                p.compressive_strength[i] = alpha * pm * pow(((phimax - phi0) / (phimax - phi) - 1.0), delta * 2.302585);
            if (phi >= 0.58)
                p.compressive_strength[i] = MAXFLOAT;
        } else {
            p.compressive_strength[i] = MAXFLOAT;
        }
    }
}

__global__ void calculateTensileStrength()
{
    register int i, inc, matId;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];
        if (matEOS[matId] == EOS_TYPE_SIRONO) {
            double rho_s = matporsirono_rho_s[matId];
            double phi = p.rho[i] / rho_s;
            double tensStrength;
            tensStrength = pow(10.0, (2.8 + 1.48 * phi));
            p.tensile_strength[i] = tensStrength * (-1.0);
        } else {
            p.tensile_strength[i] = -MAXFLOAT;
        }
    }
}

#endif
