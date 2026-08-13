
/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com
 *
 * @section     LICENSE
 * Copyright (c) 2026 Christoph Schaefer
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

#include "fast_integration.h"

#if FAST_INTEGRATION_SCHEME

#include <stdio.h>
#include "miluph.h"
#include "config_parameter.h"
#include "timeintegration.h"
#include "cuda_utils.h"
#include "pressure.h"


/* shared trigger condition, used by both kernels below */
__device__ int fastSchemeTriggers(int m, double time)
{
    if (matFastSwitched[m]) {
        return 0;
    }
    if (matFastTransitionTime[m] < 0.0) {
        return 0;
    }
    if (time < matFastTransitionTime[m]) {
        return 0;
    }
    return 1;
}


#if SOLID
/* S is a state variable holding the deformation history multiplied by the OLD shear modulus.
   Rescaling keeps the elastic strain invariant across the switch. */
__global__ void scaleDeviatoricStress(double *S, int *materialId, double time, int nMaterials)
{
    int i, d, e, m;
    double nu, G_new, scale;

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += blockDim.x * gridDim.x) {
        m = materialId[i];
        if (m < 0 || m >= nMaterials) {
            continue;
        }
        if (!fastSchemeTriggers(m, time)) {
            continue;
        }
        if (matShearmodulus[m] <= 0.0) {
            continue;
        }
        nu = matFastPoissonRatio[m];
        if (nu < 0.0) {
            nu = (3.0*matBulkmodulus[m] - 2.0*matShearmodulus[m]) / 2.0 / (3.0*matBulkmodulus[m] + matShearmodulus[m]);
        }
        G_new = 3.0 * matFastBulkmodulus[m] * (1.0 - 2.0*nu) / (2.0 * (1.0 + nu));
        scale = G_new / matShearmodulus[m];
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                S[stressIndex(i, d, e)] *= scale;
            }
        }
    }
}
#endif


__global__ void applyFastScheme(double time, int nMaterials)
{
    int m = threadIdx.x + blockIdx.x * blockDim.x;
    double K_new;
#if SOLID
    double nu, G_new;
#endif

    if (m >= nMaterials) {
        return;
    }
    if (!fastSchemeTriggers(m, time)) {
        return;
    }

    K_new = matFastBulkmodulus[m];

#if SOLID
    /* Poisson ratio from the config, or preserved from the current elastic moduli.
       Validity was already checked on the host, see config_parameter.cu */
    nu = matFastPoissonRatio[m];
    if (nu < 0.0) {
        nu = (3.0*matBulkmodulus[m] - 2.0*matShearmodulus[m]) / 2.0 / (3.0*matBulkmodulus[m] + matShearmodulus[m]);
    }
    G_new = 3.0 * K_new * (1.0 - 2.0*nu) / (2.0 * (1.0 + nu));
#endif

    switch (matEOS[m]) {
        case (EOS_TYPE_MURNAGHAN):
        case (EOS_TYPE_TILLOTSON):
        case (EOS_TYPE_ANEOS):
            /* the Murnaghan law K/n * (eta^n - 1) with n = 1 is exactly p = K*mu, which is the
               leading Tillotson term with all energy dependence removed */
            matEOS[m] = EOS_TYPE_MURNAGHAN;
            break;
#if PALPHA_POROSITY
        case (EOS_TYPE_JUTZI_MURNAGHAN):
        case (EOS_TYPE_JUTZI):
        case (EOS_TYPE_JUTZI_ANEOS):
            matEOS[m] = EOS_TYPE_JUTZI_MURNAGHAN;
            /* With alpha frozen, p = K/alpha * (alpha*rho/rho_0 - 1) gives dp/drho = K/rho_0,
               independent of alpha. The interpolation between cs_solid and cs_porous in
               soundspeed.cu is then pointless, and the factor 0.5 in cs_porous
               (config_parameter.cu:675) would make the Courant timestep a factor of two too large. */
            matcs_solid[m] = sqrt(K_new / matFastRho0[m]);
            matcs_porous[m] = matcs_solid[m];
            break;
#endif
        default:
            printf("fast integration scheme: material %d has eos type %d, not supported, leaving it alone\n", m, matEOS[m]);
            matFastSwitched[m] = 1;
            return;
    }

    matRho0[m] = matFastRho0[m];
    matTillRho0[m] = matFastRho0[m];
    matN[m] = 1.0;
    matBulkmodulus[m] = K_new;
    matRhoLimit[m] = matFastRhoLimit[m];
    matcsLimit[m] = 0.01 * sqrt(K_new / matFastRho0[m]);
#if SOLID
    matShearmodulus[m] = G_new;
    matYoungModulus[m] = 9.0 * K_new * G_new / (3.0 * K_new + G_new);
#endif

    matFastSwitched[m] = 1;
    printf("fast integration scheme: material %d switched at t = %e to K = %e Pa, cs = %e m/s\n", m, time, K_new, sqrt(K_new/matFastRho0[m]));
#if SOLID
    printf("fast integration scheme: material %d shear modulus is now %e Pa (nu = %e)\n", m, G_new, nu);
#endif
}


void check_fast_scheme(void)
{
#if SOLID
    /* ORDER MATTERS: this needs the OLD shear modulus, so it has to run before
       applyFastScheme() overwrites matShearmodulus. Swapping the two calls is silent:
       matFastSwitched would already be set and every particle would be skipped. */
    cudaVerifyKernel((scaleDeviatoricStress<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>(p_device.S, p_device.materialId, currentTime, numberOfMaterials)));
    cudaVerify(cudaDeviceSynchronize());
#endif
    /* numberOfMaterials is small (order 1 to 10), one block is always enough */
    cudaVerifyKernel((applyFastScheme<<<1, numberOfMaterials>>>(currentTime, numberOfMaterials)));
    cudaVerify(cudaDeviceSynchronize());
}

#endif