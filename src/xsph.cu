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
#include "xsph.h"
#include "miluph.h"
#include "config_parameter.h"
#include "timeintegration.h"
#include "parameter.h"
#include "pressure.h"
#include "kernel.h"


extern __device__ SPH_kernel kernel;


__global__ void calculateXSPHchanges(int *interactions)
{
    register int64_t interactions_index;
    register int i, k, inc, j, numInteractions, e;

    double W;
    double Wj;
    double dWdx[DIM];
    double dWdr;
    double dx[DIM];
    double hbar;

    double vx;
#if DIM > 1
    double vy;
#endif
#if DIM > 2
    double vz;
#endif

    double dvx;
#if DIM > 1
    double dvy;
#endif
#if DIM > 2
    double dvz;
#endif

    inc = blockDim.x * gridDim.x;
    // particle loop to smooth velocity field
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[i]] || EOS_TYPE_IGNORE == p_rhs.materialId[i]) {
                continue;
        }
        numInteractions = p.noi[i];
        hbar = p.h[i];
        vx = p.vx[i];
#if DIM > 1
        vy = p.vy[i];
#if DIM == 3
        vz = p.vz[i];
#endif
#endif
        p.xsphvx[i] = 0.0;
#if DIM > 1
        p.xsphvy[i] = 0.0;
#if DIM == 3
        p.xsphvz[i] = 0.0;
#endif
#endif
        // neighbours loop
        for (k = 0; k < numInteractions; k++) {
            interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + k;
            j = interactions[interactions_index];

            // if j is brush, continue
            if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[j]] || EOS_TYPE_IGNORE == p_rhs.materialId[j]) {
                continue;
            }

#if VARIABLE_SML
            hbar = 0.5*(p.h[i] + p.h[j]);
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
            kernel(&Wj, dWdx, &dWdr, dx, p.h[j]);
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            Wj /= p_rhs.shepard_correction[j];
# endif
            W = 0.5 * (W + Wj);
#else
            kernel(&W, dWdx, &dWdr, dx, hbar);
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
# endif
#endif

            dvx = vx - p.vx[j];
#if DIM > 1
            dvy = vy - p.vy[j];
#if DIM == 3
            dvz = vz - p.vz[j];
#endif
#endif
            p.xsphvx[i] -= p.m[j] / (0.5 * (p.rho[i] + p.rho[j])) * W * dvx;
#if DIM > 1
            p.xsphvy[i] -= p.m[j] / (0.5 * (p.rho[i] + p.rho[j])) * W * dvy;
#if DIM == 3
            p.xsphvz[i] -= p.m[j] / (0.5 * (p.rho[i] + p.rho[j])) * W * dvz;
#endif
#endif
        } /* neighbours loop end */
    }  /* first particle loop end */

}
