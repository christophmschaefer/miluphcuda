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
#include "timeintegration.h"
#include "parameter.h"
#include "pressure.h"


extern __device__ SPH_kernel kernel;
extern __device__ SPH_kernel wendlandc2_p;


__global__ void calculateDensity(int *interactions) {
    int i;
    int j;
    int inc;
    int ip;
    int d;
    double W;
    double dx[DIM];
    double dWdx[DIM];
    double dWdr;
    double rho;
    double sml;

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        if (EOS_TYPE_IGNORE == matEOS[p.materialId[i]] || p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
                continue;
        }
        sml = p.h[i];
#if 0
#if (DIM == 2)
        rho = 40.0/(7.0*M_PI*sml*sml) * p.m[i];
#elif (DIM == 3)
        rho = 8.0/(M_PI*sml*sml*sml) * p.m[i];
#elif (DIM == 1)
        rho = 4.0/(3.0*sml) * p.m[i];
#endif
#endif

        // self density is m_i W_ii
        for (d = 0; d < DIM; d++) {
            dx[d] = 0;
        }
        kernel(&W, dWdx, &dWdr, dx, sml);
        rho = p.m[i] * W;

// correction factors for Wendland CX kernels
        if (kernel == wendlandc2_p) {
            // these values are for 3D from Dehnen and Aly 2012. aiaiaiaiaiai
            rho -= rho*0.0294*powf((double) p.noi[i], -0.977);
        }


        // sph sum for particle i
        for (j = 0; j < p.noi[i]; j++) {
            ip = interactions[i * MAX_NUM_INTERACTIONS + j];
            if (EOS_TYPE_IGNORE == matEOS[p.materialId[ip]] || p_rhs.materialId[ip] == EOS_TYPE_IGNORE) {
                continue;
            }
#if (VARIABLE_SML || INTEGRATE_SML || DEAL_WITH_TOO_MANY_INTERACTIONS)
            sml = 0.5*(p.h[i] + p.h[ip]);
#endif

            dx[0] = p.x[i] - p.x[ip];
#if DIM > 1
            dx[1] = p.y[i] - p.y[ip];
#if DIM > 2
            dx[2] = p.z[i] - p.z[ip];
#endif
#endif
            kernel(&W, dWdx, &dWdr, dx, sml);
            // contribution of interaction
            rho += p.m[ip] * W;
        }
        // write to global memory
        p.rho[i] = rho;
    }
}
