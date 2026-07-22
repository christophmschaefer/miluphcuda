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


#include "parameter.h"
#include "pressure.h"
#include "miluph.h"
#include "config_parameter.h"
#include "little_helpers.h"


#if USE_SIGNAL_HANDLER
extern volatile int terminate_flag;


// handles the SIGTERM
void signal_handler(int signum)
{
   printf("Caught signal %d, trying to write particle data and exit...\n", signum);
   terminate_flag = 1;
}

#endif

__global__ void checkNaNs(int *interactions)
{
    int i, k, inc, j, numInteractions;
    int d;



    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {
        if (p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
                continue;
        }
        assert(!isnan(p.e[i]));
        assert(!isnan(p.p[i]));
        assert(!isnan(p.rho[i]));
#if FRAGMENTATION
        assert(!isnan(p.d[i]));
#endif
#if PALPHA_POROSITY
        assert(!isnan(p.alpha_jutzi[i]));
#endif
        assert(!isnan(p.x[i]));
#if DIM > 1
        assert(!isnan(p.y[i]));
# if DIM > 2
        assert(!isnan(p.z[i]));
# endif
#endif
        for (d = 0; d < DIM*DIM; d++) {
#if TENSORIAL_CORRECTION
            assert(!isnan(p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+d]));
#endif
#if SOLID
            if (isnan(p.S[i*DIM*DIM+d])) {
                printf("stress component of particle %d is nan, dying here \n", i);
                for (int e = 0; e < DIM; e++) {
                        for (int f = 0; f < DIM; f++) {
                                printf("%e \t", p.S[i*DIM*DIM+e*DIM+f]);
                        }
                        printf("\n");
                }
            }
            assert(!isnan(p.S[i*DIM*DIM+d]));
            assert(!isnan(p_rhs.sigma[i*DIM*DIM+d]));
#endif
        }
#if PALPHA_POROSITY
        assert(!isnan(p.dalphadt[i]));
#endif
#if INTEGRATE_ENERGY
        assert(!isnan(p.dedt[i]));
#endif
#if INTEGRATE_DENSITY
        assert(!isnan(p.drhodt[i]));
#endif
#if FRAGMENTATION
        assert(!isnan(p.dddt[i]));
#endif
#if SOLID
        for (d = 0; d < DIM*DIM; d++) {
            assert(!isnan(p.dSdt[i*DIM*DIM+d]));
        }
#endif
        assert(!isnan(p.ax[i]));
#if DIM > 1
        assert(!isnan(p.ay[i]));
# if DIM > 2
        assert(!isnan(p.az[i]));
# endif
#endif
        if (p.noi[i] == 0) {
            printf("particle %d with no interactions...\n", i);
        }
    }
}

#if TENSORIAL_CORRECTION
__global__ void printTensorialCorrectionMatrix(int *interactions)
{
    int i, k, inc, j, numInteractions;
    int d, dd;

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {
        printf("%d\n", i);
        for (d = 0; d < DIM; d++) {
            for (dd = 0; dd < DIM; dd++) {
                printf("%.17lf \t", p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+d*DIM+dd]);
            }
            printf("\n");
        }
    }
}
#endif
