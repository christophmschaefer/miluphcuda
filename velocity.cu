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
#include "velocity.h"
#include "miluph.h"
#include "config_parameter.h"
#include "timeintegration.h"
#include "parameter.h"
#include "pressure.h"


__global__ void setlocationchanges(int *interactions)
{

    register int i, inc;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[i]] || p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
                continue;
        }
        p.dxdt[i] = p.vx[i];
#if DIM > 1
        p.dydt[i] = p.vy[i];
#endif
#if XSPH
        p.dxdt[i] += 0.5 * p.xsphvx[i];
#if DIM > 1
        p.dydt[i] += 0.5 * p.xsphvy[i];
#endif
#endif
#if DIM == 3
        p.dzdt[i] = p.vz[i];
#if XSPH
        p.dzdt[i] += 0.5 * p.xsphvz[i];
#endif
#endif
    }
}
