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
#include "damage.h"
#include "miluph.h"
#include "timeintegration.h"
#include "parameter.h"
#include "pressure.h"

#if FRAGMENTATION
__global__ void damageLimit(void) {
    register int i, inc;
    volatile int nof, noaf;
    volatile double dmg, dmgMax;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        if (EOS_TYPE_IGNORE == matEOS[p.materialId[i]] || p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
                continue;
        }
        dmg = p.d[i];
        if (dmg < 0) dmg = 0;
        noaf = p.numActiveFlaws[i];
        dmgMax = 1;
        if (noaf < 1 && dmg > 0) {
            printf("Error, not possible: noaf: %d, dmg %e\n", noaf, dmg);
            assert(0);
        }
        if (noaf > 0) {
            nof = p.numFlaws[i];
            dmgMax = pow( ((double)noaf) / ((double)nof) , 1./DIM);
        }

        if (dmg > dmgMax) {
            dmg = dmgMax;
        } else if (dmg < 0) {
            printf("ERROR: DAMAGE is negative \t \t");
            printf("%e %e %e %d %d\n", p.x[i], p.y[i], p.d[i], p.numFlaws[i], p.numActiveFlaws[i]);
            assert(0);
        }
        p.d[i] = dmg;
#if PALPHA_POROSITY
        if (p.damage_porjutzi[i] > 1.0) {
            p.damage_porjutzi[i] = 1.0;
        } else if (p.damage_porjutzi[i] < 0) {
            p.damage_porjutzi[i] = 0.0;
        }
        dmg = p.d[i] + p.damage_porjutzi[i];
        if (dmg > 1) dmg = 1.0;
#endif
        p.damage_total[i] = dmg;
    }
}
#endif
