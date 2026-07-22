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
#include "config_parameter.h"
#include "timeintegration.h"
#include "parameter.h"
#include "pressure.h"


#if FRAGMENTATION
__global__ void damageLimit(void)
{
    register int i, inc;
    volatile int nof, noaf;
    volatile double dmg, dmgMax;

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[i]] || p_rhs.materialId[i] == EOS_TYPE_IGNORE)
            continue;

        dmg = p.d[i];   // note: that's DIM-root of tensile damage
        nof = p.numFlaws[i];
        noaf = p.numActiveFlaws[i];

        if (noaf > nof) {
            printf("ERROR. Found %d activated flaws, but only %d actual flaws...\n", noaf, nof);
            assert(0);
        }

        // limit the tensile damage
        dmgMax = 1.0;
        if (dmg < 0.0)
            dmg = 0.0;
        if (nof > 0)
            // note: damage is limited simply via noaf/nof, but 'dmg' is DIM-root of damage...
            dmgMax = pow( ((double)noaf) / ((double)nof), 1./DIM);
        if (dmg > dmgMax)
            dmg = dmgMax;

        // set (DIM-root of) tensile damage
        p.d[i] = dmg;

#if PALPHA_POROSITY
        if (p.damage_porjutzi[i] > 1.0) {
            p.damage_porjutzi[i] = 1.0;
        } else if (p.damage_porjutzi[i] < 0.0) {
            p.damage_porjutzi[i] = 0.0;
        }

        // note: here 'dmg' is the total damage (above it's DIM-root of tensile damage)
        dmg = pow(p.d[i], DIM) + pow(p.damage_porjutzi[i], DIM);
        if (dmg > 1.0)
            dmg = 1.0;
        p.damage_total[i] = dmg;
#else
        p.damage_total[i] = pow(dmg, DIM);
#endif
    }
}
#endif  // FRAGMENTATION
