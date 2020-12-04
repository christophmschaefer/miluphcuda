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
