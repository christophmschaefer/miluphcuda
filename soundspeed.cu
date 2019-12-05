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

#include "miluph.h"
#include "soundspeed.h"
#include "pressure.h"


__global__ void calculateSoundSpeed()
{
    register int i, inc, matId;
    int d;
    int j;
    double m_com;

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];
        if (EOS_TYPE_POLYTROPIC_GAS == matEOS[matId]) {
            p.cs[i] = sqrt(matPolytropicK[matId] * pow(p.rho[i], matPolytropicGamma[matId]-1.0));
        } else if (EOS_TYPE_LOCALLY_ISOTHERMAL_GAS == matEOS[matId]) {
            // p = vkep \times scale_height
            double distance = 0.0;
            distance = p.x[i] * p.x[i];
#if DIM > 1
            distance += p.y[i]*p.y[i];
#if DIM > 2
            distance += p.z[i]*p.z[i];
#endif
#endif
            distance = sqrt(distance);
            m_com = 0;
            for (j = 0; j < numPointmasses; j++) {
                m_com += pointmass.m[j];
            }

            double vkep = sqrt(C_GRAVITY_SI * m_com/distance);
            p.cs[i] = vkep * scale_height;
        } else if (EOS_TYPE_IDEAL_GAS == matEOS[matId]) {
            p.cs[i] = sqrt(matPolytropicGamma[matId] * p.p[i] / p.rho[i]);
        } else if (EOS_TYPE_JUTZI == matEOS[matId] || EOS_TYPE_JUTZI_MURNAGHAN == matEOS[matId]) {
#if PALPHA_POROSITY
            //p.cs[i] = sqrt(matBulkmodulus[matId]/matTillRho0[matId]);
//            if (p.alpha_jutzi[i] > 1.0 && abs(p.dalphadp[i]) > 0) {
//                if (abs(p.delpdelrho[i]) > 0.0 || abs(p.delpdele[i]) > 0.0) {
//                    p.cs[i] = sqrt((p.alpha_jutzi[i] * p.delpdelrho[i] + p.delpdele[i] * p.p[i] / (p.rho[i] * p.rho[i]))
//                                / (p.alpha_jutzi[i] + p.dalphadp[i] * (p.p[i] - p.rho[i] * p.delpdelrho[i])));
//                }
//            }
//            if (!isnan(p.cs[i])) {
//                p_rhs.cs_old[i] = p.cs[i];
//            } else {
//                p.cs[i] = p_rhs.cs_old[i];
//            }
//#if 0
			/* switched from jutzis implementation of the soundspeed to a linear soundspeed from cs_porous with alpha=alpha0 to cs_solid with alpha=1 (also done in iSale) */
			p.cs[i] = matcs_solid[matId] + (matcs_porous[matId] - matcs_solid[matId]) * (p.alpha_jutzi[i] - 1.0) / (matporjutzi_alpha_0[matId] - 1.0);
#if DEBUG
            if (isnan(p.cs[i])) {
                printf("i %d alpha_jutzi %e delpdelrho %e delpdele %e dalphadp %e p %e rho %e\n", i, p.alpha_jutzi[i], p.delpdelrho[i], p.delpdele[i], p.dalphadp[i], p.p[i], p.rho[i]);
                        assert(0);
              }
//#endif
#endif

#endif
#if SIRONO_POROSITY
        } else if (EOS_TYPE_SIRONO == matEOS[matId]) {
            if (p.flag_plastic[i] > 0)
                p.cs[i] = sqrt(p.compressive_strength[i] / p.rho[i]);
            else
                p.cs[i] = sqrt(p.K[i] / p.rho_0prime[i]);
#endif
#if EPSALPHA_POROSITY
        /* Improvements to epsilon-alpha model by Collins et al 2010 */
        } else if (EOS_TYPE_EPSILON == matEOS[matId]) {
            double c_s0 = sqrt(matBulkmodulus[matId]/matTillRho0[matId]);
            double c_p0 = sqrt(matBulkmodulus[matId]/(matTillRho0[matId] / matporepsilon_alpha_0[matId]));
            p.cs[i] = c_s0 + (p.alpha_epspor[i] - 1.0) / (matporepsilon_alpha_0[matId] - 1.0) * (c_p0 - c_s0);
#endif
        }
        // other material types have a constant soundspeed which is calculated in initializeSoundspeed
    }
}



__global__ void initializeSoundspeed()
{
    register int i, inc, matId;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];
        if (EOS_TYPE_POLYTROPIC_GAS == matEOS[matId]) {
            p.cs[i] = 0.0; // for gas this will be calculated each step by kernel calculateSoundSpeed
        } else if (EOS_TYPE_ISOTHERMAL_GAS == matEOS[matId]) {
            /* this is pure molecular hydrogen at 10 K */
            p.cs[i] = 203;
        } else if (EOS_TYPE_TILLOTSON == matEOS[matId]) {
            p.cs[i] = sqrt(matBulkmodulus[matId]/matTillRho0[matId]);
        } else if (EOS_TYPE_ANEOS == matEOS[matId]) {
            p.cs[i] = aneos_bulk_cs_c[matId];
        } else if (EOS_TYPE_MURNAGHAN == matEOS[matId]) {
            p.cs[i] = sqrt(matBulkmodulus[matId]/matRho0[matId]);
        } else if (EOS_TYPE_REGOLITH == matEOS[matId]) {
            //sound speed in soil is typically between 450 and 600 m/s according to Ha H. Bui 2008
            p.cs[i] = 500.0;
//        } else if (EOS_TYPE_EPSILON == matEOS[matId]) {
//            p.cs[i] = sqrt(matBulkmodulus[matId]/matTillRho0[matId]);
        }
    }
}
