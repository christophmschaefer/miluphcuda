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
#include "config_parameter.h"
#include "pressure.h"
#include "aneos.h"


__global__ void calculateSoundSpeed()
{
    register int i, inc, matId;
    int d;
    int j;
    double m_com;
    register double cs, rho, pressure, eta, omega0, z, cs_sq,  cs_c_sq, cs_e_sq, Gamma_e, mu, y; //Gamma_c;
    int i_rho, i_e;

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

            double vkep = sqrt(gravConst * m_com/distance);
            p.cs[i] = vkep * scale_height;
        } else if (EOS_TYPE_IDEAL_GAS == matEOS[matId]) {
            p.cs[i] = sqrt(matPolytropicGamma[matId] * p.p[i] / p.rho[i]);
        } else if (EOS_TYPE_ISOTHERMAL_GAS == matEOS[matId]) {
            p.cs[i] = matIsothermalSoundSpeed[matId];
        } else if (EOS_TYPE_TILLOTSON == matEOS[matId]) {
            rho = p.rho[i];
            eta = rho / matTillRho0[matId];
            omega0 = p.e[i]/(matTillE0[matId]*eta*eta) + 1.0;
            pressure = p.p[i];
            mu = eta - 1.0;
            z = (1.0 - eta)/eta;
            //condensed and expanded cold states
            if (eta >= 0.0 || p.e[i] < matTillEiv[matId]) {
                if (pressure < 0.0 || eta < matRhoLimit[matId]) pressure = 0.0;
                cs_sq = matTilla[matId]*p.e[i]+(matTillb[matId]*p.e[i])/(omega0*omega0)*(3.0*omega0-2.0) +
                    (matTillA[matId]+2.0*matTillB[matId]*mu)/rho + pressure/(rho*rho)*(matTilla[matId]*rho+matTillb[matId]*rho/(omega0*omega0));
            }
            //expanded hot states
            else if (p.e[i] > matTillEcv[matId]) {
                Gamma_e = matTilla[matId] + matTillb[matId]/omega0*exp(-matTillBeta[matId]*z*z);
                cs_sq = (Gamma_e+1.0)*pressure/rho+matTillA[matId]/rho*exp(-(matTillAlpha[matId]*z+matTillBeta[matId]*z*z))*(1.0+mu)/(eta*eta)*(matTillAlpha[matId]+2.0*matTillBeta[matId]*z-eta)
                    + matTillb[matId]*rho*p.e[i]/(omega0*omega0*eta*eta)
                    *exp(-matTillBeta[matId]*z*z)*(2.0*matTillBeta[matId]*z*omega0/matTillRho0[matId] + 1.0)/(matTillE0[matId]*rho)*(2.0*p.e[i]-pressure/rho);
            }
            //intermediate states
            else {
                Gamma_e = matTilla[matId] + matTillb[matId]/omega0*exp(-matTillBeta[matId]*z*z);
                cs_e_sq = (Gamma_e+1.0)*pressure/rho+matTillA[matId]/rho*exp(-(matTillAlpha[matId]*z+matTillBeta[matId]*z*z))*(1.0+mu)/(eta*eta)*(matTillAlpha[matId]+2.0*matTillBeta[matId]*z-eta)
                    + matTillb[matId]*rho*p.e[i]/(omega0*omega0*eta*eta)
                    *exp(-matTillBeta[matId]*z*z)*(2.0*matTillBeta[matId]*z*omega0/matTillRho0[matId] + 1.0)/(matTillE0[matId]*rho)*(2.0*p.e[i]-pressure/rho);
                if (pressure < 0.0 || eta < matRhoLimit[matId]) pressure = 0.0;  //set pressure to zero only for condensed state
                cs_c_sq = matTilla[matId]*p.e[i]+(matTillb[matId]*p.e[i])/(omega0*omega0)*(3.0*omega0-2.0) +
                    (matTillA[matId]+2.0*matTillB[matId]*mu)/rho + pressure/(rho*rho)*(matTilla[matId]*rho+matTillb[matId]*rho/(omega0*omega0));
                y = (p.e[i]-matTillEiv[matId])/(matTillEcv[matId]-matTillEiv[matId]);
                cs_sq = cs_e_sq*(1.0-y)+cs_c_sq*y;
            }
            // set to >= lower limit
            if (cs_sq < matcsLimit[matId]*matcsLimit[matId]){
                p.cs[i] = matcsLimit[matId];
            } else {
                p.cs[i] = sqrt(cs_sq);
            }
        } else if (EOS_TYPE_ANEOS == matEOS[matId]) {
            // find array-indices just below the actual values of rho and e
            i_rho = array_index(p.rho[i], aneos_rho_c+aneos_rho_id_c[matId], aneos_n_rho_c[matId]);
            i_e = array_index(p.e[i], aneos_e_c+aneos_e_id_c[matId], aneos_n_e_c[matId]);
            // interpolate (bi)linearly to obtain the sound speed
            p.cs[i] = bilinear_interpolation_from_linearized(p.rho[i], p.e[i], aneos_cs_c+aneos_matrix_id_c[matId], aneos_rho_c+aneos_rho_id_c[matId], aneos_e_c+aneos_e_id_c[matId], i_rho, i_e, aneos_n_rho_c[matId], aneos_n_e_c[matId], i);
            // set to >= lower limit
            if (p.cs[i] < matcsLimit[matId]) {
                p.cs[i] = matcsLimit[matId];
            }
#if PALPHA_POROSITY
        } else if (EOS_TYPE_JUTZI_MURNAGHAN == matEOS[matId]) {
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
            /* switched from jutzis implementation of the soundspeed to a linear soundspeed from cs_porous with alpha=alpha0 to cs_solid with alpha=1 (also done in iSale) */
            p.cs[i] = matcs_solid[matId] + (matcs_porous[matId] - matcs_solid[matId]) * (p.alpha_jutzi[i] - 1.0) / (matporjutzi_alpha_0[matId] - 1.0);
#if DEBUG_MISC
            if (isnan(p.cs[i])) {
                printf("i %d alpha_jutzi %e delpdelrho %e delpdele %e dalphadp %e p %e rho %e\n", i, p.alpha_jutzi[i], p.delpdelrho[i], p.delpdele[i], p.dalphadp[i], p.p[i], p.rho[i]);
                assert(0);
            }
#endif
        } else if (EOS_TYPE_JUTZI_ANEOS == matEOS[matId]) {
            // find array-indices just below the actual values of rho and e
            i_rho = array_index(p.rho[i], aneos_rho_c+aneos_rho_id_c[matId], aneos_n_rho_c[matId]);
            i_e = array_index(p.e[i], aneos_e_c+aneos_e_id_c[matId], aneos_n_e_c[matId]);
            // interpolate (bi)linearly to obtain the sound speed
            cs = bilinear_interpolation_from_linearized(p.rho[i], p.e[i], aneos_cs_c+aneos_matrix_id_c[matId], aneos_rho_c+aneos_rho_id_c[matId], aneos_e_c+aneos_e_id_c[matId], i_rho, i_e, aneos_n_rho_c[matId], aneos_n_e_c[matId], i);
            // do interpolation only if computed sound speed is above cs_porous, to capture
            // only compaction process, but not expanded states for example...
            if( cs > matcs_porous[matId] ) {
                // linear interpolation between the sound speed in the matrix (from above) and cs_porous (a constant)
                cs = cs + (matcs_porous[matId] - cs) * (p.alpha_jutzi[i] - 1.0) / (matporjutzi_alpha_0[matId] - 1.0);
            }
            // set to >= lower limit
            if (cs < matcsLimit[matId]) {
                p.cs[i] = matcsLimit[matId];
            } else {
                p.cs[i] = cs;
            }
#if DEBUG_MISC
            if (isnan(p.cs[i])) {
                printf("i %d alpha_jutzi %e delpdelrho %e delpdele %e dalphadp %e p %e rho %e\n", i, p.alpha_jutzi[i], p.delpdelrho[i], p.delpdele[i], p.dalphadp[i], p.p[i], p.rho[i]);
                assert(0);
            }
#endif
        } else if (EOS_TYPE_JUTZI == matEOS[matId]) {
            rho = p.rho[i];
            eta = rho / matTillRho0[matId];
            omega0 = p.e[i]/(matTillE0[matId]*eta*eta) + 1.0;
            pressure = p.p[i];
            mu = eta - 1.0;
            z = (1.0 - eta)/eta;
            //condensed and expanded cold states
            if (eta >= 0.0 || p.e[i] < matTillEiv[matId]) {
                if (pressure < 0.0 || eta < matRhoLimit[matId])
                    pressure = 0.0;
                cs_sq = matTilla[matId]*p.e[i]+(matTillb[matId]*p.e[i])/(omega0*omega0)*(3.0*omega0-2.0) +
                    (matTillA[matId]+2.0*matTillB[matId]*mu)/rho + pressure/(rho*rho)*(matTilla[matId]*rho+matTillb[matId]*rho/(omega0*omega0));
            }
            //expanded hot states
            else if (p.e[i] > matTillEcv[matId]) {
                Gamma_e = matTilla[matId] + matTillb[matId]/omega0*exp(-matTillBeta[matId]*z*z);
                cs_sq = (Gamma_e+1.0)*pressure/rho+matTillA[matId]/rho*exp(-(matTillAlpha[matId]*z+matTillBeta[matId]*z*z))*(1.0+mu)/(eta*eta)*(matTillAlpha[matId]+2.0*matTillBeta[matId]*z-eta)
                    + matTillb[matId]*rho*p.e[i]/(omega0*omega0*eta*eta)
                    *exp(-matTillBeta[matId]*z*z)*(2.0*matTillBeta[matId]*z*omega0/matTillRho0[matId] + 1.0)/(matTillE0[matId]*rho)*(2.0*p.e[i]-pressure/rho);
            }
            //intermediate states
            else {
                Gamma_e = matTilla[matId] + matTillb[matId]/omega0*exp(-matTillBeta[matId]*z*z);
                cs_e_sq = (Gamma_e+1.0)*pressure/rho+matTillA[matId]/rho*exp(-(matTillAlpha[matId]*z+matTillBeta[matId]*z*z))*(1.0+mu)/(eta*eta)*(matTillAlpha[matId]+2.0*matTillBeta[matId]*z-eta)
                    + matTillb[matId]*rho*p.e[i]/(omega0*omega0*eta*eta)
                    *exp(-matTillBeta[matId]*z*z)*(2.0*matTillBeta[matId]*z*omega0/matTillRho0[matId] + 1.0)/(matTillE0[matId]*rho)*(2.0*p.e[i]-pressure/rho);
                if (pressure < 0.0 || eta < matRhoLimit[matId]) pressure = 0.0;  //set pressure to zero only for condensed state
                cs_c_sq = matTilla[matId]*p.e[i]+(matTillb[matId]*p.e[i])/(omega0*omega0)*(3.0*omega0-2.0) +
                    (matTillA[matId]+2.0*matTillB[matId]*mu)/rho + pressure/(rho*rho)*(matTilla[matId]*rho+matTillb[matId]*rho/(omega0*omega0));
                y = (p.e[i]-matTillEiv[matId])/(matTillEcv[matId]-matTillEiv[matId]);
                cs_sq = cs_e_sq*(1.0-y)+cs_c_sq*y;
            }
            // do interpolation only if computed sound speed is above cs_porous, to capture
            // only compaction process, but not expanded states for example...
            if( cs_sq > matcs_porous[matId]*matcs_porous[matId] ) {
                cs = sqrt(cs_sq);
                // linear interpolation between the sound speed in the matrix (from above) and cs_porous (a constant)
                cs = cs + (matcs_porous[matId] - cs) * (p.alpha_jutzi[i] - 1.0) / (matporjutzi_alpha_0[matId] - 1.0);
                // set to >= lower limit
                if (cs < matcsLimit[matId]) {
                    p.cs[i] = matcsLimit[matId];
                } else {
                    p.cs[i] = cs;
                }
            } else {
                // set to >= lower limit
                if (cs_sq < matcsLimit[matId]*matcsLimit[matId]){
                    p.cs[i] = matcsLimit[matId];
                } else {
                    p.cs[i] = sqrt(cs_sq);
                }
            }
#if DEBUG_MISC
            if (isnan(p.cs[i])) {
                printf("i %d alpha_jutzi %e delpdelrho %e delpdele %e dalphadp %e p %e rho %e\n", i, p.alpha_jutzi[i], p.delpdelrho[i], p.delpdele[i], p.dalphadp[i], p.p[i], p.rho[i]);
                assert(0);
            }
#endif
#endif // PALPHA_POROSITY
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
        // other material types have a constant soundspeed which is set in initializeSoundspeed()
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
            p.cs[i] = 203.0;
        } else if (EOS_TYPE_TILLOTSON == matEOS[matId]) {
            p.cs[i] = sqrt(matBulkmodulus[matId]/matTillRho0[matId]);
        } else if (EOS_TYPE_ANEOS == matEOS[matId]) {
            p.cs[i] = aneos_bulk_cs_c[matId];
        } else if (EOS_TYPE_MURNAGHAN == matEOS[matId]) {
            p.cs[i] = sqrt(matBulkmodulus[matId]/matRho0[matId]);
        } else if (EOS_TYPE_JUTZI == matEOS[matId]) {
            p.cs[i] = matcs_porous[matId];
        } else if (EOS_TYPE_JUTZI_ANEOS == matEOS[matId]) {
            p.cs[i] = matcs_porous[matId];
        } else if (EOS_TYPE_JUTZI_MURNAGHAN == matEOS[matId]) {
            p.cs[i] = matcs_porous[matId];
        } else if (EOS_TYPE_REGOLITH == matEOS[matId]) {
            //sound speed in soil is typically between 450 and 600 m/s according to Ha H. Bui 2008
            p.cs[i] = 500.0;
//        } else if (EOS_TYPE_EPSILON == matEOS[matId]) {
//            p.cs[i] = sqrt(matBulkmodulus[matId]/matTillRho0[matId]);
        }
    }
}
