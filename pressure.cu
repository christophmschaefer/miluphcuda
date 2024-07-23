/**
 * @author      Oliver Wandel, Christoph Burger, Christoph Schaefer and Thomas I. Maindl
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


#include "pressure.h"
#include "parameter.h"
#include "config_parameter.h"
#include "miluph.h"
#include "aneos.h"

__global__ void calculatePressure() {
    register int i, inc, matId;
    register double eta, e, rho, rho0, mu, p1, p2;
    int i_rho, i_e;
    double pressure;

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        pressure = 0.0;
        matId = p_rhs.materialId[i];
        if (EOS_TYPE_IGNORE == matEOS[matId] || matId == EOS_TYPE_IGNORE) {
            continue;
        }
        if (EOS_TYPE_POLYTROPIC_GAS == matEOS[matId]) {
            p.p[i] = matPolytropicK[matId] * pow(p.rho[i], matPolytropicGamma[matId]);
        } else if (EOS_TYPE_IDEAL_GAS == matEOS[matId]) {
            p.p[i] = (matPolytropicGamma[matId] - 1) * p.rho[i] * p.e[i];
        } else if (EOS_TYPE_LOCALLY_ISOTHERMAL_GAS == matEOS[matId]) {
            p.p[i] = p.cs[i]*p.cs[i] * p.rho[i];
        } else if (EOS_TYPE_ISOTHERMAL_GAS == matEOS[matId]) {
            /* this is pure molecular hydrogen at 10 K */
//            p.p[i] = 41255.407 * p.rho[i];
            p.p[i] = p.cs[i]*p.cs[i] * p.rho[i];
        } else if (EOS_TYPE_MURNAGHAN == matEOS[matId] || EOS_TYPE_VISCOUS_REGOLITH == matEOS[matId]) {
            eta = p.rho[i] / matRho0[matId];
            if (eta < matRhoLimit[matId]) {
                p.p[i] = 0.0;
            } else {
                p.p[i] = (matBulkmodulus[matId]/matN[matId])*(pow(eta, matN[matId]) - 1.0);
            }
        } else if (EOS_TYPE_TILLOTSON == matEOS[matId]) {
            rho = p.rho[i];
            e = p.e[i];
            eta = rho / matTillRho0[matId];
            mu = eta - 1.0;
            if (eta < matRhoLimit[matId] && e < matTillEcv[matId]) {
                p.p[i] = 0.0;
            } else {
                if (e <= matTillEiv[matId] || eta >= 1.0) {
                    p.p[i] = (matTilla[matId] + matTillb[matId]/(e/(eta*eta*matTillE0[matId])+1.0))
                        * rho * e + matTillA[matId]*mu + matTillB[matId]*mu*mu;
                } else if (e >= matTillEcv[matId] && eta >= 0.0) {
                    p.p[i] = matTilla[matId]*rho*e + (matTillb[matId]*rho*e/(e/(eta*eta*matTillE0[matId])+1.0)
                        + matTillA[matId] * mu * exp(-matTillBeta[matId]*(matTillRho0[matId]/rho - 1.0)))
                        * exp(-matTillAlpha[matId] * (pow(matTillRho0[matId]/rho-1.0, 2)));
                } else if (e > matTillEiv[matId] && e < matTillEcv[matId]) {
                    // for intermediate states:
                    // weighted average of pressures calculated by expanded
                    // and compressed versions of Tillotson (both evaluated at e)
                    p1 = (matTilla[matId]+matTillb[matId]/(e/(eta*eta*matTillE0[matId])+1.0)) * rho*e
                        + matTillA[matId]*mu + matTillB[matId]*mu*mu;
                    p2 = matTilla[matId]*rho*e + (matTillb[matId]*rho*e/(e/(eta*eta*matTillE0[matId])+1.0)
                        + matTillA[matId] * mu * exp(-matTillBeta[matId]*(matTillRho0[matId]/rho -1.0)))
                        * exp(-matTillAlpha[matId] * (pow(matTillRho0[matId]/rho-1.0, 2)));
                    p.p[i] = ( p1*(matTillEcv[matId]-e) + p2*(e-matTillEiv[matId]) ) / (matTillEcv[matId]-matTillEiv[matId]);
                } else {
                    printf("\n\nDeep trouble in pressure.\nenergy[%d] = %e\nE_iv = %e, E_cv = %e\n\n", i, e, matTillEiv[matId], matTillEcv[matId]);
                    p.p[i] = 0.0;
                }
            }
        } else if (EOS_TYPE_ANEOS == matEOS[matId]) {
            /* find array-indices just below the actual values of rho and e */
            i_rho = array_index(p.rho[i], aneos_rho_c+aneos_rho_id_c[matId], aneos_n_rho_c[matId]);
            i_e = array_index(p.e[i], aneos_e_c+aneos_e_id_c[matId], aneos_n_e_c[matId]);
            /* interpolate (bi)linearly to obtain the pressure */
            p.p[i] = bilinear_interpolation_from_linearized(p.rho[i], p.e[i], aneos_p_c+aneos_matrix_id_c[matId], aneos_rho_c+aneos_rho_id_c[matId], aneos_e_c+aneos_e_id_c[matId], i_rho, i_e, aneos_n_rho_c[matId], aneos_n_e_c[matId], i);
#if SIRONO_POROSITY
        } else if (matEOS[matId] == EOS_TYPE_SIRONO) {
            double K_0 = matporsirono_K_0[matId];
            double rho_0 = matporsirono_rho_0[matId];
            double gamma_K = matporsirono_gamma_K[matId];
            pressure = p.K[i] * (p.rho[i] / p.rho_0prime[i] - 1.0);
            p.flag_rho_0prime[i] = -1;
            if (pressure >= 0.0) {
                if (p.rho[i] >= p.rho_c_plus[i]) {
                    if (pressure >= p.compressive_strength[i]) {
                        pressure = p.compressive_strength[i];
                        p.flag_plastic[i] = 1;
                        p.rho_c_plus[i] = p.rho[i];
                        p.flag_rho_0prime[i] = 1;
                    } else {
                        p.flag_plastic[i] = -1;
                        p.flag_rho_0prime[i] = -1;
                    }
                } else {
                    if (pressure >= p.compressive_strength[i]) {
                        pressure = p.compressive_strength[i];
                    }
                    if (p.flag_plastic[i] == 1) {
                        p.flag_rho_0prime[i] = 1;
                        p.flag_plastic[i] = -1;
                    } else {
                        p.flag_rho_0prime[i] = -1;
                        p.flag_plastic[i] = -1;
                    }
                }
            } else {
                if (p.rho[i] <= p.rho_c_minus[i]) {
                    if (pressure <= p.tensile_strength[i]) {
                        pressure = p.tensile_strength[i];
                        p.flag_plastic[i] = 1;
                        p.rho_c_minus[i] = p.rho[i];
                        p.flag_rho_0prime[i] = 1;
                    } else {
                        p.flag_plastic[i] = -1;
                        p.flag_rho_0prime[i] = -1;
                    }
                } else {
                    if (pressure <= p.tensile_strength[i]) {
                        pressure = p.tensile_strength[i];
                    }
                    if (p.flag_plastic[i] == 1) {
                        p.flag_rho_0prime[i] = 1;
                        p.flag_plastic[i] = -1;
                    } else {
                        p.flag_plastic[i] = -1;
                        p.flag_rho_0prime[i] = -1;
                    }
                }
            }
            /* determine new rho_0prime and K if flag is set */
            if (p.flag_rho_0prime[i] == 1) {
                p.flag_rho_0prime[i] = -1;
                p.flag_plastic[i] = -1;
                p.K[i] = K_0 * pow((p.rho[i] / rho_0), gamma_K);
                if (p.K[i] < 2000.0) {
                    p.K[i] = 2000.0;
                    printf("p.K[%d] is small and set to 2000.0", i);
                }
                p.rho_0prime[i] = p.rho[i] / (1.0 + (pressure / p.K[i]));
            }
            p.p[i] = pressure;
#endif
#if PALPHA_POROSITY
        } else if (matEOS[matId] == EOS_TYPE_JUTZI || matEOS[matId] == EOS_TYPE_JUTZI_MURNAGHAN || matEOS[matId] == EOS_TYPE_JUTZI_ANEOS) {
            double pressure_solid = 0.0;
            double p_e = matporjutzi_p_elastic[matId];  	/* pressure at which the material switches from elastic to plastic */
            double p_t = matporjutzi_p_transition[matId]; /* pressure indicating a transition */
            double p_s = matporjutzi_p_compacted[matId];  /* pressure at which all pores are compacted (alpha = 1) */
            double alpha_0 = matporjutzi_alpha_0[matId];  /* distention at which the material switches from elastic to plastic */
            double alpha_t = matporjutzi_alpha_t[matId];  /* distention indicating a transition */
            double n1 = matporjutzi_n1[matId];			/* individual slope */
            double n2 = matporjutzi_n2[matId];			/* individual slope */
            double alpha_e = matporjutzi_alpha_e[matId];	/* simplified otherwise alpha_e = alpha(p_e) */
//            p.alpha_jutzi_old[i] = p.alpha_jutzi[i];	/* saving the unchanged alpha value */
            int flag_alpha_quad;	/* if this flag is set -> alpha gets calculated by a quadradic equation and not via the crush curve */
            double dp; 			/* pressure change for the calculation of dalphadp */
            double rho_0 = matTillRho0[matId];      /* parameters for Tillotson EOS -> calc pressure solid */
            double eta = p.rho[i] * p.alpha_jutzi[i] / rho_0;
            int crushcurve_style = matcrushcurve_style[matId]; /* crushcurve_style from material.cfg -> 0 is the quadratic crush curve, 1 is the real/steep crush curve by jutzi */
            if (matEOS[matId] == EOS_TYPE_JUTZI) {
                double alpha_till = matTillAlpha[matId];
                double beta_till = matTillBeta[matId];
                double a = matTilla[matId];
                double b = matTillb[matId];
                double A = matTillA[matId];
                double B = matTillB[matId];
                double E_0 = matTillE0[matId];
                double E_iv = matTillEiv[matId];
                double E_cv = matTillEcv[matId];
                p.delpdele[i] = 0.0;
                p.delpdelrho[i] = 0.0;
                /* calculate the pressure of the solid material and also
                 * calculate the derivative del p / del e and the derivative del p / del rho */
                if (eta < matRhoLimit[matId] && p.e[i] < E_cv) {
                    pressure_solid = 0.0;
                } else {
                    mu = eta - 1.0;
                    if (p.e[i] < E_iv || eta  >= 1.0) {
                        pressure_solid = (a + b / (p.e[i] / (eta * eta * E_0) + 1.0))
                                       * p.rho[i] * p.alpha_jutzi[i] * p.e[i] + A * mu + B * mu * mu;
                        p.delpdele[i] = a * p.rho[i] * p.alpha_jutzi[i] + p.rho[i] * p.alpha_jutzi[i]
                                      * b / (pow(p.e[i] / (E_0 * eta * eta) + 1.0, 2));
                        p.delpdelrho[i] = a * p.e[i] + p.e[i] * b * (1.0 + 3.0 * p.e[i] / (E_0 * eta * eta))
                                        / (pow(p.e[i] / (E_0 * eta * eta) + 1.0, 2))
                                        + A / rho_0 + 2.0 * B / rho_0 * (eta - 1.0);
                    } else if (p.e[i] > E_cv && eta < 1.0) {
                        pressure_solid = a * p.rho[i] * p.alpha_jutzi[i] * p.e[i]
                                       + (b * p.rho[i] * p.alpha_jutzi[i] * p.e[i] / (p.e[i] / (eta * eta * E_0) + 1.0)
                                       + A * mu * exp(-beta_till * (rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0)))
                                       * exp(-alpha_till * (pow(rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0, 2)));
                        p.delpdele[i] = a * p.rho[i] * p.alpha_jutzi[i] + p.rho[i] * p.alpha_jutzi[i] * b / (pow(p.e[i]/(E_0 * eta * eta) + 1.0, 2))
                                      * exp(-alpha_till * (pow(rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0, 2)));
                        p.delpdelrho[i] = a * p.e[i] + exp(-alpha_till * (pow(rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0, 2)))
                                        * (2.0 * alpha_till * rho_0 / (p.rho[i] * p.rho[i] * p.alpha_jutzi[i]
                                        * p.alpha_jutzi[i]) * (rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0)
                                        * (b * p.rho[i] * p.alpha_jutzi[i] * p.e[i] / (p.e[i] / (E_0 * eta * eta) + 1.0)
                                        + A * mu * exp(-beta_till * (rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0)))
                                        + b * p.e[i] * (1.0 + 3.0 * p.e[i] / (E_0 * eta * eta)) / (pow(p.e[i] / (E_0 * eta * eta) + 1.0, 2))
                                        + A * exp(-beta_till * (rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0))
                                        * (1.0 / rho_0 + beta_till / (p.rho[i] * p.alpha_jutzi[i])
                                        - beta_till * rho_0 / (p.rho[i] * p.rho[i] * p.alpha_jutzi[i] * p.alpha_jutzi[i])));
                    } else if (p.e[i] > E_iv && eta < 1.0) {
                        /* for intermediate states:
                         * weighted average of pressures calculated by expanded
                         * and compressed versions of Tillotson (both evaluated at e)
                         */
                        p1 = (a + b / (p.e[i] / (eta * eta * E_0) + 1.0))
                           * p.rho[i] * p.alpha_jutzi[i] * p.e[i] + A * mu + B * mu * mu;
                        p2 = a * p.rho[i] * p.alpha_jutzi[i] * p.e[i]
                           + (b * p.rho[i] * p.alpha_jutzi[i] * p.e[i] / (p.e[i] / (eta * eta * E_0) + 1.0)
                           + A * mu * exp(-beta_till * (rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0)))
                           * exp(-alpha_till * (pow(rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0, 2)));
                        pressure_solid = ((p.e[i] - E_iv) * p2 + (E_cv - p.e[i]) * p1) / (E_cv - E_iv);
                        p.delpdele[i] = ((p2 - p1) + (p.e[i] - E_iv) * a * p.rho[i] * p.alpha_jutzi[i]
                                      + p.rho[i] * p.alpha_jutzi[i] * b / (pow(p.e[i]/(E_0 * eta * eta) + 1.0, 2))
                                      * exp(-alpha_till * (pow(rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0, 2)))
                                      + (E_cv - p.e[i]) * a * p.rho[i] * p.alpha_jutzi[i] + p.rho[i] * p.alpha_jutzi[i]
                                      * b / (pow(p.e[i] / (E_0 * eta * eta) + 1.0, 2))) / (E_cv - E_iv);
                        p.delpdelrho[i] = ((a * p.e[i] + exp(-alpha_till * (pow(rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0, 2)))
                                        * (2.0 * alpha_till * rho_0 / (p.rho[i] * p.rho[i] * p.alpha_jutzi[i]
                                        * p.alpha_jutzi[i]) * (rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0)
                                        * (b * p.rho[i] * p.alpha_jutzi[i] * p.e[i] / (p.e[i] / (E_0 * eta * eta) + 1.0)
                                        + A * mu * exp(-beta_till * (rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0)))
                                        + b * p.e[i] * (1.0 + 3.0 * p.e[i] / (E_0 * eta * eta)) / (pow(p.e[i] / (E_0 * eta * eta) + 1.0, 2))
                                        + A * exp(-beta_till * (rho_0 / (p.rho[i] * p.alpha_jutzi[i]) - 1.0))
                                        * (1.0 / rho_0 + beta_till / (p.rho[i] * p.alpha_jutzi[i]) - beta_till * rho_0
                                        / (p.rho[i] * p.rho[i] * p.alpha_jutzi[i] * p.alpha_jutzi[i])))) * (p.e[i] - E_iv)
                                        + (a * p.e[i] + p.e[i] * b * (1.0 + 3.0 * p.e[i]  / (E_0 * eta * eta))
                                        / (pow(p.e[i] / (E_0 * eta * eta) + 1.0, 2))
                                        + A / rho_0 + 2.0 * B / rho_0 * (eta - 1.0))
                                        * (E_cv - p.e[i])) / (E_cv - E_iv);
                    } else {
                        printf("Deep trouble in pressure.\n");
                        printf("p[%d].e = %e\n", i, p.e[i]);
                        printf("E_iv: %e, E_cv: %e\n", E_iv, E_cv);
                        pressure_solid = 0.0;
                    }
                }
            } else if (matEOS[matId] == EOS_TYPE_JUTZI_MURNAGHAN) {
                double rho_0 = matRho0[matId];
                double n = matN[matId];
                double K_0 = matBulkmodulus[matId];
                double eta = p.rho[i] * p.alpha_jutzi[i] / rho_0;
                pressure_solid = K_0 / n * (pow(eta, n) - 1.0);
                p.delpdele[i] = 0.0;
                p.delpdelrho[i] = K_0 / rho_0 * (pow(eta, n - 1.0));
            } else if (matEOS[matId] == EOS_TYPE_JUTZI_ANEOS) {
                /* find array-indices just below the actual values of rho and e */
                i_rho = array_index(p.alpha_jutzi[i] * p.rho[i], aneos_rho_c+aneos_rho_id_c[matId], aneos_n_rho_c[matId]);
                i_e = array_index(p.e[i], aneos_e_c+aneos_e_id_c[matId], aneos_n_e_c[matId]);
                /* interpolate (bi)linearly to obtain the pressure and dp/drho and dp/de */
                bilinear_interpolation_from_linearized_plus_derivatives(p.alpha_jutzi[i] * p.rho[i], p.e[i], aneos_p_c+aneos_matrix_id_c[matId], aneos_rho_c+aneos_rho_id_c[matId], aneos_e_c+aneos_e_id_c[matId], i_rho, i_e, aneos_n_rho_c[matId], aneos_n_e_c[matId], &pressure_solid, &(p.delpdelrho[i]), &(p.delpdele[i]), i);
            }

            pressure = pressure_solid / p.alpha_jutzi[i]; /* from the P-alpha model */
            /* calculate the derivative dalpha / dpressure */
            double dalphadp_elastic = 0.0;
            // double c_0 = 5350.0; /* If dalpha_dp_elastic is NOT Zero then you need values for c_0 and c_e */
            // double c_e = 4110.0;
            // double h = 1 + (p.alpha_jutzi[i] - 1.0) * (c_e - c_0) / (c_0 * (alpha_e - 1.0));	  	/* needs to have c_e and c_0 set */
            // dalphadp_elastic = p.alpha_jutzi[i] * p.alpha_jutzi[i] / (c_0 * c_0 * rho_0) * (1.0 - (1.0 / (h * h)));
            p.dalphadp[i] = 0.0;
            if (crushcurve_style == 0) {   // quadratic crush curve
                if (pressure <= p_e) {
                    p.dalphadp[i] = dalphadp_elastic;
                } else if (pressure > p_e && pressure < p_s) {
                    p.dalphadp[i] = - 2.0 * (alpha_0 - 1.0) * (p_s - pressure) / (pow((p_s - p_e), 2));
                } else if (pressure >= p_s) {
                    p.dalphadp[i] = 0.0;
//                    p.alpha_jutzi[i] = 1.0;
				}
            } else if (crushcurve_style == 1) {   // real/steep crush curve
                if (pressure <= p_e) {
                    p.dalphadp[i] = dalphadp_elastic;
                } else if (pressure > p_e && pressure < p_t) {
                    p.dalphadp[i] = - ((alpha_0 - 1.0) / (alpha_e - 1.0)) * (alpha_e - alpha_t) * n1 * (pow(p_t - pressure, n1 - 1.0) / pow(p_t - p_e, n1))
                                  - ((alpha_0 - 1.0) / (alpha_e - 1.0)) * (alpha_t - 1.0) * n2 * (pow(p_s - pressure, n2 - 1.0) / pow(p_s - p_e, n2));
                } else if (pressure >= p_t && pressure < p_s) {
                    p.dalphadp[i] = - ((alpha_0 - 1.0) / (alpha_e - 1.0)) * (alpha_t - 1.0) * n2 * (pow(p_s - pressure, n2 - 1.0) / pow(p_s - p_e, n2));
                } else if (pressure >= p_s) {
                    p.dalphadp[i] = 0.0;
//                    p.alpha_jutzi[i] = 1.0;
                }
            } else if (crushcurve_style == 2) {  // Blum et al. 2023 experimental crush curve
                // values will go to material.cfg eventually
                // constants from Max rescaled from MPa to Pa
                const double P0 = 0.044*1e6;
                const double phi_max = 0.875;
                const double x = 8.915;
                const double a = 0;
                const double b = 7e-4*1e-6;
                // see doc/papers_and_models/porosity_models/crush_curve_Blum2023
                // alpha = (P0/P+a)**(1/x+b/x*P) + 1/phi_max
                // dalphadp = (P0/P + a)**(1/x+b/x*P) * (b/x*np.log(P0/P+a) - P0*(1/x+b/x*P)/(P**2*(P0/P)+a))
                p.dalphadp[i] = 0.0;
                //if (pressure > 0.0) {
                if (pressure > 1e0) {
                    p.dalphadp[i] = pow((P0/pressure + a), (1/x+b/x*pressure)) * (b/x*log(P0/pressure+a) -
                                P0*(1/x+b/x*pressure)/(P0*pressure+a*pressure*pressure));
                }
                if (isnan(p.dalphadp[i])) {
                    printf("ISNAN in pressure.cu: particle no. %d is killing the day.... with: p.dalphadp: %lf pressure: %.17lf\n", i, p.dalphadp[i], pressure);
                }
                if (isinf(p.dalphadp[i])) {
                    printf("ISINF in pressure.cu: particle no. %d is killing the day.... with: p.dalphadp: %lf pressure: %.17lf\n", i, p.dalphadp[i], pressure);
                }
                // printf("p.dalphadp %lf pressure %lf", p.dalphadp[i], pressure);
            } else if (crushcurve_style == 3) {  // Malamud 2023 experimental crush curve
                // if (pressure > 6e0) { // valid for pressures > 6 Pa ...
                if (pressure > 1e2) { // elastic pressure is given by the initial alpha0, see uri_crush_curve_plot.py
                    // dalpha / dp = - 0.084 * ln(10) / (-P * (0.084 * ln(P) - 0.064 * ln(10))**2)
                    p.dalphadp[i] = - 0.19341714781149988/(-pressure * (pow((0.084 * log(pressure) - 0.14736544595161893),2)));
                }
                if (isnan(p.dalphadp[i])) {
                    printf("ISNAN in pressure.cu: particle no. %d is killing the day.... with: p.dalphadp: %lf pressure: %.17lf\n", i, p.dalphadp[i], pressure);
                }
                if (isinf(p.dalphadp[i])) {
                    printf("ISINF in pressure.cu: particle no. %d is killing the day.... with: p.dalphadp: %lf pressure: %.17lf\n", i, p.dalphadp[i], pressure);
                }

            } else if (crushcurve_style == 4) {  // Malamud 2023 experimental crush curve, blue curve in figure 2-c
                // if (pressure > 6e0) { // valid for pressures > 6 Pa ...
                //if (pressure > 1e4) { // elastic pressure is given by the initial alpha0, see blue_curve_fig2c.py
                const double a = 0.41;
                const double b = 0.09;
                const double pelastic_uri = 1e6 * pow(1/(a*alpha_0), 1./b);
                //printf("pelastic uri: %le\n\n", pelastic_uri);
                if (pressure > pelastic_uri) { // elastic pressure is given by the initial alpha0, see blue_curve_fig2c.py
                // from VFF = a*p**b with a=0.41 and b=0.09 and p in MPa
                    p.dalphadp[i] = -0.7611296717250695*pow(pressure, -1.09);
                }
                if (isnan(p.dalphadp[i])) {
                    printf("ISNAN in pressure.cu: particle no. %d is killing the day.... with: p.dalphadp: %lf pressure: %.17lf\n", i, p.dalphadp[i], pressure);
                }
                if (isinf(p.dalphadp[i])) {
                    printf("ISINF in pressure.cu: particle no. %d is killing the day.... with: p.dalphadp: %lf pressure: %.17lf\n", i, p.dalphadp[i], pressure);
                }
            }
            p.dalphadrho[i] = ((pressure / (p.rho[i] * p.rho[i]) * p.delpdele[i] + p.alpha_jutzi[i] * p.delpdelrho[i]) * p.dalphadp[i])
                            / (p.alpha_jutzi[i] + p.dalphadp[i] * (pressure - p.rho[i] * p.delpdelrho[i]));
            p.f[i] = 1.0 + p.dalphadrho[i] * p.rho[i] / p.alpha_jutzi[i];
            if (p.alpha_jutzi[i] <= 1.0) {
                p.f[i] = 1.0;
                p.alpha_jutzi[i] = 1.0;
                p.dalphadp[i] = 0.0;
                p.dalphadrho[i] = 0.0;
            }
#endif
#if EPSALPHA_POROSITY
        } else if (EOS_TYPE_EPSILON == matEOS[matId]) {
            double pressure_solid = 0.0;
            double rho_0 = matTillRho0[matId];      /* parameters for Tillotson EOS -> calc pressure solid */
            double eta = p.rho[i] * p.alpha_epspor[i] / rho_0;
            double alpha_till = matTillAlpha[matId];
            double beta_till = matTillBeta[matId];
            double a = matTilla[matId];
            double b = matTillb[matId];
            double A = matTillA[matId];
            double B = matTillB[matId];
            double E_0 = matTillE0[matId];
            double E_iv = matTillEiv[matId];
            double E_cv = matTillEcv[matId];
            if (eta < matRhoLimit[matId] && p.e[i] < E_cv) {
                pressure_solid = 0.0;
            } else {
                mu = eta - 1.0;
                if (p.e[i] <= E_iv || eta  >= 1.0) {
                    pressure_solid = (a + b / (p.e[i] / (eta * eta * E_0) + 1.0))
                                   * p.rho[i] * p.alpha_epspor[i] * p.e[i] + A * mu + B * mu * mu;
                } else if (p.e[i] >= E_cv && eta >= 0.0) {
                    pressure_solid = a * p.rho[i] * p.alpha_epspor[i] * p.e[i]
                                   + (b * p.rho[i] * p.alpha_epspor[i] * p.e[i] / (p.e[i] / (eta * eta * E_0) + 1.0)
                                   + A * mu * exp(-beta_till * (rho_0 / (p.rho[i] * p.alpha_epspor[i]) - 1.0)))
                                   * exp(-alpha_till * (pow(rho_0 / (p.rho[i]
                                   * p.alpha_epspor[i]) - 1.0, 2)));
                } else if (p.e[i] > E_iv && p.e[i] < E_cv) {
                    /* for intermediate states:
                    * weighted average of pressures calculated by expanded
                    * and compressed versions of Tillotson (both evaluated at e)
                    */
                    p1 = (a + b / (p.e[i] / (eta * eta * E_0) + 1.0))
                       * p.rho[i] * p.alpha_epspor[i] * p.e[i] + A * mu + B * mu * mu;
                    p2 = a * p.rho[i] * p.alpha_epspor[i] * p.e[i]
                       + (b * p.rho[i] * p.alpha_epspor[i] * p.e[i] / (p.e[i] / (eta * eta * E_0) + 1.0)
                       + A * mu * exp(-beta_till * (rho_0 / (p.rho[i] * p.alpha_epspor[i]) - 1.0)))
                       * exp(-alpha_till * (pow(rho_0 / (p.rho[i] * p.alpha_epspor[i]) - 1.0, 2)));
                    pressure_solid = ((p.e[i] - E_iv) * p2 + (E_cv - p.e[i]) * p1) / (E_cv - E_iv);
                } else {
                    printf("Deep trouble in pressure.\n");
                    printf("p[%d].e = %e\n", i, p.e[i]);
                    printf("E_iv: %e, E_cv: %e\n", E_iv, E_cv);
                    pressure_solid = 0.0;
                }
            }
            pressure = pressure_solid / p.alpha_epspor[i]; /* from the P-alpha model which is also used here */
            p.p[i] = pressure;
            //            printf("Particle: %d \t P: %e \t Alpha: %e \t Rho: %e \t E: %e \t Mu: %e\n", i, pressure, p.alpha_epspor[i], p.rho[i], p.e[i], mu);
#endif
        } else if (EOS_TYPE_REGOLITH == matEOS[matId]) {
#if SOLID
            p.p[i] = 0.0;
            double I1;
#if DIM == 2
            double shear = matShearmodulus[matId];
            double bulk = matBulkmodulus[matId];
            double poissons_ratio = (3.0*bulk - 2.0*shear) / (2.0*(3.0*bulk + shear));
            I1 = (1 + poissons_ratio) * (p.S[stressIndex(i, 0, 0)] + p.S[stressIndex(i, 1, 1)]);
#else
            I1 = p.S[stressIndex(i,0,0)] + p.S[stressIndex(i,1,1)] + p.S[stressIndex(i,2,2)];
#endif
            p.p[i] = -I1/3.0;
#endif
        } else {
            printf("No such EOS. %d\n", matEOS[matId]);
        }

#if PALPHA_POROSITY
        if (matEOS[matId] == EOS_TYPE_JUTZI || matEOS[matId] == EOS_TYPE_JUTZI_MURNAGHAN || matEOS[matId] == EOS_TYPE_JUTZI_ANEOS) {
            p.p[i] = pressure;
        } else {
            p.alpha_jutzi_old[i] = p.alpha_jutzi[i];
        }
#endif

        // negative-pressure cap
        // note: for COLLINS_PLASTICITY, neg. pressures are adjusted only in plasticity.cu, to avoid double modification by (1-damage)
#if MOHR_COULOMB_PLASTICITY || COLLINS_PLASTICITY_SIMPLE
        register double y_0 = matCohesion[matId];
# if LOW_DENSITY_WEAKENING  // reduce strength by reducing the cohesion for low densities
        register double ldw_f, ldw_eta_limit, ldw_alpha, ldw_beta, ldw_gamma;
        if( matEOS[matId] == EOS_TYPE_MURNAGHAN ) {
            rho0 = matRho0[matId];
            eta = p.rho[i] / rho0;
        } else if( matEOS[matId] == EOS_TYPE_JUTZI ) {
            // work only with matrix densities for porous media
            rho0 = matTillRho0[matId];
            eta = p.rho[i] * p.alpha_jutzi[i] / rho0;
        } else if( matEOS[matId] == EOS_TYPE_TILLOTSON ) {
            rho0 = matTillRho0[matId];
            eta = p.rho[i] / rho0;
        } else {
            printf("ERROR. EOS_TYPE %d is not yet implemented with LOW_DENSITY_WEAKENING.\n", matEOS[matId]);
        }
        // compute  weakening factor
        if( eta >= 1.0 ) {
            ldw_f = 1.0;
        } else {
            ldw_eta_limit = matLdwEtaLimit[matId];
            ldw_gamma = matLdwGamma[matId];
            if( eta > ldw_eta_limit  ||  ldw_eta_limit <= 0.0 ) {
                ldw_alpha = matLdwAlpha[matId];
                ldw_f = pow( (eta-ldw_eta_limit)/(1.0-ldw_eta_limit), ldw_alpha ) * (1.0-ldw_gamma) + ldw_gamma;
            } else {
                ldw_beta = matLdwBeta[matId];
                ldw_f = pow( eta/ldw_eta_limit, ldw_beta ) * ldw_gamma;
            }
        }
        // finally reduce cohesion (locally)
        if( ldw_f <= 1.0  &&  ldw_f >= 0.0 ) {
            y_0 *= ldw_f;
        } else {
            printf("ERROR. Found low-density weakening factor outside [0,1], with ldw_f = %e...\n", ldw_f);
        }
# endif
        // limit negative pressures to value at zero of yield strength curve (at -cohesion)
        if( p.p[i] < -y_0)
            p.p[i] = -y_0;
#endif

#if REAL_HYDRO
        if (p.p[i] < 0.0)
            p.p[i] = 0.0;
#endif
    }   // particle loop
}
