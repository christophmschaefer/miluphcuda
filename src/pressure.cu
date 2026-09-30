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

/*
 * Tillotson EOS (notation as in Melosh 1989, "Impact Cratering") for density rho and specific internal energy e
 * of material matId. Returns the pressure and its partial derivatives dp/drho (at constant e) and dp/de (at constant
 * rho), so that the adiabatic sound speed c_s^2 = dp/drho + p/rho^2 * dp/de can be computed consistently with the
 * pressure; soundspeed.cu uses this function for EOS_TYPE_TILLOTSON and EOS_TYPE_JUTZI.
 *
 * Regimes, with eta = rho/rho_0 and z = rho_0/rho - 1:
 *   eta < rho_limit and e <= E_iv:  cold, fragmented material, p = 0
 *   eta >= 1 or e <= E_iv:          compressed form p_c = (a + b/(x + 1)) rho e + A mu + B mu^2, x = e/(E_0 eta^2)
 *   e >= E_cv:                      expanded form p_e = a rho e + (b rho e/(x + 1) + A mu exp(-beta z)) exp(-alpha z^2)
 *   E_iv < e < E_cv:                p = ((E_cv - e) p_c + (e - E_iv) p_e) / (E_cv - E_iv),
 *                                   with p_c = 0 for eta < rho_limit, so that p is continuous in e
 * For EOS_TYPE_TILLOTSON only, e is clamped to e >= 0, and for e > 100 E_cv and eta < 1 the material is treated as
 * ideal gas with polytropic_gamma from material.cfg.
 */
__device__ void tillotson_eos(double rho, double e, int matId, double *pressure, double *dpdrho, double *dpde)
{
    double rho_0 = matTillRho0[matId];
    double a = matTilla[matId];
    double b = matTillb[matId];
    double A = matTillA[matId];
    double B = matTillB[matId];
    double E_0 = matTillE0[matId];
    double E_iv = matTillEiv[matId];
    double E_cv = matTillEcv[matId];
    double alpha = matTillAlpha[matId];
    double beta = matTillBeta[matId];
    double eta = rho / rho_0;
    double mu = eta - 1.0;
    double x, z, w, exp_a, exp_b;
    double p_c, dp_c_drho, dp_c_de;
    double p_e, dp_e_drho, dp_e_de;

    if (EOS_TYPE_TILLOTSON == matEOS[matId]) {
        /* clamp e positive, will be removed when tensile strength is implemented */
        if (e < 0.0)
            e = 0.0;
        /* completely vaporized state: ideal gas, polytropic_gamma has to be set in material.cfg */
        if (e > 1e2 * E_cv && eta < 1.0) {
            double gamma = matPolytropicGamma[matId];
#if DEBUG_PRESSURE
            printf("complete vaporized state with e = %e and eta = %e, using ideal gas with gamma = %e\n",
                    e, eta, gamma);
#endif
            *pressure = (gamma - 1.0) * rho * e;
            *dpdrho = (gamma - 1.0) * e;
            *dpde = (gamma - 1.0) * rho;
            return;
        }
    }

    *pressure = 0.0;
    *dpdrho = 0.0;
    *dpde = 0.0;

    /* cold, fragmented material below rho_limit carries no pressure */
    if (eta < matRhoLimit[matId] && e <= E_iv)
        return;

    /* compressed form */
    x = e / (E_0 * eta * eta);
    p_c = (a + b / (x + 1.0)) * rho * e + A * mu + B * mu * mu;
    dp_c_drho = a * e + b * e * (1.0 + 3.0 * x) / ((x + 1.0) * (x + 1.0)) + (A + 2.0 * B * mu) / rho_0;
    dp_c_de = a * rho + b * rho / ((x + 1.0) * (x + 1.0));
    if (e <= E_iv || eta >= 1.0) {
        *pressure = p_c;
        *dpdrho = dp_c_drho;
        *dpde = dp_c_de;
        return;
    }

    /* expanded form, eta < 1 and e > E_iv from here on */
    z = rho_0 / rho - 1.0;
    exp_a = exp(-alpha * z * z);
    exp_b = exp(-beta * z);
    p_e = a * rho * e + (b * rho * e / (x + 1.0) + A * mu * exp_b) * exp_a;
    dp_e_drho = a * e + exp_a * (2.0 * alpha * z * rho_0 / (rho * rho) * (b * rho * e / (x + 1.0) + A * mu * exp_b)
            + b * e * (1.0 + 3.0 * x) / ((x + 1.0) * (x + 1.0))
            + A * exp_b * (1.0 / rho_0 + beta * mu * rho_0 / (rho * rho)));
    dp_e_de = a * rho + b * rho / ((x + 1.0) * (x + 1.0)) * exp_a;
    if (e >= E_cv) {
        *pressure = p_e;
        *dpdrho = dp_e_drho;
        *dpde = dp_e_de;
        return;
    }

    /* intermediate states, interpolation in e between compressed and expanded form */
    if (e > E_iv) {
        if (eta < matRhoLimit[matId]) {
            p_c = 0.0;
            dp_c_drho = 0.0;
            dp_c_de = 0.0;
        }
        w = (e - E_iv) / (E_cv - E_iv);
        *pressure = (1.0 - w) * p_c + w * p_e;
        *dpdrho = (1.0 - w) * dp_c_drho + w * dp_e_drho;
        *dpde = (p_e - p_c) / (E_cv - E_iv) + (1.0 - w) * dp_c_de + w * dp_e_de;
        return;
    }

    /* only reached for e = NaN */
    printf("\n\nDeep trouble in tillotson_eos.\nmaterial %d: e = %e, eta = %e, E_iv = %e, E_cv = %e\n\n",
            matId, e, eta, E_iv, E_cv);
}

__global__ void calculatePressure() {
    register int i, inc, matId;
    register double eta, rho0;
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
            double dpdrho, dpde;

            tillotson_eos(p.rho[i], p.e[i], matId, &p.p[i], &dpdrho, &dpde);
        } else if (EOS_TYPE_ANEOS == matEOS[matId]) {
            if (p.rho[i] <= 0.0) {
                p.p[i] = 0.0;
#if ANEOS_VAPOR_NO_STRENGTH
                p_rhs.aneos_phase_flag[i] = ANEOS_PHASE_ONE_PHASE;
#endif
                continue;
            }
            /* find array-indices just below the actual values of rho and e */
            i_rho = array_index(p.rho[i], aneos_rho_c+aneos_rho_id_c[matId], aneos_n_rho_c[matId]);
            // care for boundaries, we know that this might be wrong, but we trust the ANEOS tables.... and our users.. 
            if (i_rho < 0) {
                i_rho = (p.rho[i] < aneos_rho_c[aneos_rho_id_c[matId]]) ? 0 : aneos_n_rho_c[matId] - 2;
            }
            i_e = array_index(p.e[i], aneos_e_c+aneos_e_id_c[matId], aneos_n_e_c[matId]);
            if (i_e < 0 && p.e[i] >= aneos_e_c[aneos_e_id_c[matId] + aneos_n_e_c[matId] - 1]) {
                /* e above table maximum: fully vaporized, use ideal gas fallback */
                p.p[i] = (aneos_gamma_c[matId] - 1.0) * p.rho[i] * p.e[i];
#if DEBUG_PRESSURE
                printf("ideal gas fallback for particle %d with rho=%g and e=%g\n", i, p.rho[i], p.e[i]);
#endif
#if ANEOS_VAPOR_NO_STRENGTH
                p_rhs.aneos_phase_flag[i] = ANEOS_PHASE_TWO_PHASE_LV;
#endif
            } else if (i_e < 0) {
                /* e below table minimum: clamp to cold curve */
                i_e = 0;
                p.p[i] = bilinear_interpolation_from_linearized(p.rho[i], aneos_e_c[aneos_e_id_c[matId]], aneos_p_c+aneos_matrix_id_c[matId], aneos_rho_c+aneos_rho_id_c[matId], aneos_e_c+aneos_e_id_c[matId], i_rho, i_e, aneos_n_rho_c[matId], aneos_n_e_c[matId], i);
#if ANEOS_VAPOR_NO_STRENGTH
                p_rhs.aneos_phase_flag[i] = aneos_phase_flag_c[aneos_matrix_id_c[matId] + i_rho * aneos_n_e_c[matId]];
#endif       
            } else {
                /* interpolate (bi)linearly to obtain the pressure */
                p.p[i] = bilinear_interpolation_from_linearized(p.rho[i], p.e[i], aneos_p_c+aneos_matrix_id_c[matId], aneos_rho_c+aneos_rho_id_c[matId], aneos_e_c+aneos_e_id_c[matId], i_rho, i_e, aneos_n_rho_c[matId], aneos_n_e_c[matId], i);
#if ANEOS_VAPOR_NO_STRENGTH
                p_rhs.aneos_phase_flag[i] = aneos_phase_flag_c[aneos_matrix_id_c[matId] + i_rho * aneos_n_e_c[matId] + i_e];
#endif
            }
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
            int crushcurve_style = matcrushcurve_style[matId]; /* crushcurve_style from material.cfg -> 0 is the quadratic crush curve, 1 is the real/steep crush curve by jutzi */
            if (matEOS[matId] == EOS_TYPE_JUTZI) {
                /* pressure of the matrix material and its derivatives del p / del rho and del p / del e,
                 * the matrix EOS is evaluated at the matrix density alpha * rho */
                tillotson_eos(p.rho[i] * p.alpha_jutzi[i], p.e[i], matId, &pressure_solid,
                        &p.delpdelrho[i], &p.delpdele[i]);
            } else if (matEOS[matId] == EOS_TYPE_JUTZI_MURNAGHAN) {
                double rho_0 = matRho0[matId];
                double n = matN[matId];
                double K_0 = matBulkmodulus[matId];
                double eta = p.rho[i] * p.alpha_jutzi[i] / rho_0;

                if (eta < matRhoLimit[matId]) {
                    pressure_solid = 0.0;
                    p.delpdelrho[i] = 0.0;
                } else {
                    pressure_solid = K_0 / n * (pow(eta, n) - 1.0);
                    p.delpdelrho[i] = K_0 / rho_0 * (pow(eta, n - 1.0));
                }
                p.delpdele[i] = 0.0;
            } else if (matEOS[matId] == EOS_TYPE_JUTZI_ANEOS) {
                 if (p.rho[i] <= 0.0) {
                    pressure_solid = 0.0;
                    p.delpdelrho[i] = 0.0;
                    p.delpdele[i] = 0.0;
#if ANEOS_VAPOR_NO_STRENGTH
                    p_rhs.aneos_phase_flag[i] = ANEOS_PHASE_ONE_PHASE;
#endif
                    continue;
                } 
                /* find array-indices just below the actual values of rho and e */
                i_rho = array_index(p.alpha_jutzi[i] * p.rho[i], aneos_rho_c+aneos_rho_id_c[matId], aneos_n_rho_c[matId]);
                // check for underflow
                if (i_rho < 0) {
                    i_rho = (p.alpha_jutzi[i] * p.rho[i] < aneos_rho_c[aneos_rho_id_c[matId]]) ? 0 : aneos_n_rho_c[matId] - 2;
                }
                i_e = array_index(p.e[i], aneos_e_c+aneos_e_id_c[matId], aneos_n_e_c[matId]);
                if (i_e < 0 && p.e[i] >= aneos_e_c[aneos_e_id_c[matId] + aneos_n_e_c[matId] - 1]) {
                    /* e above table maximum: ideal gas fallback */
                    pressure_solid = (aneos_gamma_c[matId] - 1.0) * p.rho[i] * p.alpha_jutzi[i] * p.e[i];
                    p.delpdelrho[i] = (aneos_gamma_c[matId] - 1.0) * p.e[i];
                    p.delpdele[i]   = (aneos_gamma_c[matId] - 1.0) * p.rho[i] * p.alpha_jutzi[i];
#if DEBUG_PRESSURE
                printf("ideal gas fallback for particle %d with rho=%g and e=%g\n", i, p.rho[i], p.e[i]);
#endif
#if ANEOS_VAPOR_NO_STRENGTH
                    p_rhs.aneos_phase_flag[i] = ANEOS_PHASE_TWO_PHASE_LV;
#endif
                } else if (i_e < 0) {
                    /* e below table minimum: clamp to cold curve */
                    i_e = 0;
                    bilinear_interpolation_from_linearized_plus_derivatives(p.alpha_jutzi[i] * p.rho[i], aneos_e_c[aneos_e_id_c[matId]], aneos_p_c+aneos_matrix_id_c[matId], aneos_rho_c+aneos_rho_id_c[matId], aneos_e_c+aneos_e_id_c[matId], i_rho, i_e, aneos_n_rho_c[matId], aneos_n_e_c[matId], &pressure_solid, &(p.delpdelrho[i]), &(p.delpdele[i]), i);
#if ANEOS_VAPOR_NO_STRENGTH
                    p_rhs.aneos_phase_flag[i] = aneos_phase_flag_c[aneos_matrix_id_c[matId] + i_rho * aneos_n_e_c[matId]];
#endif
                } else {
                    /* interpolate (bi)linearly to obtain the pressure and dp/drho and dp/de */
                    bilinear_interpolation_from_linearized_plus_derivatives(p.alpha_jutzi[i] * p.rho[i], p.e[i], aneos_p_c+aneos_matrix_id_c[matId], aneos_rho_c+aneos_rho_id_c[matId], aneos_e_c+aneos_e_id_c[matId], i_rho, i_e, aneos_n_rho_c[matId], aneos_n_e_c[matId], &pressure_solid, &(p.delpdelrho[i]), &(p.delpdele[i]), i);
#if ANEOS_VAPOR_NO_STRENGTH
                    p_rhs.aneos_phase_flag[i] = aneos_phase_flag_c[aneos_matrix_id_c[matId] + i_rho * aneos_n_e_c[matId] + i_e];
#endif
                }
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

            /* f = d ln(rho_s) / d ln(rho) <= 1 by construction, since dalpha/drho <= 0.
               f > 1 can only arise from a sign change of the denominator of dalphadrho above. */
            if (p.f[i] > 1.0) {
                p.f[i] = 1.0;
            }

            if (p.alpha_jutzi[i] <= 1.0) {
                p.f[i] = 1.0;
                p.alpha_jutzi[i] = 1.0;
                p.dalphadp[i] = 0.0;
                p.dalphadrho[i] = 0.0;
            }
#endif
#if EPSALPHA_POROSITY
        } else if (EOS_TYPE_EPSILON == matEOS[matId]) {
            double pressure_solid, dpdrho, dpde;

            /* the matrix EOS is evaluated at the matrix density alpha * rho */
            tillotson_eos(p.rho[i] * p.alpha_epspor[i], p.e[i], matId, &pressure_solid, &dpdrho, &dpde);
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
