/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com and Thomas I. Maindl
 *
 * @section     LICENSE
 * Copyright (c) 2020 Christoph Schaefer
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

#ifndef _CONFIG_PARAMETER_H
#define _CONFIG_PARAMETER_H

#include <stdio.h>
#include <libconfig.h>
#include <pthread.h>
#include <assert.h>
#include "parameter.h"
#include "miluph.h"
#include "io.h"
#include "cuda_utils.h"
#include "kernel.h"
#include "extrema.h"
#include "sinking.h"


extern double *sml;
extern double *bulk_modulus;
extern double *cs_porous;
extern double *till_rho_0;
extern int *matdensity_via_kernel_sum_d;
extern __constant__ int *matdensity_via_kernel_sum;

extern double *matporjutzi_p_elastic_d;
extern double *matporjutzi_p_transition_d;
extern double *matporjutzi_p_compacted_d;
extern double *matporjutzi_alpha_0_d;
extern double *matporjutzi_alpha_e_d;
extern double *matporjutzi_alpha_t_d;
extern double *matporjutzi_n1_d;
extern double *matporjutzi_n2_d;
extern double *matcs_porous_d;
extern double *matcs_solid_d;
extern double *matisothermal_cs_d;
extern int *crushcurve_style_d;

extern double *matzeta_d;
extern double *matalpha_shakura_d;
extern double *matnu_d;
extern __constant__ double *matzeta;
extern __constant__ double *matalpha_shakura;
extern __constant__ double *matnu;

extern double *matporsirono_K_0_d;
extern double *matporsirono_rho_0_d;
extern double *matporsirono_rho_s_d;
extern double *matporsirono_gamma_K_d;
extern double *matporsirono_alpha_d;
extern double *matporsirono_pm_d;
extern double *matporsirono_phimax_d;
extern double *matporsirono_phi0_d;
extern double *matporsirono_delta_d;

extern double *matporepsilon_kappa_d;
extern double *matporepsilon_alpha_0_d;
extern double *matporepsilon_epsilon_e_d;
extern double *matporepsilon_epsilon_x_d;
extern double *matporepsilon_epsilon_c_d;

extern double *mat_f_sml_min_d;
extern double *mat_f_sml_max_d;
extern double *matSml_d;
extern int *matnoi_d;
extern int *matEOS_d;
extern double *matPolytropicK_d;
extern double *matPolytropicGamma_d;
extern double *matAlpha_d;
extern double *matBeta_d;
extern double *matBulkmodulus_d;
extern double *matShearmodulus_d;
extern double *matYieldStress_d;
extern double *matRho0_d;
extern double *matTillRho0_d;
extern double *matTillEiv_d;
extern double *matTillEcv_d;
extern double *matTillE0_d;
extern double *matTilla_d;
extern double *matTillb_d;
extern double *matTillA_d;
extern double *matTillB_d;
extern double *matTillAlpha_d;
extern double *matTillBeta_d;
extern double *matcsLimit_d;
extern double *matRhoLimit_d;
extern double *matN_d;
extern double *matCohesion_d;
extern double *matCohesionDamaged_d;
extern double *matFrictionAngle_d;
extern double *matFrictionAngleDamaged_d;
extern double *matAlphaPhi_d;
extern double *matCohesionCoefficient_d;
extern double *matMeltEnergy_d;
extern double *matDensityFloor_d;
extern double *matEnergyFloor_d;
extern double *matIsothermalSoundSpeed_d;
// ANEOS device variables
extern int *aneos_n_rho_d;
extern int *aneos_n_e_d;
extern double *aneos_bulk_cs_d;
extern double *aneos_rho_d;
extern double *aneos_e_d;
extern double *aneos_p_d;
extern double *aneos_cs_d;
extern int *aneos_rho_id_d;
extern int *aneos_e_id_d;
extern int *aneos_matrix_id_d;
extern __constant__ int *aneos_n_rho_c;
extern __constant__ int *aneos_n_e_c;
extern __constant__ double *aneos_bulk_cs_c;
extern __constant__ double *aneos_rho_c;
extern __constant__ double *aneos_e_c;
extern __constant__ double *aneos_p_c;
extern __constant__ double *aneos_cs_c;
extern __constant__ int *aneos_rho_id_c;
extern __constant__ int *aneos_e_id_c;
extern __constant__ int *aneos_matrix_id_c;

#if JC_PLASTICITY
extern double *matjc_y0_d;
extern double *matjc_B_d;
extern double *matjc_n_d;
extern double *matjc_m_d;
extern double *matjc_edot0_d;
extern double *matjc_C_d;
extern double *matjc_Tref_d;
extern double *matjc_Tmelt_d;
extern double *matCp_d;
extern double *matCV_d;
#endif

#if SOLID
extern double *matYoungModulus_d;
extern __constant__ double *matYoungModulus;
#endif

extern __constant__ double *matSml;
extern __constant__ double *mat_f_sml_min;
extern __constant__ double *mat_f_sml_max;
extern __constant__ int *matnoi;
extern __constant__ int *matEOS;
extern __constant__ double *matPolytropicK;
extern __constant__ double *matPolytropicGamma;
extern __constant__ double *matIsothermalSoundSpeed;
extern __constant__ double *matBeta;
extern __constant__ double *matAlpha;
extern __constant__ double *matBulkmodulus;
extern __constant__ double *matShearmodulus;
extern __constant__ double *matYieldStress;
extern __constant__ double *matInternalFriction;
extern __constant__ double *matInternalFrictionDamaged;
extern __constant__ double *matRho0;
extern __constant__ double *matTillRho0;
extern __constant__ double *matTillEiv;
extern __constant__ double *matTillEcv;
extern __constant__ double *matTillE0;
extern __constant__ double *matTilla;
extern __constant__ double *matTillb;
extern __constant__ double *matTillA;
extern __constant__ double *matTillB;
extern __constant__ double *matTillAlpha;
extern __constant__ double *matTillBeta;
extern __constant__ double *matcsLimit;

extern __constant__ double *matporjutzi_p_elastic;
extern __constant__ double *matporjutzi_p_transition;
extern __constant__ double *matporjutzi_p_compacted;
extern __constant__ double *matporjutzi_alpha_0;
extern __constant__ double *matporjutzi_alpha_e;
extern __constant__ double *matporjutzi_alpha_t;
extern __constant__ double *matporjutzi_n1;
extern __constant__ double *matporjutzi_n2;
extern __constant__ double *matcs_porous;
extern __constant__ double *matcs_solid;
extern __constant__ int *matcrushcurve_style;

extern __constant__ double *matporsirono_K_0;
extern __constant__ double *matporsirono_rho_0;
extern __constant__ double *matporsirono_rho_s;
extern __constant__ double *matporsirono_gamma_K;
extern __constant__ double *matporsirono_alpha;
extern __constant__ double *matporsirono_pm;
extern __constant__ double *matporsirono_phimax;
extern __constant__ double *matporsirono_phi0;
extern __constant__ double *matporsirono_delta;

extern __constant__ double *matporepsilon_kappa;
extern __constant__ double *matporepsilon_alpha_0;
extern __constant__ double *matporepsilon_epsilon_e;
extern __constant__ double *matporepsilon_epsilon_x;
extern __constant__ double *matporepsilon_epsilon_c;

#if JC_PLASTICITY
extern __constant__ double *matjc_y0;
extern __constant__ double *matjc_B;
extern __constant__ double *matjc_n;
extern __constant__ double *matjc_m;
extern __constant__ double *matjc_edot0;
extern __constant__ double *matjc_C;
extern __constant__ double *matjc_Tref;
extern __constant__ double *matjc_Tmelt;
extern __constant__ double *matCp;
extern __constant__ double *matCV;
#endif

#if ARTIFICIAL_STRESS
extern __constant__ double *matexponent_tensor;
extern __constant__ double *matepsilon_stress;
extern __constant__ double *matmean_particle_distance;
#endif

extern __constant__ int *materialId;
extern __constant__ int *density_via_kernel_sum;
extern __constant__ double *matRhoLimit;
extern __constant__ double *matN;
extern __constant__ double *matCohesion;
extern __constant__ double *matCohesionDamaged;
extern __constant__ double *matFrictionAngle;
extern __constant__ double *matFrictionAngleDamaged;
extern __constant__ double *matAlphaPhi;
extern __constant__ double *matCohesionCoefficient;
extern __constant__ double *matMeltEnergy;
extern __constant__ double *matDensityFloor;
extern __constant__ double *matEnergyFloor;
extern __constant__ double *matLdwEtaLimit;
extern __constant__ double *matLdwAlpha;
extern __constant__ double *matLdwBeta;
extern __constant__ double *matLdwGamma;
extern __constant__ double *tensorialCorrectionMatrix;
extern __constant__ double *tensorialCorrectiondWdrr;
extern __device__ double scale_height;
extern __device__ double max_abs_pressure_change;
extern __constant__ int maxNumParticles;
extern __constant__ int numRealParticles;
extern __constant__ int numChildren;
extern __constant__ int numNodes;
extern __constant__ int maxNumFlaws;
extern __constant__ double theta; // tree theta

extern int numberOfMaterials;

extern double grav_const;
extern __device__ double gravConst;


void transferMaterialsToGPU();
void cleanupMaterials(void);

#endif
