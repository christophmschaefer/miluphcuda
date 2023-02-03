/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com
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


#include "config_parameter.h"
#include "miluph.h"
#include "boundary.h"
#include "timeintegration.h"
#include "tree.h"
#include "porosity.h"
#include "pressure.h"
#include "plasticity.h"
#include "soundspeed.h"
#include "parameter.h"
#include "io.h"
#include "xsph.h"
#include "aneos.h"
#include "linalg.h"
#include "density.h"
#include "rhs.h"
#include "viscosity.h"
#include "float.h"
#include "extrema.h"
#include "sinking.h"
#include "config_parameter.h"


double *matSml_d;
int *matnoi_d;
int *matdensity_via_kernel_sum_d;
int *matEOS_d;
double *matPolytropicK_d;
double *matPolytropicGamma_d;
double *matAlpha_d;
double *matBeta_d;
double *matBulkmodulus_d;
double *matShearmodulus_d;
double *matYieldStress_d;
double *matInternalFriction_d;
double *matInternalFrictionDamaged_d;
double *matRho0_d;
double *matTillRho0_d;
double *matTillEiv_d;
double *matTillEcv_d;
double *matTillE0_d;
double *matTilla_d;
double *matTillb_d;
double *matTillA_d;
double *matTillB_d;
double *matTillAlpha_d;
double *matTillBeta_d;
double *matcsLimit_d;
double *matRhoLimit_d;
double *matN_d;
double *matCohesion_d;
double *matCohesionDamaged_d;
double *matFrictionAngle_d;
double *matFrictionAngleDamaged_d;
double *matAlphaPhi_d;
double *matCohesionCoefficient_d;
double *matMeltEnergy_d;
double *matDensityFloor_d;
double *matEnergyFloor_d;
double *matIsothermalSoundSpeed_d;
// viscosity coefficients for Navier-Stokes
double *matnu_d;
double *matalpha_shakura_d;
double *matzeta_d;
// low-density weakening parameters
double *matLdwEtaLimit_d;
double *matLdwAlpha_d;
double *matLdwBeta_d;
double *matLdwGamma_d;

#if ARTIFICIAL_STRESS
double *matexponent_tensor_d;
double *matepsilon_stress_d;
double *matmean_particle_distance_d;
__constant__ double *matexponent_tensor;
__constant__ double *matepsilon_stress;
__constant__ double *matmean_particle_distance;

// material stress parameters / material specific -> needed for ARTIFICIAL_STRESS
double *exponent_tensor;
double *epsilon_stress;
double *mean_particle_distance;
#endif

double *mat_f_sml_min_d;
double *mat_f_sml_max_d;
__constant__ double *mat_f_sml_min;
__constant__ double *mat_f_sml_max;

// ANEOS device variables (in global and constant memory)
int *aneos_n_rho_d;
int *aneos_n_e_d;
double *aneos_bulk_cs_d;
double *aneos_rho_d;
double *aneos_e_d;
double *aneos_p_d;
double *aneos_cs_d;
int *aneos_rho_id_d;
int *aneos_e_id_d;
int *aneos_matrix_id_d;
__constant__ int *aneos_n_rho_c;
__constant__ int *aneos_n_e_c;
__constant__ double *aneos_bulk_cs_c;
__constant__ double *aneos_rho_c;
__constant__ double *aneos_e_c;
__constant__ double *aneos_p_c;
__constant__ double *aneos_cs_c;
__constant__ int *aneos_rho_id_c;
__constant__ int *aneos_e_id_c;
__constant__ int *aneos_matrix_id_c;

// PALPHA_POROSITY device variables
double *matporjutzi_p_elastic_d;
double *matporjutzi_p_transition_d;
double *matporjutzi_p_compacted_d;
double *matporjutzi_alpha_0_d;
double *matporjutzi_alpha_e_d;
double *matporjutzi_alpha_t_d;
double *matporjutzi_n1_d;
double *matporjutzi_n2_d;
double *matcs_porous_d;
double *matcs_solid_d;
int *matcrushcurve_style_d;

// SIRONO_POROSITY device variables
double *matporsirono_K_0_d;
double *matporsirono_rho_0_d;
double *matporsirono_rho_s_d;
double *matporsirono_gamma_K_d;
double *matporsirono_alpha_d;
double *matporsirono_pm_d;
double *matporsirono_phimax_d;
double *matporsirono_phi0_d;
double *matporsirono_delta_d;

// EPSALPHA_POROSITY device variables
double *matporepsilon_kappa_d;
double *matporepsilon_alpha_0_d;
double *matporepsilon_epsilon_e_d;
double *matporepsilon_epsilon_x_d;
double *matporepsilon_epsilon_c_d;

double *matjc_y0_d;
double *matjc_B_d;
double *matjc_n_d;
double *matjc_m_d;
double *matjc_edot0_d;
double *matjc_C_d;
double *matjc_Tref_d;
double *matjc_Tmelt_d;
double *matCp_d;
double *matCV_d;

/* for the predictor corrector integrator */
double Smin;
double rhomin;
double emin;
double damagemin;
double alphamin;
double betamin;
double alpha_epspormin;
double epsilon_vmin;
__device__ double Smin_d;
__device__ double rhomin_d;
__device__ double emin_d;
__device__ double damagemin_d;
__device__ double alphamin_d;
__device__ double betamin_d;
__device__ double alpha_epspormin_d;
__device__ double epsilon_vmin_d;
__device__ double maxpressureDiff = 0.0;
__device__ int pressureChangeSmallEnough = FALSE;

__device__ double scale_height;

double *matYoungModulus_d;
__constant__ double *matYoungModulus;

__constant__ double *matporjutzi_p_elastic;
__constant__ double *matporjutzi_p_transition;
__constant__ double *matporjutzi_p_compacted;
__constant__ double *matporjutzi_alpha_0;
__constant__ double *matporjutzi_alpha_e;
__constant__ double *matporjutzi_alpha_t;
__constant__ double *matporjutzi_n1;
__constant__ double *matporjutzi_n2;
__constant__ double *matcs_porous;
__constant__ double *matcs_solid;
__constant__ int *matcrushcurve_style;

__constant__ double *matporsirono_K_0;
__constant__ double *matporsirono_rho_0;
__constant__ double *matporsirono_rho_s;
__constant__ double *matporsirono_gamma_K;
__constant__ double *matporsirono_alpha;
__constant__ double *matporsirono_pm;
__constant__ double *matporsirono_phimax;
__constant__ double *matporsirono_phi0;
__constant__ double *matporsirono_delta;

__constant__ double *matporepsilon_kappa;
__constant__ double *matporepsilon_alpha_0;
__constant__ double *matporepsilon_epsilon_e;
__constant__ double *matporepsilon_epsilon_x;
__constant__ double *matporepsilon_epsilon_c;

__constant__ double *matjc_y0;
__constant__ double *matjc_B;
__constant__ double *matjc_n;
__constant__ double *matjc_m;
__constant__ double *matjc_edot0;
__constant__ double *matjc_C;
__constant__ double *matjc_Tref;
__constant__ double *matjc_Tmelt;
__constant__ double *matCp;
__constant__ double *matCV;
__constant__ double *matnu;
__constant__ double *matalpha_shakura;
__constant__ double *matzeta;

__constant__ double *matSml;
__constant__ int *matnoi;
__constant__ int *matdensity_via_kernel_sum;
__constant__ int *matEOS;
__constant__ double *matPolytropicK;
__constant__ double *matPolytropicGamma;
__constant__ double *matBeta;
__constant__ double *matAlpha;
__constant__ double *matBulkmodulus;
__constant__ double *matShearmodulus;
__constant__ double *matYieldStress;
__constant__ double *matInternalFriction;
__constant__ double *matInternalFrictionDamaged;
__constant__ double *matRho0;
__constant__ double *matTillRho0;
__constant__ double *matTillEiv;
__constant__ double *matTillEcv;
__constant__ double *matTillE0;
__constant__ double *matTilla;
__constant__ double *matTillb;
__constant__ double *matTillA;
__constant__ double *matTillB;
__constant__ double *matTillAlpha;
__constant__ double *matTillBeta;
__constant__ double *matcsLimit;
__constant__ int *materialId;
__constant__ double *matRhoLimit;
__constant__ double *matIsothermalSoundSpeed;
__constant__ double *matN;
__constant__ double *matCohesion;
__constant__ double *matCohesionDamaged;
__constant__ double *matFrictionAngle;
__constant__ double *matFrictionAngleDamaged;
__constant__ double *matAlphaPhi;
__constant__ double *matCohesionCoefficient;
__constant__ double *matMeltEnergy;
__constant__ double *matDensityFloor;
__constant__ double *matEnergyFloor;
__constant__ double *matLdwEtaLimit;
__constant__ double *matLdwAlpha;
__constant__ double *matLdwBeta;
__constant__ double *matLdwGamma;
__constant__ double *tensorialCorrectionMatrix;
__constant__ double *tensorialCorrectiondWdrr;
__device__ int numParticles;
__device__ int numPointmasses;
__constant__ int maxNumParticles;
__constant__ int numRealParticles;
__constant__ int numChildren;
__constant__ int numNodes;
__constant__ int maxNumFlaws;
__device__ double max_abs_pressure_change;
__constant__ double theta; // tree theta
int *relaxedPerBlock;
double *sml;
double *till_rho_0;
double *bulk_modulus;
double *cs_porous;

int numberOfMaterials;

double grav_const;
__device__ double gravConst;



void transferMaterialsToGPU()
{
    double *pc_pointer;
    double scale_height_host;
    config_setting_t *global, *materials, *disk;

    // set some stuff for some integrators
    set_integration_parameters();

    // read global properties
    grav_const = 6.67408e-11;   // default in SI
    global = config_lookup(&param.config, "global");
    if( global != NULL ) {
        config_setting_lookup_float(global, "c_gravity", &grav_const);
    }

    // read disk properties
    disk = config_lookup(&param.config, "disk");
    if (disk != NULL) {
        config_setting_lookup_float(disk, "scale_height", &scale_height_host);
        fprintf(stdout, "Found disk scale height: %e\n", scale_height_host);
    }

    // read material properties
    materials = config_lookup(&param.config, "materials");

    if (materials != NULL) {
        int numberOfElements = config_setting_length(materials);
        int i,j;
        int maxId = 0;
        config_setting_t *material, *subset;

        // find max ID of materials
        for (i = 0; i < numberOfElements; ++i) {
            material = config_setting_get_elem(materials, i);
            int ID;
            if( !config_setting_lookup_int(material, "ID", &ID) ) {
                fprintf(stderr, "Error. Found material without ID in config file...\n");
                exit(1);
            }
            if (param.verbose) {
                fprintf(stdout, "Found material ID: %d\n", ID);
            }
            maxId = max(ID, maxId);
        }
        if( maxId != numberOfElements - 1 ) {
            fprintf(stderr, "Error. Material-IDs in config file have to be 0, 1, 2,...\n");
            exit(1);
        }

        numberOfMaterials = numberOfElements;   // global, needed externally

        // allocate memory on host
        sml = (double*)calloc(numberOfElements,sizeof(double));
        int *eos = (int*)calloc(numberOfElements,sizeof(int));
        int *noi = (int*)calloc(numberOfElements,sizeof(int));
        double *f_sml_min = (double *) calloc(numberOfElements, sizeof(double));
        double *f_sml_max = (double *) calloc(numberOfElements, sizeof(double));
        // seting some reasonable values for sml factor
        for (i = 0; i < numberOfMaterials; i++) {
            f_sml_min[i] = 1.0;
            f_sml_max[i] = 1.0;
        }
        double *alpha = (double*)calloc(numberOfElements, sizeof(double));
        double *beta = (double*)calloc(numberOfElements, sizeof(double));
        double *polytropic_K = (double*)calloc(numberOfElements, sizeof(double));
        double *polytropic_gamma = (double*)calloc(numberOfElements, sizeof(double));
        double *n = (double*)calloc(numberOfElements, sizeof(double));
        double *nu = (double *) calloc(numberOfElements, sizeof(double));
	    double *alpha_shakura = (double *) calloc(numberOfElements, sizeof(double));
        double *eta = (double *) calloc(numberOfElements, sizeof(double));
        double *zeta = (double *) calloc(numberOfElements, sizeof(double));
        double *rho_0 = (double*)calloc(numberOfElements, sizeof(double));
        double *rho_limit = (double*)calloc(numberOfElements, sizeof(double));
        till_rho_0 = (double*)calloc(numberOfElements, sizeof(double));
        double *till_A = (double*)calloc(numberOfElements, sizeof(double));
        double *till_B = (double*)calloc(numberOfElements, sizeof(double));
        double *till_E_0 = (double*)calloc(numberOfElements, sizeof(double));
        double *till_E_iv = (double*)calloc(numberOfElements, sizeof(double));
        double *till_E_cv = (double*)calloc(numberOfElements, sizeof(double));
        double *till_a = (double*)calloc(numberOfElements, sizeof(double));
        double *till_b = (double*)calloc(numberOfElements, sizeof(double));
        double *till_alpha = (double*)calloc(numberOfElements, sizeof(double));
        double *till_beta = (double*)calloc(numberOfElements, sizeof(double));
        double *csLimit = (double*)calloc(numberOfElements, sizeof(double));
        double *isothermal_cs = (double*)calloc(numberOfElements, sizeof(double));
        // begin of ANEOS allocations in host memory (global variables, defined in 'aneos.cu')
        g_eos_is_aneos = (int*)calloc(numberOfElements, sizeof(int));
        g_aneos_tab_file = (const char**)calloc(numberOfElements, sizeof(const char*));     // not necessary to allocate (and free) mem for individual strings - this should be managed by libconfig
        g_aneos_n_rho = (int*)calloc(numberOfElements, sizeof(int));
        g_aneos_n_e = (int*)calloc(numberOfElements, sizeof(int));
        g_aneos_rho_0 = (double*)calloc(numberOfElements, sizeof(double));
        g_aneos_bulk_cs = (double*)calloc(numberOfElements, sizeof(double));
        g_aneos_rho = (double**)calloc(numberOfElements, sizeof(double*));
        g_aneos_e = (double**)calloc(numberOfElements, sizeof(double*));
        g_aneos_p = (double***)calloc(numberOfElements, sizeof(double**));
        g_aneos_cs = (double***)calloc(numberOfElements, sizeof(double**));
#if MORE_ANEOS_OUTPUT
        g_aneos_T = (double***)calloc(numberOfElements, sizeof(double**));
        g_aneos_entropy = (double***)calloc(numberOfElements, sizeof(double**));
        g_aneos_phase_flag = (int***)calloc(numberOfElements, sizeof(int**));
#endif
        int n_aneos_mat = 0;
        /* arrays to hold 'start indices' for all ANEOS materials and -1 if EOS != ANEOS (e.g. aneos_rho_id = [0,-1,g_aneos_n_rho[0],-1,-1]
        if materials no. 0 and 2 use ANEOS), necessary for resolving linearizations of multi-dim arrays on GPU */
        int *aneos_rho_id = (int*)calloc(numberOfElements, sizeof(int));
        int *aneos_e_id = (int*)calloc(numberOfElements, sizeof(int));
        int *aneos_matrix_id = (int*)calloc(numberOfElements, sizeof(int));
        for (i = 0; i < numberOfElements; i++)
            aneos_rho_id[i] = aneos_e_id[i] = aneos_matrix_id[i] = -1;
        /* variables to hold running indices for filling arrays holding 'start indices' */
        int run_aneos_rho_id = 0;
        int run_aneos_e_id = 0;
        int run_aneos_matrix_id = 0;
        // end of ANEOS allocations in host memory
        double *shear_modulus = (double*)calloc(numberOfElements, sizeof(double));
        bulk_modulus = (double*)calloc(numberOfElements, sizeof(double));
        int *density_via_kernel_sum = (int*)calloc(numberOfElements, sizeof(int));
        double *yield_stress = (double*)calloc(numberOfElements, sizeof(double));
        double *internal_friction = (double*)calloc(numberOfElements, sizeof(double));
        double *internal_friction_damaged = (double*)calloc(numberOfElements, sizeof(double));
        double *cohesion = (double*)calloc(numberOfElements, sizeof(double));
        double *cohesion_damaged = (double*)calloc(numberOfElements, sizeof(double));
        double *friction_angle = (double*)calloc(numberOfElements, sizeof(double));
        double *friction_angle_damaged = (double*)calloc(numberOfElements, sizeof(double));
        double *alpha_phi = (double*)calloc(numberOfElements, sizeof(double));
        double *cohesion_coefficient = (double*)calloc(numberOfElements, sizeof(double));
        double *melt_energy = (double*)calloc(numberOfElements, sizeof(double));
        double *density_floor = (double*)calloc(numberOfElements,sizeof(double));
        double *energy_floor = (double*)calloc(numberOfElements,sizeof(double));
#if ARTIFICIAL_STRESS
        double *exponent_tensor = (double*) calloc(numberOfElements, sizeof(double));
        double *epsilon_stress = (double*) calloc(numberOfElements, sizeof(double));
        double *mean_particle_distance = (double*) calloc(numberOfElements, sizeof(double));
#endif
#if LOW_DENSITY_WEAKENING
        double *ldw_eta_limit = (double*)calloc(numberOfElements, sizeof(double));
        double *ldw_alpha = (double*)calloc(numberOfElements, sizeof(double));
        double *ldw_beta = (double*)calloc(numberOfElements, sizeof(double));
        double *ldw_gamma = (double*)calloc(numberOfElements, sizeof(double));
#endif
#if PALPHA_POROSITY
        double *porjutzi_p_elastic = (double*)calloc(numberOfElements, sizeof(double));
        double *porjutzi_p_transition = (double*)calloc(numberOfElements, sizeof(double));
        double *porjutzi_p_compacted = (double*)calloc(numberOfElements, sizeof(double));
        double *porjutzi_alpha_0 = (double*)calloc(numberOfElements, sizeof(double));
        double *porjutzi_alpha_e = (double*)calloc(numberOfElements, sizeof(double));
        double *porjutzi_alpha_t = (double*)calloc(numberOfElements, sizeof(double));
        double *porjutzi_n1 = (double*)calloc(numberOfElements, sizeof(double));
        double *porjutzi_n2 = (double*)calloc(numberOfElements, sizeof(double));
        cs_porous = (double*)calloc(numberOfElements, sizeof(double));
        double *cs_solid = (double*)calloc(numberOfElements, sizeof(double));
        double max_abs_pressure_change_host = DBL_MAX;
        int *crushcurve_style = (int*)calloc(numberOfElements, sizeof(int));
#endif
#if SIRONO_POROSITY
        double *porsirono_K_0 = (double*)calloc(numberOfElements, sizeof(double));
        double *porsirono_rho_0 = (double*)calloc(numberOfElements, sizeof(double));
        double *porsirono_rho_s = (double*)calloc(numberOfElements, sizeof(double));
        double *porsirono_gamma_K = (double*)calloc(numberOfElements, sizeof(double));
        double *porsirono_alpha = (double*)calloc(numberOfElements, sizeof(double));
        double *porsirono_pm = (double*)calloc(numberOfElements, sizeof(double));
        double *porsirono_phimax = (double*)calloc(numberOfElements, sizeof(double));
        double *porsirono_phi0 = (double*)calloc(numberOfElements, sizeof(double));
        double *porsirono_delta = (double*)calloc(numberOfElements, sizeof(double));
#endif
#if EPSALPHA_POROSITY
        double *porepsilon_kappa = (double*)calloc(numberOfElements, sizeof(double));
        double *porepsilon_alpha_0 = (double*)calloc(numberOfElements, sizeof(double));
        double *porepsilon_epsilon_e = (double*)calloc(numberOfElements, sizeof(double));
        double *porepsilon_epsilon_x = (double*)calloc(numberOfElements, sizeof(double));
        double *porepsilon_epsilon_c = (double*)calloc(numberOfElements, sizeof(double));
#endif
#if SOLID
        double *young_modulus = (double*)calloc(numberOfElements, sizeof(double));
#endif
#if JC_PLASTICITY
        double *jc_y0 = (double*)calloc(numberOfElements, sizeof(double));
        double *jc_B = (double*)calloc(numberOfElements, sizeof(double));
        double *jc_n = (double*)calloc(numberOfElements, sizeof(double));
        double *jc_m = (double*)calloc(numberOfElements, sizeof(double));
        double *jc_edot0 = (double*)calloc(numberOfElements, sizeof(double));
        double *jc_C = (double*)calloc(numberOfElements, sizeof(double));
        double *jc_Tref = (double*)calloc(numberOfElements, sizeof(double));
        double *jc_Tmelt = (double*)calloc(numberOfElements, sizeof(double));
        double *Cp = (double*)calloc(numberOfElements, sizeof(double));
        double *CV = (double*)calloc(numberOfElements, sizeof(double));
#endif


        // read material config file
        for (i = 0; i < numberOfElements; ++i) {
            material = config_setting_get_elem(materials, i);
            int ID;
            config_setting_lookup_int(material, "ID", &ID);
	        fprintf(stdout, "Reading information about material ID %d out of %d in total...\n", ID, numberOfElements);
            config_setting_lookup_float(material, "sml", &sml[ID]);
#if VARIABLE_SML
#if FIXED_NOI
            config_setting_lookup_int(material, "interactions", &noi[ID]);
#endif
            // sml will not get smaller than factor_sml_min * sml
            // and not get larger than factor_sml_max * sml
            config_setting_lookup_float(material, "factor_sml_min", &f_sml_min[ID]);
            config_setting_lookup_float(material, "factor_sml_max", &f_sml_max[ID]);
#endif
#if ARTIFICIAL_VISCOSITY
            alpha[ID] = 1.0;   // set defaults
            beta[ID] = 2.0;
            if( subset = config_setting_get_member(material, "artificial_viscosity") ) {
                config_setting_lookup_float(subset, "alpha", &alpha[ID]);
                config_setting_lookup_float(subset, "beta", &beta[ID]);
            }
#endif
#if ARTIFICIAL_STRESS
            if( !(subset = config_setting_get_member(material, "artificial_stress")) ) {
                fprintf(stderr, "Error reading material config file. Subgroup 'artificial_stress' is missing for material with ID %d.\n", ID);
                exit(EXIT_FAILURE);
            }
            config_setting_lookup_float(subset, "exponent_tensor", &exponent_tensor[ID]);
            config_setting_lookup_float(subset, "epsilon_stress", &epsilon_stress[ID]);
            config_setting_lookup_float(subset, "mean_particle_distance", &mean_particle_distance[ID]);
#endif
#if NAVIER_STOKES
            if( !(subset = config_setting_get_member(material, "physical_viscosity")) ) {
                fprintf(stderr, "Error reading material config file. Subgroup 'physical_viscosity' is missing for material with ID %d.\n", ID);
                exit(EXIT_FAILURE);
            }
            // note nu and eta depend via density: nu = eta/rho
            config_setting_lookup_float(subset, "eta", &eta[ID]);
            config_setting_lookup_float(subset, "zeta", &zeta[ID]);
  	        config_setting_lookup_float(subset, "alpha_shakura", &alpha_shakura[ID]);
            config_setting_lookup_float(subset, "nu", &nu[ID]);
#endif
#if LOW_DENSITY_WEAKENING
            ldw_gamma[ID] = 1.0;    // set default
            if( subset = config_setting_get_member(material, "low_density_weakening") ) {
                config_setting_lookup_float(subset, "eta_limit", &ldw_eta_limit[ID]);
                config_setting_lookup_float(subset, "alpha", &ldw_alpha[ID]);
                config_setting_lookup_float(subset, "beta", &ldw_beta[ID]);
                config_setting_lookup_float(subset, "gamma", &ldw_gamma[ID]);
            }
#endif

            // read group 'eos'
            if( !(subset = config_setting_get_member(material, "eos")) ) {
                fprintf(stderr, "Error reading material config file. Subgroup 'eos' is missing for material with ID %d.\n", ID);
                exit(EXIT_FAILURE);
            }
            if( !config_setting_lookup_int(subset, "type", &eos[ID]) ) {
                fprintf(stderr, "Error. Each material needs an eos.type in the material config file.\n");
                exit(EXIT_FAILURE);
            }
            config_setting_lookup_float(subset, "polytropic_K", &polytropic_K[ID]);
            config_setting_lookup_float(subset, "polytropic_gamma", &polytropic_gamma[ID]);
            config_setting_lookup_float(subset, "isothermal_soundspeed", &isothermal_cs[ID]);
            config_setting_lookup_float(subset, "bulk_modulus", &bulk_modulus[ID]);
            config_setting_lookup_float(subset, "shear_modulus", &shear_modulus[ID]);
            config_setting_lookup_float(subset, "yield_stress", &yield_stress[ID]);
            config_setting_lookup_float(subset, "rho_0", &rho_0[ID]);
            config_setting_lookup_float(subset, "till_rho_0", &till_rho_0[ID]);
            config_setting_lookup_float(subset, "till_E_0", &till_E_0[ID]);
            config_setting_lookup_float(subset, "till_E_iv", &till_E_iv[ID]);
            config_setting_lookup_float(subset, "till_E_cv", &till_E_cv[ID]);
            config_setting_lookup_float(subset, "till_a", &till_a[ID]);
            config_setting_lookup_float(subset, "till_b", &till_b[ID]);
            // begin reading ANEOS data from material file to host memory
            config_setting_lookup_string(subset, "table_path", &g_aneos_tab_file[ID]);
            config_setting_lookup_int(subset, "n_rho", &g_aneos_n_rho[ID]);
            config_setting_lookup_int(subset, "n_e", &g_aneos_n_e[ID]);
            config_setting_lookup_int(subset, "density_via_kernel_sum", &density_via_kernel_sum[ID]);
#if INTEGRATE_DENSITY
            if (density_via_kernel_sum[ID] > 0) {
                fprintf(stdout, "Determing the density of material %d via kernel sum and not by integration of the continuity equation.\n", ID);
            }
#endif
            config_setting_lookup_float(subset, "aneos_rho_0", &g_aneos_rho_0[ID]);
            config_setting_lookup_float(subset, "aneos_bulk_cs", &g_aneos_bulk_cs[ID]);
            // end reading ANEOS data from material file to host memory, begin reading ANEOS lookup table to host memory
            if (eos[ID] == EOS_TYPE_ANEOS  ||  eos[ID] == EOS_TYPE_JUTZI_ANEOS) {
                g_eos_is_aneos[ID] = TRUE;
                aneos_rho_id[ID] = run_aneos_rho_id;
                run_aneos_rho_id += g_aneos_n_rho[ID];
                aneos_e_id[ID] = run_aneos_e_id;
                run_aneos_e_id += g_aneos_n_e[ID];
                aneos_matrix_id[ID] = run_aneos_matrix_id;
                run_aneos_matrix_id += g_aneos_n_rho[ID]*g_aneos_n_e[ID];
                n_aneos_mat++;
                if ((g_aneos_rho[ID] = (double*)calloc(g_aneos_n_rho[ID], sizeof(double))) == NULL)
                    ERRORVAR("ERROR during memory allocation for ANEOS lookup table in '%s'\n", g_aneos_tab_file[ID])
                if ((g_aneos_e[ID] = (double*)calloc(g_aneos_n_e[ID], sizeof(double))) == NULL)
                    ERRORVAR("ERROR during memory allocation for ANEOS lookup table in '%s'\n", g_aneos_tab_file[ID])
                if ((g_aneos_p[ID] = (double**)calloc(g_aneos_n_rho[ID], sizeof(double*))) == NULL)
                    ERRORVAR("ERROR during memory allocation for ANEOS lookup table in '%s'\n", g_aneos_tab_file[ID])
                for (j=0; j<g_aneos_n_rho[ID]; j++)
                    if ((g_aneos_p[ID][j] = (double*)calloc(g_aneos_n_e[ID], sizeof(double))) == NULL)
                        ERRORVAR("ERROR during memory allocation for ANEOS lookup table in '%s'\n", g_aneos_tab_file[ID])
                if ((g_aneos_cs[ID] = (double**)calloc(g_aneos_n_rho[ID], sizeof(double*))) == NULL)
                    ERRORVAR("ERROR during memory allocation for ANEOS lookup table in '%s'\n", g_aneos_tab_file[ID])
                for (j=0; j<g_aneos_n_rho[ID]; j++)
                    if ((g_aneos_cs[ID][j] = (double*)calloc(g_aneos_n_e[ID], sizeof(double))) == NULL)
                        ERRORVAR("ERROR during memory allocation for ANEOS lookup table in '%s'\n", g_aneos_tab_file[ID])
#if MORE_ANEOS_OUTPUT
                if ((g_aneos_T[ID] = (double**)calloc(g_aneos_n_rho[ID], sizeof(double*))) == NULL)
                    ERRORVAR("ERROR during memory allocation for ANEOS lookup table in '%s'\n", g_aneos_tab_file[ID])
                for (j=0; j<g_aneos_n_rho[ID]; j++)
                    if ((g_aneos_T[ID][j] = (double*)calloc(g_aneos_n_e[ID], sizeof(double))) == NULL)
                        ERRORVAR("ERROR during memory allocation for ANEOS lookup table in '%s'\n", g_aneos_tab_file[ID])
                if ((g_aneos_entropy[ID] = (double**)calloc(g_aneos_n_rho[ID], sizeof(double*))) == NULL)
                    ERRORVAR("ERROR during memory allocation for ANEOS lookup table in '%s'\n", g_aneos_tab_file[ID])
                for (j=0; j<g_aneos_n_rho[ID]; j++)
                    if ((g_aneos_entropy[ID][j] = (double*)calloc(g_aneos_n_e[ID], sizeof(double))) == NULL)
                        ERRORVAR("ERROR during memory allocation for ANEOS lookup table in '%s'\n", g_aneos_tab_file[ID])
                if ((g_aneos_phase_flag[ID] = (int**)calloc(g_aneos_n_rho[ID], sizeof(int*))) == NULL)
                    ERRORVAR("ERROR during memory allocation for ANEOS lookup table in '%s'\n", g_aneos_tab_file[ID])
                for (j=0; j<g_aneos_n_rho[ID]; j++)
                    if ((g_aneos_phase_flag[ID][j] = (int*)calloc(g_aneos_n_e[ID], sizeof(int))) == NULL)
                        ERRORVAR("ERROR during memory allocation for ANEOS lookup table in '%s'\n", g_aneos_tab_file[ID])
#endif
#if MORE_ANEOS_OUTPUT
                initialize_aneos_eos_full(g_aneos_tab_file[ID], g_aneos_n_rho[ID], g_aneos_n_e[ID], g_aneos_rho[ID], g_aneos_e[ID], g_aneos_p[ID], g_aneos_T[ID], g_aneos_cs[ID], g_aneos_entropy[ID], g_aneos_phase_flag[ID] );
#else
                initialize_aneos_eos_basic(g_aneos_tab_file[ID], g_aneos_n_rho[ID], g_aneos_n_e[ID], g_aneos_rho[ID], g_aneos_e[ID], g_aneos_p[ID], g_aneos_cs[ID] );
#endif
            }
            //end reading ANEOS lookup table to host memory
#if PALPHA_POROSITY
            config_setting_lookup_float(subset, "porjutzi_p_elastic", &porjutzi_p_elastic[ID]);
            if (porjutzi_p_elastic[ID] > 0.0)
                max_abs_pressure_change_host = fmin(max_abs_pressure_change_host, porjutzi_p_elastic[ID]);
            config_setting_lookup_float(subset, "porjutzi_p_transition", &porjutzi_p_transition[ID]);
            config_setting_lookup_float(subset, "porjutzi_p_compacted", &porjutzi_p_compacted[ID]);
            if( !config_setting_lookup_float(subset, "porjutzi_alpha_0", &porjutzi_alpha_0[ID]) )
                porjutzi_alpha_0[ID] = 1.0;
            if( !config_setting_lookup_float(subset, "porjutzi_alpha_e", &porjutzi_alpha_e[ID]) )
                porjutzi_alpha_e[ID] = 1.0;
            if( !config_setting_lookup_float(subset, "porjutzi_alpha_t", &porjutzi_alpha_t[ID]) )
                porjutzi_alpha_t[ID] = 1.0;
            config_setting_lookup_float(subset, "porjutzi_n1", &porjutzi_n1[ID]);
            config_setting_lookup_float(subset, "porjutzi_n2", &porjutzi_n2[ID]);
            // read cs_porous or set default
            if( !config_setting_lookup_float(subset, "cs_porous", &cs_porous[ID]) ) {
                if( eos[ID] == EOS_TYPE_JUTZI ) {
                    cs_porous[ID] = 0.5 * sqrt(till_A[ID]/till_rho_0[ID]);
                } else if( eos[ID] == EOS_TYPE_JUTZI_ANEOS ) {
                    cs_porous[ID] = 0.5 * g_aneos_bulk_cs[ID];
                } else if( eos[ID] == EOS_TYPE_JUTZI_MURNAGHAN ) {
                    cs_porous[ID] = 0.5 * sqrt(bulk_modulus[ID]/rho_0[ID]);
                }
            }
            // set cs_solid for p-alpha + Murnaghan EoS, for all other p-alpha EoS the solid sound speed is computed dynamically
            if (eos[ID] == EOS_TYPE_JUTZI_MURNAGHAN) {
                cs_solid[ID] = sqrt(bulk_modulus[ID] / rho_0[ID] / porjutzi_alpha_0[ID]);
            }
            config_setting_lookup_int(subset, "crushcurve_style", &crushcurve_style[ID]);
#endif
#if SIRONO_POROSITY
            config_setting_lookup_float(subset, "porsirono_K_0", &porsirono_K_0[ID]);
            config_setting_lookup_float(subset, "porsirono_rho_0", &porsirono_rho_0[ID]);
            config_setting_lookup_float(subset, "porsirono_rho_s", &porsirono_rho_s[ID]);
            config_setting_lookup_float(subset, "porsirono_gamma_K", &porsirono_gamma_K[ID]);
            config_setting_lookup_float(subset, "porsirono_alpha", &porsirono_alpha[ID]);
            config_setting_lookup_float(subset, "porsirono_pm", &porsirono_pm[ID]);
            config_setting_lookup_float(subset, "porsirono_phimax", &porsirono_phimax[ID]);
            config_setting_lookup_float(subset, "porsirono_phi0", &porsirono_phi0[ID]);
            config_setting_lookup_float(subset, "porsirono_delta", &porsirono_delta[ID]);
#endif
#if EPSALPHA_POROSITY
            config_setting_lookup_float(subset, "porepsilon_kappa", &porepsilon_kappa[ID]);
            config_setting_lookup_float(subset, "porepsilon_alpha_0", &porepsilon_alpha_0[ID]);
            config_setting_lookup_float(subset, "porepsilon_epsilon_e", &porepsilon_epsilon_e[ID]);
            config_setting_lookup_float(subset, "porepsilon_epsilon_x", &porepsilon_epsilon_x[ID]);
            config_setting_lookup_float(subset, "porepsilon_epsilon_c", &porepsilon_epsilon_c[ID]);
#endif

            config_setting_lookup_float(subset, "till_A", &till_A[ID]);
            config_setting_lookup_float(subset, "till_B", &till_B[ID]);
            config_setting_lookup_float(subset, "till_alpha", &till_alpha[ID]);
            config_setting_lookup_float(subset, "till_beta", &till_beta[ID]);
            // read cs_limit or set default
  	        if( !config_setting_lookup_float(subset, "cs_limit", &csLimit[ID]) ) {
                if( eos[ID] == EOS_TYPE_TILLOTSON  ||  eos[ID] == EOS_TYPE_JUTZI ) {
  		            csLimit[ID] = 0.01 * sqrt(till_A[ID]/till_rho_0[ID]);
                } else if( eos[ID] == EOS_TYPE_ANEOS  ||  eos[ID] == EOS_TYPE_JUTZI_ANEOS ) {
                    csLimit[ID] = 0.01 * g_aneos_bulk_cs[ID];
                }
            }
            // read rho_limit or set default
            if( !config_setting_lookup_float(subset, "rho_limit", &rho_limit[ID]) ) {
                if (eos[ID] == EOS_TYPE_TILLOTSON  ||  eos[ID] == EOS_TYPE_JUTZI || eos[ID] == EOS_TYPE_MURNAGHAN || eos[ID] == EOS_TYPE_JUTZI_MURNAGHAN) {
                    rho_limit[ID] = 0.9;
                }
            }
            // read n (for Murnaghan EoS) or set default
            if( !config_setting_lookup_float(subset, "n", &n[ID]) ) {
                n[ID] = 1.0;
            }
            config_setting_lookup_float(subset, "cohesion", &cohesion[ID]);
            config_setting_lookup_float(subset, "cohesion_damaged", &cohesion_damaged[ID]);
            config_setting_lookup_float(subset, "friction_angle", &friction_angle[ID]);
            config_setting_lookup_float(subset, "friction_angle_damaged", &friction_angle_damaged[ID]);
            config_setting_lookup_float(subset, "melt_energy", &melt_energy[ID]);
#if DIM == 2
            alpha_phi[ID] = tan(friction_angle[ID]) / sqrt(9 + 12*tan(friction_angle[ID])*tan(friction_angle[ID]));
            cohesion_coefficient[ID] = 3*(cohesion[ID]) / sqrt(9 + 12*tan(friction_angle[ID])*tan(friction_angle[ID]));
#else
            alpha_phi[ID] = 2*sin(friction_angle[ID]) / (sqrt(3.)*(3-sin(friction_angle[ID])));
            cohesion_coefficient[ID] = 6*(cohesion[ID]) * cos(friction_angle[ID]) / (sqrt(3.)*(3-sin(friction_angle[ID])));
#endif
            /* internal friction coefficient (normally \mu) is tan of angle of internal friction */
            internal_friction[ID] = tan(friction_angle[ID]);
            internal_friction_damaged[ID] = tan(friction_angle_damaged[ID]);
#if SOLID
            young_modulus[ID] = 9.0*bulk_modulus[ID]*shear_modulus[ID]/(3.0*bulk_modulus[ID]+shear_modulus[ID]);
#endif

#if JC_PLASTICITY
            config_setting_lookup_float(subset, "jc_y0", &jc_y0[ID]);
            config_setting_lookup_float(subset, "jc_B", &jc_B[ID]);
            config_setting_lookup_float(subset, "jc_n", &jc_n[ID]);
            config_setting_lookup_float(subset, "jc_m", &jc_m[ID]);
            config_setting_lookup_float(subset, "jc_edot0", &jc_edot0[ID]);
            config_setting_lookup_float(subset, "jc_C", &jc_C[ID]);
            config_setting_lookup_float(subset, "jc_Tref", &jc_Tref[ID]);
            config_setting_lookup_float(subset, "jc_Tmelt", &jc_Tmelt[ID]);
            config_setting_lookup_float(subset, "Cp", &Cp[ID]);
            config_setting_lookup_float(subset, "CV", &CV[ID]);
#endif

            // read density_floor or set default
            if( !config_setting_lookup_float(material, "density_floor", &density_floor[ID]) ) {
                switch (eos[ID]) {
                    case (EOS_TYPE_MURNAGHAN):
                        density_floor[ID] = rho_0[ID] * 0.01;
                        break;
                    case (EOS_TYPE_JUTZI_MURNAGHAN):
                        density_floor[ID] = rho_0[ID] * 0.01;
                        break;
                    case (EOS_TYPE_TILLOTSON):
                        density_floor[ID] = till_rho_0[ID] * 0.01;
                        break;
                    case (EOS_TYPE_JUTZI):
                        density_floor[ID] = till_rho_0[ID] * 0.01;
                        break;
                    case (EOS_TYPE_ANEOS):
                        density_floor[ID] = g_aneos_rho_0[ID] * 0.01;
                        break;
                    case (EOS_TYPE_JUTZI_ANEOS):
                        density_floor[ID] = g_aneos_rho_0[ID] * 0.01;
                        break;
#if SIRONO_POROSITY
                    case (EOS_TYPE_SIRONO):
                        density_floor[ID] = porsirono_rho_s[ID] * 0.01;
                        break;
#endif
                    case (EOS_TYPE_EPSILON):
                        density_floor[ID] = till_rho_0[ID] * 0.01;
                        break;
                    case (EOS_TYPE_VISCOUS_REGOLITH):
                        density_floor[ID] = rho_0[ID] * 0.01;
                        break;
                    case (EOS_TYPE_IDEAL_GAS):
                        density_floor[ID] = 0.0;
                        break;
                    case (EOS_TYPE_LOCALLY_ISOTHERMAL_GAS):
                        density_floor[ID] = 0.0;
                        break;
                    default:
                        density_floor[ID] = 0.0;
                        break;
                }
            }
            
            // read energy_floor or set to -inf
            if( !config_setting_lookup_float(material, "energy_floor", &energy_floor[ID]) )
                energy_floor[ID] = -1e30;

        }  // loop over materials


        // begin memory allocation on device
#if PALPHA_POROSITY
        cudaVerify(cudaMalloc((void **)&matporjutzi_p_elastic_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporjutzi_p_transition_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporjutzi_p_compacted_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporjutzi_alpha_0_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporjutzi_alpha_e_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporjutzi_alpha_t_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporjutzi_n1_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporjutzi_n2_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matcs_porous_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matcs_solid_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matcrushcurve_style_d, numberOfElements*sizeof(int)));
#endif
#if VARIABLE_SML
        cudaVerify(cudaMalloc((void **)&mat_f_sml_max_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&mat_f_sml_min_d, numberOfElements*sizeof(double)));
#endif
#if SIRONO_POROSITY
        cudaVerify(cudaMalloc((void **)&matporsirono_K_0_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporsirono_rho_0_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporsirono_rho_s_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporsirono_gamma_K_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporsirono_alpha_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporsirono_pm_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporsirono_phimax_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporsirono_phi0_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporsirono_delta_d, numberOfElements*sizeof(double)));
#endif
#if EPSALPHA_POROSITY
        cudaVerify(cudaMalloc((void **)&matporepsilon_kappa_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporepsilon_alpha_0_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporepsilon_epsilon_e_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporepsilon_epsilon_x_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matporepsilon_epsilon_c_d, numberOfElements*sizeof(double)));
#endif
#if NAVIER_STOKES
        cudaVerify(cudaMalloc((void **)&matnu_d, numberOfElements*sizeof(double)));
	    cudaVerify(cudaMalloc((void **)&matalpha_shakura_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matzeta_d, numberOfElements*sizeof(double)));
#endif
#if ARTIFICIAL_STRESS
        cudaVerify(cudaMalloc((void **)&matexponent_tensor_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matepsilon_stress_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matmean_particle_distance_d, numberOfElements*sizeof(double)));
#endif
#if LOW_DENSITY_WEAKENING
        cudaVerify(cudaMalloc((void **)&matLdwEtaLimit_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matLdwAlpha_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matLdwBeta_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matLdwGamma_d, numberOfElements*sizeof(double)));
#endif
        //begin of ANEOS allocations in (global) device memory (everything linearized)
        cudaVerify(cudaMalloc((void **)&aneos_n_rho_d, numberOfElements*sizeof(int)));
        cudaVerify(cudaMalloc((void **)&aneos_n_e_d, numberOfElements*sizeof(int)));
        cudaVerify(cudaMalloc((void **)&aneos_bulk_cs_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&aneos_rho_d, run_aneos_rho_id*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&aneos_e_d, run_aneos_e_id*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&aneos_p_d, run_aneos_matrix_id*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&aneos_cs_d, run_aneos_matrix_id*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&aneos_rho_id_d, numberOfElements*sizeof(int)));
        cudaVerify(cudaMalloc((void **)&aneos_e_id_d, numberOfElements*sizeof(int)));
        cudaVerify(cudaMalloc((void **)&aneos_matrix_id_d, numberOfElements*sizeof(int)));
        //end of ANEOS allocations in (global) device memory
        cudaVerify(cudaMalloc((void **)&matSml_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matnoi_d, numberOfElements*sizeof(int)));
        cudaVerify(cudaMalloc((void **)&matEOS_d, numberOfElements*sizeof(int)));
        cudaVerify(cudaMalloc((void **)&matPolytropicK_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matPolytropicGamma_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matAlpha_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matBeta_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matBulkmodulus_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matShearmodulus_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matYieldStress_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matInternalFriction_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matInternalFrictionDamaged_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matRho0_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matIsothermalSoundSpeed_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matTillRho0_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matTillE0_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matTillEiv_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matTillEcv_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matTilla_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matTillb_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matTillA_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matTillB_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matTillAlpha_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matTillBeta_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matcsLimit_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matRhoLimit_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matN_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matCohesion_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matCohesionDamaged_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matFrictionAngle_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matFrictionAngleDamaged_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matAlphaPhi_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matCohesionCoefficient_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matMeltEnergy_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matDensityFloor_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matEnergyFloor_d, numberOfElements*sizeof(double)));
#if JC_PLASTICITY
        cudaVerify(cudaMalloc((void **)&matjc_y0_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matjc_B_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matjc_n_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matjc_m_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matjc_edot0_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matjc_C_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matjc_Tref_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matjc_Tmelt_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matCp_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMalloc((void **)&matCV_d, numberOfElements*sizeof(double)));
#endif
        cudaVerify(cudaMalloc((void **)&matdensity_via_kernel_sum_d, numberOfElements*sizeof(int)));
        cudaVerify(cudaMemcpy(matdensity_via_kernel_sum_d, density_via_kernel_sum, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
#if SOLID
        cudaVerify(cudaMalloc((void **)&matYoungModulus_d, numberOfElements*sizeof(double)));
        cudaVerify(cudaMemcpy(matYoungModulus_d, young_modulus, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpyToSymbol(matYoungModulus, &matYoungModulus_d, sizeof(void*)));
#endif

        cudaGetSymbolAddress((void **)&pc_pointer, gravConst);
        cudaMemcpy(pc_pointer, &grav_const, sizeof(double), cudaMemcpyHostToDevice);

        cudaGetSymbolAddress((void **)&pc_pointer, scale_height);
        cudaMemcpy(pc_pointer, &scale_height_host, sizeof(double), cudaMemcpyHostToDevice);

        /* predictor corrector integration parameters are set in symbol memory */
        cudaGetSymbolAddress((void **)&pc_pointer, Smin_d);
        cudaMemcpy(pc_pointer, &Smin, sizeof(double), cudaMemcpyHostToDevice);
        cudaGetSymbolAddress((void **)&pc_pointer, emin_d);
        cudaMemcpy(pc_pointer, &emin, sizeof(double), cudaMemcpyHostToDevice);
        cudaGetSymbolAddress((void **)&pc_pointer, rhomin_d);
        cudaMemcpy(pc_pointer, &rhomin, sizeof(double), cudaMemcpyHostToDevice);
        cudaGetSymbolAddress((void **)&pc_pointer, damagemin_d);
        cudaMemcpy(pc_pointer, &damagemin, sizeof(double), cudaMemcpyHostToDevice);
        cudaGetSymbolAddress((void **)&pc_pointer, alphamin_d);
        cudaMemcpy(pc_pointer, &alphamin, sizeof(double), cudaMemcpyHostToDevice);
        cudaGetSymbolAddress((void **)&pc_pointer, betamin_d);
        cudaMemcpy(pc_pointer, &betamin, sizeof(double), cudaMemcpyHostToDevice);
        cudaGetSymbolAddress((void **)&pc_pointer, alpha_epspormin_d);
        cudaMemcpy(pc_pointer, &alpha_epspormin, sizeof(double), cudaMemcpyHostToDevice);
        cudaGetSymbolAddress((void **)&pc_pointer, epsilon_vmin_d);
        cudaMemcpy(pc_pointer, &epsilon_vmin, sizeof(double), cudaMemcpyHostToDevice);

#if PALPHA_POROSITY
        cudaGetSymbolAddress((void **)&pc_pointer, max_abs_pressure_change);
        cudaMemcpy(pc_pointer, &max_abs_pressure_change_host, sizeof(double), cudaMemcpyHostToDevice);
        fprintf(stdout, "setting max allowed pressure change for time integration to %e\n", max_abs_pressure_change_host);
        if( !(max_abs_pressure_change_host > 0.0) ) {
            fprintf(stderr, "ERROR. Max allowed pressure change for time integration is set to %e (set as smallest 'porjutzi_p_elastic' found in material config).\n",
                    max_abs_pressure_change_host);
            fprintf(stderr, "Stopping here for now...\n");
            exit(1);
        }
#endif

        cudaVerify(cudaMemcpy(matSml_d, sml, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
#if VARIABLE_SML
        cudaVerify(cudaMemcpy(mat_f_sml_max_d, f_sml_max , numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(mat_f_sml_min_d, f_sml_min , numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
#endif
        cudaVerify(cudaMemcpy(matnoi_d, noi, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matEOS_d, eos, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matPolytropicK_d, polytropic_K, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matPolytropicGamma_d, polytropic_gamma, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
  	    cudaVerify(cudaMemcpy(matAlpha_d, alpha, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matBeta_d, beta, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matBulkmodulus_d, bulk_modulus, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
#if NAVIER_STOKES
        cudaVerify(cudaMemcpy(matnu_d, nu, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
  	    cudaVerify(cudaMemcpy(matalpha_shakura_d, alpha_shakura, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matzeta_d, zeta, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
#endif
        cudaVerify(cudaMemcpy(matShearmodulus_d, shear_modulus, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matYieldStress_d, yield_stress, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matInternalFriction_d, internal_friction, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matInternalFrictionDamaged_d, internal_friction_damaged, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matRho0_d, rho_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matIsothermalSoundSpeed_d, isothermal_cs, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matTillRho0_d, till_rho_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matTillE0_d, till_E_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matTillEcv_d, till_E_cv, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matTillEiv_d, till_E_iv, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matTilla_d, till_a, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matTillb_d, till_b, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matTillA_d, till_A, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matTillB_d, till_B, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matTillAlpha_d, till_alpha, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matTillBeta_d, till_beta, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matcsLimit_d, csLimit, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matRhoLimit_d, rho_limit, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matN_d, n, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matCohesion_d, cohesion, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matCohesionDamaged_d, cohesion_damaged, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matFrictionAngle_d, friction_angle, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matFrictionAngleDamaged_d, friction_angle_damaged, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matAlphaPhi_d, alpha_phi, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matCohesionCoefficient_d, cohesion_coefficient, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matMeltEnergy_d, melt_energy, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matDensityFloor_d, density_floor, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matEnergyFloor_d, energy_floor, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        //begin copying ANEOS data from host to (global) device memory
        cudaVerify(cudaMemcpy(aneos_n_rho_d, g_aneos_n_rho, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(aneos_n_e_d, g_aneos_n_e, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(aneos_bulk_cs_d, g_aneos_bulk_cs, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        for (i=0; i<numberOfMaterials; i++) {
            if (eos[i] == EOS_TYPE_ANEOS  ||  eos[i] == EOS_TYPE_JUTZI_ANEOS) {
                cudaVerify(cudaMemcpy(aneos_rho_d+aneos_rho_id[i], g_aneos_rho[i], g_aneos_n_rho[i]*sizeof(double), cudaMemcpyHostToDevice));
                cudaVerify(cudaMemcpy(aneos_e_d+aneos_e_id[i], g_aneos_e[i], g_aneos_n_e[i]*sizeof(double), cudaMemcpyHostToDevice));
                for(j=0; j<g_aneos_n_rho[i]; j++) {
                    cudaVerify(cudaMemcpy(aneos_p_d+aneos_matrix_id[i]+j*g_aneos_n_e[i], g_aneos_p[i][j], g_aneos_n_e[i]*sizeof(double), cudaMemcpyHostToDevice));
                    cudaVerify(cudaMemcpy(aneos_cs_d+aneos_matrix_id[i]+j*g_aneos_n_e[i], g_aneos_cs[i][j], g_aneos_n_e[i]*sizeof(double), cudaMemcpyHostToDevice));
                }
            }
        }
        cudaVerify(cudaMemcpy(aneos_rho_id_d, aneos_rho_id, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(aneos_e_id_d, aneos_e_id, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(aneos_matrix_id_d, aneos_matrix_id, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
        //end copying ANEOS data from host to (global) device memory, begin copying pointers to constant device memory
        /* the '_d' pointers are still in host memory and copied to (constant) device memory to be accessible from device code */
        cudaVerify(cudaMemcpyToSymbol(aneos_n_rho_c, &aneos_n_rho_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(aneos_n_e_c, &aneos_n_e_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(aneos_bulk_cs_c, &aneos_bulk_cs_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(aneos_rho_c, &aneos_rho_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(aneos_e_c, &aneos_e_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(aneos_p_c, &aneos_p_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(aneos_cs_c, &aneos_cs_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(aneos_rho_id_c, &aneos_rho_id_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(aneos_e_id_c, &aneos_e_id_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(aneos_matrix_id_c, &aneos_matrix_id_d, sizeof(void*)));
        //end copying pointers to constant device memory
#if JC_PLASTICITY
        cudaVerify(cudaMemcpy(matjc_y0_d, jc_y0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matjc_B_d, jc_B, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matjc_n_d, jc_n, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matjc_m_d, jc_m, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matjc_edot0_d, jc_edot0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matjc_C_d, jc_C, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matjc_Tref_d, jc_Tref, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matjc_Tmelt_d, jc_Tmelt, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matCp_d, Cp, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matCV_d, CV, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
#endif
#if NAVIER_STOKES
        cudaVerify(cudaMemcpyToSymbol(matnu, &matnu_d, sizeof(void*)));
  	    cudaVerify(cudaMemcpyToSymbol(matalpha_shakura, &matalpha_shakura_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matzeta, &matzeta_d, sizeof(void*)));
#endif
#if ARTIFICIAL_STRESS
        cudaVerify(cudaMemcpy(matexponent_tensor_d, exponent_tensor, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matepsilon_stress_d, epsilon_stress, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matmean_particle_distance_d, mean_particle_distance, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpyToSymbol(matexponent_tensor, &matexponent_tensor_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matepsilon_stress, &matepsilon_stress_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matmean_particle_distance, &matmean_particle_distance_d, sizeof(void*)));
#endif
#if LOW_DENSITY_WEAKENING
        cudaVerify(cudaMemcpy(matLdwEtaLimit_d, ldw_eta_limit, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matLdwAlpha_d, ldw_alpha, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matLdwBeta_d, ldw_beta, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matLdwGamma_d, ldw_gamma, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpyToSymbol(matLdwEtaLimit, &matLdwEtaLimit_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matLdwAlpha, &matLdwAlpha_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matLdwBeta, &matLdwBeta_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matLdwGamma, &matLdwGamma_d, sizeof(void*)));
#endif
#if PALPHA_POROSITY
        cudaVerify(cudaMemcpy(matporjutzi_p_elastic_d, porjutzi_p_elastic, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporjutzi_p_transition_d, porjutzi_p_transition, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporjutzi_p_compacted_d, porjutzi_p_compacted, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporjutzi_alpha_0_d, porjutzi_alpha_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporjutzi_alpha_e_d, porjutzi_alpha_e, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporjutzi_alpha_t_d, porjutzi_alpha_t, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporjutzi_n1_d, porjutzi_n1, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporjutzi_n2_d, porjutzi_n2, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matcs_porous_d, cs_porous, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matcs_solid_d, cs_solid, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matcrushcurve_style_d, crushcurve_style, numberOfElements*sizeof(int), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpyToSymbol(matporjutzi_p_elastic, &matporjutzi_p_elastic_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporjutzi_p_transition, &matporjutzi_p_transition_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporjutzi_p_compacted, &matporjutzi_p_compacted_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporjutzi_alpha_0, &matporjutzi_alpha_0_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporjutzi_alpha_e, &matporjutzi_alpha_e_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporjutzi_alpha_t, &matporjutzi_alpha_t_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporjutzi_n1, &matporjutzi_n1_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporjutzi_n2, &matporjutzi_n2_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matcs_porous, &matcs_porous_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matcs_solid, &matcs_solid_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matcrushcurve_style, &matcrushcurve_style_d, sizeof(void*)));
#endif
#if SIRONO_POROSITY
        cudaVerify(cudaMemcpy(matporsirono_K_0_d, porsirono_K_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporsirono_rho_0_d, porsirono_rho_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporsirono_rho_s_d, porsirono_rho_s, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporsirono_gamma_K_d, porsirono_gamma_K, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporsirono_alpha_d, porsirono_alpha, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporsirono_pm_d, porsirono_pm, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporsirono_phimax_d, porsirono_phimax, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporsirono_phi0_d, porsirono_phi0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporsirono_delta_d, porsirono_delta, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpyToSymbol(matporsirono_K_0, &matporsirono_K_0_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporsirono_rho_0, &matporsirono_rho_0_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporsirono_rho_s, &matporsirono_rho_s_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporsirono_gamma_K, &matporsirono_gamma_K_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporsirono_alpha, &matporsirono_alpha_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporsirono_pm, &matporsirono_pm_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporsirono_phimax, &matporsirono_phimax_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporsirono_phi0, &matporsirono_phi0_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporsirono_delta, &matporsirono_delta_d, sizeof(void*)));
#endif
#if EPSALPHA_POROSITY
        cudaVerify(cudaMemcpy(matporepsilon_kappa_d, porepsilon_kappa, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporepsilon_alpha_0_d, porepsilon_alpha_0, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporepsilon_epsilon_e_d, porepsilon_epsilon_e, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporepsilon_epsilon_x_d, porepsilon_epsilon_x, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpy(matporepsilon_epsilon_c_d, porepsilon_epsilon_c, numberOfElements*sizeof(double), cudaMemcpyHostToDevice));
        cudaVerify(cudaMemcpyToSymbol(matporepsilon_kappa, &matporepsilon_kappa_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporepsilon_alpha_0, &matporepsilon_alpha_0_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporepsilon_epsilon_e, &matporepsilon_epsilon_e_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporepsilon_epsilon_x, &matporepsilon_epsilon_x_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matporepsilon_epsilon_c, &matporepsilon_epsilon_c_d, sizeof(void*)));
#endif

        cudaVerify(cudaMemcpyToSymbol(matSml, &matSml_d, sizeof(void*)));
#if VARIABLE_SML
        cudaVerify(cudaMemcpyToSymbol(mat_f_sml_max, &mat_f_sml_max_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(mat_f_sml_min, &mat_f_sml_min_d, sizeof(void*)));
#endif
        cudaVerify(cudaMemcpyToSymbol(matnoi, &matnoi_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matdensity_via_kernel_sum, &matdensity_via_kernel_sum_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matEOS, &matEOS_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matPolytropicK, &matPolytropicK_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matPolytropicGamma, &matPolytropicGamma_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matAlpha, &matAlpha_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matBeta, &matBeta_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matBulkmodulus, &matBulkmodulus_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matShearmodulus, &matShearmodulus_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matYieldStress, &matYieldStress_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matInternalFriction, &matInternalFriction_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matInternalFrictionDamaged, &matInternalFrictionDamaged_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matRho0, &matRho0_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matIsothermalSoundSpeed, &matIsothermalSoundSpeed_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matTillRho0, &matTillRho0_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matTillE0, &matTillE0_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matTillEiv, &matTillEiv_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matTillEcv, &matTillEcv_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matTilla, &matTilla_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matTillb, &matTillb_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matTillA, &matTillA_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matTillB, &matTillB_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matTillAlpha, &matTillAlpha_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matTillBeta, &matTillBeta_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matcsLimit, &matcsLimit_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matRhoLimit, &matRhoLimit_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matN, &matN_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matCohesion, &matCohesion_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matCohesionDamaged, &matCohesionDamaged_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matFrictionAngle, &matFrictionAngle_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matFrictionAngleDamaged, &matFrictionAngleDamaged_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matAlphaPhi, &matAlphaPhi_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matCohesionCoefficient, &matCohesionCoefficient_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matMeltEnergy, &matMeltEnergy_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matDensityFloor, &matDensityFloor_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matEnergyFloor, &matEnergyFloor_d, sizeof(void*)));
#if JC_PLASTICITY
        cudaVerify(cudaMemcpyToSymbol(matjc_y0, &matjc_y0_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matjc_B, &matjc_B_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matjc_n, &matjc_n_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matjc_m, &matjc_m_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matjc_edot0, &matjc_edot0_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matjc_C, &matjc_C_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matjc_Tref, &matjc_Tref_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matjc_Tmelt, &matjc_Tmelt_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matCp, &matCp_d, sizeof(void*)));
        cudaVerify(cudaMemcpyToSymbol(matCV, &matCV_d, sizeof(void*)));
#endif

        fprintf(stdout, "\nUsing grav. constant: %e\n", grav_const);

        fprintf(stdout, "\nUsing following values for general parameters:\n");
        fprintf(stdout, "    material no.    smoothing length or number of interactions    density floor    energy floor\n");
        fprintf(stdout, "    ------------    ------------------------------------------    -------------    ------------\n");
        for (i = 0; i < numberOfMaterials; i++) {
            fprintf(stdout, "    %12d    %28e  or  %8d    %13g    %12g\n", i, sml[i], noi[i], density_floor[i], energy_floor[i]);
        }
#if VARIABLE_SML
        fprintf(stdout, "    material no.    factor for min and max smoothing length and corresponding smoothing lengths\n");
        fprintf(stdout, "    ------------    ---------------------------------------------------------------------------\n");
        for (i = 0; i < numberOfMaterials; i++) {
            fprintf(stdout, "    %12d    factor_min: %e -> minimum sml: %e    factor_max: %e -> maximun sml: %e\n",
                    i, f_sml_min[i], f_sml_min[i]*sml[i], f_sml_max[i], f_sml_max[i]*sml[i]);
        }
#endif
#if ARTIFICIAL_VISCOSITY
        fprintf(stdout, "\nUsing following values for artificial viscosity:\n");
        fprintf(stdout, "    material no.           alpha            beta\n");
        fprintf(stdout, "    ------------    ------------    ------------\n");
        for (i = 0; i < numberOfMaterials; i++) {
            fprintf(stdout, "    %12d    %12g    %12g\n", i, alpha[i], beta[i]);
        }
#endif
#if PLASTICITY
        fprintf(stdout, "\nUsing following values for the plasticity model:\n");
        fprintf(stdout, "    material no.    yield_stress        cohesion    cohesion_damaged    friction_angle    friction_angle_damaged    melt_energy\n");
        fprintf(stdout, "    ------------    ------------    ------------    ----------------    --------------    ----------------------    -----------\n");
        for (i = 0; i < numberOfMaterials; i++) {
            fprintf(stdout, "    %12d    %12g    %12g    %16g    %14g    %22g    %11g\n",
                    i, yield_stress[i], cohesion[i], cohesion_damaged[i], friction_angle[i], friction_angle_damaged[i], melt_energy[i]);
        }
#endif
#if LOW_DENSITY_WEAKENING
        fprintf(stdout, "\nUsing following values for low-density weakening:\n");
        fprintf(stdout, "    material no.     eta_limit         alpha          beta         gamma\n");
        fprintf(stdout, "    ------------    ----------    ----------    ----------    ----------\n");
        for (i = 0; i < numberOfMaterials; i++) {
            fprintf(stdout, "    %12d    %10g    %10g    %10g    %10g\n", i, ldw_eta_limit[i], ldw_alpha[i], ldw_beta[i], ldw_gamma[i]);
        }
#endif
        fprintf(stdout, "\nUsing following values for the equation of state:\n");
        fprintf(stdout, "    material no.                    EoS\n");
        fprintf(stdout, "    ------------    -------------------\n");
        char eos_type[255];
        for (i = 0; i < numberOfMaterials; i++) {
            fprintf(stdout, "    %12d    ", i);
            switch (eos[i]) {
                case (EOS_TYPE_MURNAGHAN):
                    strcpy(eos_type, "          Murnaghan");
                    fprintf(stdout, "%s\n", eos_type);
                    fprintf(stdout, "\t\t EOS params:\t K \t\t rho_0 \t\t n \t\t rho_limit\n");
                    fprintf(stdout, "\t\t\t\t %e \t %e \t %e \t %e\n", bulk_modulus[i], rho_0[i], n[i], rho_limit[i]);
                    break;
#if PALPHA_POROSITY
                case (EOS_TYPE_JUTZI_MURNAGHAN):
                    strcpy(eos_type, "Murnaghan with P-alpha model");
                    fprintf(stdout, "%s\n", eos_type);
                    fprintf(stdout, "\t\t EoS params: \t K \t\t rho_0 \t\t n \t\t rho_limit \t\t p_e \t\t p_t \t\t p_c \t\t alpha_0 \t alpha_e \t alpha_t \t n1 \t\t n2 \t\t cs_porous \t\t cs_solid \t\t crushcurve_style\n");
                    fprintf(stdout, "\t\t\t\t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %d\n", bulk_modulus[i], rho_0[i], n[i], rho_limit[i], porjutzi_p_elastic[i], porjutzi_p_transition[i], porjutzi_p_compacted[i], porjutzi_alpha_0[i], porjutzi_alpha_e[i], porjutzi_alpha_t[i], porjutzi_n1[i], porjutzi_n2[i], cs_porous[i], cs_solid[i], crushcurve_style[i]);
                    break;
#endif
                case (EOS_TYPE_TILLOTSON):
                    strcpy(eos_type, "          Tillotson");
                    fprintf(stdout, "%s\n", eos_type);
                    fprintf(stdout, "\t\t EoS params:\t till_rho_0 \t till_A \t till_B \t till_E_0 \t till_E_iv \t till_E_cv \t till_a \t till_b \t till_alpha \t till_beta \t cs_limit \t rho_limit \n");
                    fprintf(stdout, "\t\t\t\t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e\n", till_rho_0[i], till_A[i], till_B[i], till_E_0[i], till_E_iv[i], till_E_cv[i], till_a[i], till_b[i], till_alpha[i], till_beta[i], csLimit[i], rho_limit[i]);
                    break;
#if PALPHA_POROSITY
                case (EOS_TYPE_JUTZI):
                    strcpy(eos_type, "Tillotson with P-alpha model");
                    fprintf(stdout, "%s\n", eos_type);
                    fprintf(stdout, "\t\t EoS params:\n");
                    fprintf(stdout, "\t\t till_rho_0 \t %e \n", till_rho_0[i]);
                    fprintf(stdout, "\t\t till_A \t %e \n", till_A[i]);
                    fprintf(stdout, "\t\t till_B \t %e \n", till_B[i]);
                    fprintf(stdout, "\t\t till_E_0 \t %e \n", till_E_0[i]);
                    fprintf(stdout, "\t\t till_E_iv \t %e \n", till_E_iv[i]);
                    fprintf(stdout, "\t\t till_E_cv \t %e \n", till_E_cv[i]);
                    fprintf(stdout, "\t\t till_a \t %e \n", till_a[i]);
                    fprintf(stdout, "\t\t till_b \t %e \n", till_b[i]);
                    fprintf(stdout, "\t\t till_alpha \t %e \n", till_alpha[i]);
                    fprintf(stdout, "\t\t till_beta \t %e \n", till_beta[i]);
                    fprintf(stdout, "\t\t rho_limit \t %e \n", rho_limit[i]);
                    fprintf(stdout, "\t\t p_e \t\t %e \n", porjutzi_p_elastic[i]);
                    fprintf(stdout, "\t\t p_t \t\t %e \n", porjutzi_p_transition[i]);
                    fprintf(stdout, "\t\t p_c \t\t %e \n", porjutzi_p_compacted[i]);
                    fprintf(stdout, "\t\t alpha_0 \t %e \n", porjutzi_alpha_0[i]);
                    fprintf(stdout, "\t\t alpha_e \t %e \n", porjutzi_alpha_e[i]);
                    fprintf(stdout, "\t\t alpha_t \t %e \n", porjutzi_alpha_t[i]);
                    fprintf(stdout, "\t\t n1 \t\t %e \n", porjutzi_n1[i]);
                    fprintf(stdout, "\t\t n2 \t\t %e \n", porjutzi_n2[i]);
                    fprintf(stdout, "\t\t cs_porous \t %e \n", cs_porous[i]);
                    fprintf(stdout, "\t\t crushcurve_style \t %d \n", crushcurve_style[i]);
                    break;
                case (EOS_TYPE_JUTZI_ANEOS):
                    strcpy(eos_type, "ANEOS with P-alpha model");
                    fprintf(stdout, "%s\n", eos_type);
                    fprintf(stdout, "\t\t EoS params:\n");
                    fprintf(stdout, "\t\t till_rho_0 \t %e \n", till_rho_0[i]);
                    fprintf(stdout, "\t\t till_A \t %e \n", till_A[i]);
                    fprintf(stdout, "\t\t till_B \t %e \n", till_B[i]);
                    fprintf(stdout, "\t\t till_E_0 \t %e \n", till_E_0[i]);
                    fprintf(stdout, "\t\t till_E_iv \t %e \n", till_E_iv[i]);
                    fprintf(stdout, "\t\t till_E_cv \t %e \n", till_E_cv[i]);
                    fprintf(stdout, "\t\t till_a \t %e \n", till_a[i]);
                    fprintf(stdout, "\t\t till_b \t %e \n", till_b[i]);
                    fprintf(stdout, "\t\t till_alpha \t %e \n", till_alpha[i]);
                    fprintf(stdout, "\t\t till_beta \t %e \n", till_beta[i]);
                    fprintf(stdout, "\t\t rho_limit \t %e \n", rho_limit[i]);
                    fprintf(stdout, "\t\t p_e \t\t %e \n", porjutzi_p_elastic[i]);
                    fprintf(stdout, "\t\t p_t \t\t %e \n", porjutzi_p_transition[i]);
                    fprintf(stdout, "\t\t p_c \t\t %e \n", porjutzi_p_compacted[i]);
                    fprintf(stdout, "\t\t alpha_0 \t %e \n", porjutzi_alpha_0[i]);
                    fprintf(stdout, "\t\t alpha_e \t %e \n", porjutzi_alpha_e[i]);
                    fprintf(stdout, "\t\t alpha_t \t %e \n", porjutzi_alpha_t[i]);
                    fprintf(stdout, "\t\t n1 \t\t %e \n", porjutzi_n1[i]);
                    fprintf(stdout, "\t\t n2 \t\t %e \n", porjutzi_n2[i]);
                    fprintf(stdout, "\t\t cs_porous \t %e \n", cs_porous[i]);
                    fprintf(stdout, "\t\t crushcurve_style \t %d \n", crushcurve_style[i]);
                    break;
#endif
#if SIRONO_POROSITY
                case (EOS_TYPE_SIRONO):
                    strcpy(eos_type, "    Sirono Porosity");
                    fprintf(stdout, "%s\n", eos_type);
                    fprintf(stdout, "\t\t EoS params:\n");
                    fprintf(stdout, "\t\t K_0 \t %e \n", porsirono_K_0[i]);
                    fprintf(stdout, "\t\t rho_0 \t %e \n", porsirono_rho_0[i]);
                    fprintf(stdout, "\t\t rho_s \t %e \n", porsirono_rho_s[i]);
                    fprintf(stdout, "\t\t gamma_K %e \n", porsirono_gamma_K[i]);
                    fprintf(stdout, "\t\t alpha \t %e \n", porsirono_alpha[i]);
                    fprintf(stdout, "\t\t pm \t %e \n", porsirono_pm[i]);
                    fprintf(stdout, "\t\t phimax  %e \n", porsirono_phimax[i]);
                    fprintf(stdout, "\t\t phi0 \t %e \n", porsirono_phi0[i]);
                    fprintf(stdout, "\t\t delta \t %e \n", porsirono_delta[i]);
                    break;
#endif
#if EPSALPHA_POROSITY
                case (EOS_TYPE_EPSILON):
                    strcpy(eos_type, "Tillotson with Epsilon-Alpha Porosity");
                    fprintf(stdout, "%s\n", eos_type);
                    fprintf(stdout, "\t\t EoS params:\n");
                    fprintf(stdout, "\t\t kappa \t %e \n", porepsilon_kappa[i]);
                    fprintf(stdout, "\t\t alpha_0 \t %e \n", porepsilon_alpha_0[i]);
                    fprintf(stdout, "\t\t epsilon_e \t %e \n", porepsilon_epsilon_e[i]);
                    fprintf(stdout, "\t\t epsilon_x \t %e \n", porepsilon_epsilon_x[i]);
                    fprintf(stdout, "\t\t epsilon_c \t %e \n", porepsilon_epsilon_c[i]);
#endif
                case (EOS_TYPE_ANEOS):
                    strcpy(eos_type, "              ANEOS");
                    fprintf(stdout, "%s\n", eos_type);
                    break;
                case (EOS_TYPE_IDEAL_GAS):
                    strcpy(eos_type, "          Ideal gas");
                    fprintf(stdout, "%s\n", eos_type);
                    break;
                case (EOS_TYPE_REGOLITH):
                    strcpy(eos_type, "   Soil model (Bui)");
                    fprintf(stdout, "%s\n", eos_type);
                    break;
                case (EOS_TYPE_IGNORE):
                    strcpy(eos_type, "     None (ignored)");
                    fprintf(stdout, "%s\n", eos_type);
                    break;
                case (EOS_TYPE_ISOTHERMAL_GAS):
                    strcpy(eos_type, "     Isothermal gas");
                    fprintf(stdout, "%s\n", eos_type);
                    fprintf(stdout, "\t\t isothermal sound speed \t %e \n", isothermal_cs[i]);
                    break;
                case (EOS_TYPE_LOCALLY_ISOTHERMAL_GAS):
                    strcpy(eos_type, "Locally isothermal gas");
                    fprintf(stdout, "%s\n", eos_type);
                    break;
                case (EOS_TYPE_POLYTROPIC_GAS):
                    strcpy(eos_type, "     Polytropic gas");
                    fprintf(stdout, "%s\n", eos_type);
                    break;
                case (EOS_TYPE_VISCOUS_REGOLITH):
                    strcpy(eos_type, "Viscous regolith (experimental)");
                    fprintf(stdout, "%s\n", eos_type);
                    break;
                default:
                    fprintf(stderr, "Error: Cannot determine rho0 for material ID %d with EOS_TYPE %d\n", i, eos[i]);
                    exit(1);
            }
        }

#if SOLID
        fprintf(stdout, "\nUsing following values for elastic constants:\n");
        fprintf(stdout, "    material no.    bulk modulus    shear modulus        Poisson's ratio    Young's modulus\n");
        fprintf(stdout, "    ------------    ------------    -------------        ---------------    ---------------\n");
        for (i = 0; i < numberOfMaterials; i++) {
            double poisson_ratio = (3.0*bulk_modulus[i] - 2.0*shear_modulus[i]) / 2.0 / (3.0*bulk_modulus[i] + shear_modulus[i]);
            fprintf(stdout, "    %12d    %12g    %13g        %15g    %15g\n", i, bulk_modulus[i], shear_modulus[i], poisson_ratio, young_modulus[i]);
        }
#endif

#if NAVIER_STOKES
        fprintf(stdout, "\nUsing following values for the physical viscosity:\n");
        fprintf(stdout, "    material no.            alpha coef                dynamic                bulk\n");
        fprintf(stdout, "    ------------    ------------------    -------------------    ----------------\n");
        for (i = 0; i < numberOfMaterials; i++) {
        	fprintf(stdout, "    %12d    %18e    %19e    %16e\n", i, alpha_shakura[i], eta[i], zeta[i]);
       }
#endif

#if GRAVITATING_POINT_MASSES
        fprintf(stdout, "\nFound %d pointmasses in pointmasses input file:\n", numberOfPointmasses);
        for (i = 0; i < numberOfPointmasses; i++) {
            fprintf(stdout, "    no.: %d", i);
            fprintf(stdout, "    x: %e", pointmass_host.x[i]);
#if DIM > 1
            fprintf(stdout, "  y: %e", pointmass_host.y[i]);
#if DIM > 2
            fprintf(stdout, "  z: %e", pointmass_host.z[i]);
#endif
#endif
            fprintf(stdout, "    vx: %e", pointmass_host.vx[i]);
#if DIM > 1
            fprintf(stdout, "  vy: %e", pointmass_host.vy[i]);
#if DIM > 2
            fprintf(stdout, "  vz: %e", pointmass_host.vz[i]);
#endif
#endif
            fprintf(stdout, "    mass: %e", pointmass_host.m[i]);
            fprintf(stdout, "    particles get no closer than: %e", pointmass_host.rmin[i]);
            fprintf(stdout, "  no farther than: %e", pointmass_host.rmax[i]);
            fprintf(stdout, "\n");
        }
#endif

#if ARTIFICIAL_STRESS
        fprintf(stdout, "\nUsing following parameters for artificial stress:\n");
        fprintf(stdout, "    material no.    exponent tensor    epsilon stress    mean particle distance\n");
        fprintf(stdout, "    ------------    ---------------    --------------    ----------------------\n");
        for (i = 0; i < numberOfMaterials; i++) {
            fprintf(stdout, "    %12d    %15g    %14g    %22e\n", i, exponent_tensor[i], epsilon_stress[i], mean_particle_distance[i]);
        }
#endif


#if ARTIFICIAL_STRESS
        free(exponent_tensor);
        free(epsilon_stress);
        free(mean_particle_distance);
#endif
        free(alpha);
        free(beta);
        free(nu);
        free(eta);
        free(zeta);
#if VARIABLE_SML
        free(f_sml_max);
        free(f_sml_min);
#endif
        // begin freeing some ANEOS memory on the host
        free(aneos_rho_id);
        free(aneos_e_id);
        free(aneos_matrix_id);
        //end freeing some ANEOS memory on the host
        free(noi);
        free(eos);
        free(alpha_shakura);
        free(polytropic_K);
        free(polytropic_gamma);
        free(n);
        free(rho_0);
        free(isothermal_cs);
        free(rho_limit);
        free(till_A);
        free(till_B);
        free(till_E_0);
        free(till_E_iv);
        free(till_a);
        free(till_b);
        free(till_alpha);
        free(till_beta);
        free(csLimit);
        free(shear_modulus);
        free(yield_stress);
        free(cohesion);
        free(cohesion_damaged);
        free(cohesion_coefficient);
        free(melt_energy);
        free(friction_angle);
        free(friction_angle_damaged);
        free(density_floor);
        free(energy_floor);
#if SOLID
        free(young_modulus);
#endif
#if LOW_DENSITY_WEAKENING
        free(ldw_eta_limit);
        free(ldw_alpha);
        free(ldw_beta);
        free(ldw_gamma);
#endif
#if PALPHA_POROSITY
        free(porjutzi_p_elastic);
        free(porjutzi_p_transition);
        free(porjutzi_p_compacted);
        free(porjutzi_alpha_0);
        free(porjutzi_alpha_e);
        free(porjutzi_alpha_t);
        free(porjutzi_n1);
        free(porjutzi_n2);
        free(cs_solid);
        free(crushcurve_style);
#endif
#if SIRONO_POROSITY
        free(porsirono_K_0);
        free(porsirono_rho_0);
        free(porsirono_rho_s);
        free(porsirono_gamma_K);
        free(porsirono_alpha);
        free(porsirono_pm);
        free(porsirono_phimax);
        free(porsirono_phi0);
        free(porsirono_delta);
#endif
#if EPSALPHA_POROSITY
        free(porepsilon_kappa);
        free(porepsilon_alpha_0);
        free(porepsilon_epsilon_e);
        free(porepsilon_epsilon_x);
        free(porepsilon_epsilon_c);
#endif
#if JC_PLASTICITY
        free(jc_y0);
        free(jc_B);
        free(jc_n);
        free(jc_m);
        free(jc_edot0);
        free(jc_C);
        free(jc_Tref);
        free(jc_Tmelt);
        free(Cp);
        free(CV);
#endif
    }
}



void cleanupMaterials()
{
    //begin freeing of ANEOS (global) device memory
    cudaVerify(cudaFree(aneos_n_rho_d));
    cudaVerify(cudaFree(aneos_n_e_d));
    cudaVerify(cudaFree(aneos_bulk_cs_d));
    cudaVerify(cudaFree(aneos_rho_d));
    cudaVerify(cudaFree(aneos_e_d));
    cudaVerify(cudaFree(aneos_p_d));
    cudaVerify(cudaFree(aneos_cs_d));
    cudaVerify(cudaFree(aneos_rho_id_d));
    cudaVerify(cudaFree(aneos_e_id_d));
    cudaVerify(cudaFree(aneos_matrix_id_d));
    //end freeing of ANEOS (global) device memory
    cudaVerify(cudaFree(matSml_d));
    cudaVerify(cudaFree(matInternalFriction_d));
    cudaVerify(cudaFree(matInternalFrictionDamaged_d));
    cudaVerify(cudaFree(matYieldStress_d));
    cudaVerify(cudaFree(matnoi_d));
    cudaVerify(cudaFree(matEOS_d));
    cudaVerify(cudaFree(matPolytropicGamma_d));
    cudaVerify(cudaFree(matPolytropicK_d));
    cudaVerify(cudaFree(matAlpha_d));
    cudaVerify(cudaFree(matAlphaPhi_d));
    cudaVerify(cudaFree(matBeta_d));
    cudaVerify(cudaFree(matBulkmodulus_d));
    cudaVerify(cudaFree(matYoungModulus_d));
#if VARIABLE_SML
    cudaVerify(cudaFree(mat_f_sml_max_d));
    cudaVerify(cudaFree(mat_f_sml_min_d));
#endif
    cudaVerify(cudaFree(matRho0_d));
    cudaVerify(cudaFree(matTillRho0_d));
    cudaVerify(cudaFree(matTilla_d));
    cudaVerify(cudaFree(matTillA_d));
    cudaVerify(cudaFree(matTillb_d));
    cudaVerify(cudaFree(matTillB_d));
    cudaVerify(cudaFree(matTillAlpha_d));
    cudaVerify(cudaFree(matTillBeta_d));
    cudaVerify(cudaFree(matcsLimit_d));
    cudaVerify(cudaFree(matTillE0_d));
    cudaVerify(cudaFree(matTillEcv_d));
    cudaVerify(cudaFree(matTillEiv_d));
    cudaVerify(cudaFree(matRhoLimit_d));
    cudaVerify(cudaFree(matShearmodulus_d));
    cudaVerify(cudaFree(matN_d));
    cudaVerify(cudaFree(matCohesion_d));
    cudaVerify(cudaFree(matCohesionDamaged_d));
    cudaVerify(cudaFree(matCohesionCoefficient_d));
    cudaVerify(cudaFree(matMeltEnergy_d));
    cudaVerify(cudaFree(matFrictionAngle_d));
    cudaVerify(cudaFree(matFrictionAngleDamaged_d));
    cudaVerify(cudaFree(matDensityFloor_d));
    cudaVerify(cudaFree(matEnergyFloor_d));
    cudaVerify(cudaFree(matLdwEtaLimit_d));
    cudaVerify(cudaFree(matLdwAlpha_d));
    cudaVerify(cudaFree(matLdwBeta_d));
    cudaVerify(cudaFree(matLdwGamma_d));
    cudaVerify(cudaFree(matporjutzi_p_elastic_d));
    cudaVerify(cudaFree(matporjutzi_p_transition_d));
    cudaVerify(cudaFree(matporjutzi_p_compacted_d));
    cudaVerify(cudaFree(matporjutzi_alpha_0_d));
    cudaVerify(cudaFree(matporjutzi_alpha_e_d));
    cudaVerify(cudaFree(matporjutzi_alpha_t_d));
    cudaVerify(cudaFree(matporjutzi_n1_d));
    cudaVerify(cudaFree(matporjutzi_n2_d));
    cudaVerify(cudaFree(matcs_porous_d));
    cudaVerify(cudaFree(matcs_solid_d));
    cudaVerify(cudaFree(matcrushcurve_style_d));
    cudaVerify(cudaFree(matporsirono_K_0_d));
    cudaVerify(cudaFree(matporsirono_rho_0_d));
    cudaVerify(cudaFree(matporsirono_rho_s_d));
    cudaVerify(cudaFree(matporsirono_gamma_K_d));
    cudaVerify(cudaFree(matporsirono_alpha_d));
    cudaVerify(cudaFree(matporsirono_pm_d));
    cudaVerify(cudaFree(matporsirono_phimax_d));
    cudaVerify(cudaFree(matporsirono_phi0_d));
    cudaVerify(cudaFree(matporsirono_delta_d));
    cudaVerify(cudaFree(matporepsilon_kappa_d));
    cudaVerify(cudaFree(matporepsilon_alpha_0_d));
    cudaVerify(cudaFree(matporepsilon_epsilon_e_d));
    cudaVerify(cudaFree(matporepsilon_epsilon_x_d));
    cudaVerify(cudaFree(matporepsilon_epsilon_c_d));
    cudaVerify(cudaFree(matjc_y0_d));
    cudaVerify(cudaFree(matjc_B_d));
    cudaVerify(cudaFree(matjc_n_d));
    cudaVerify(cudaFree(matjc_m_d));
    cudaVerify(cudaFree(matjc_edot0_d));
    cudaVerify(cudaFree(matjc_C_d));
    cudaVerify(cudaFree(matjc_Tref_d));
    cudaVerify(cudaFree(matjc_Tmelt_d));
    cudaVerify(cudaFree(matCp_d));
    cudaVerify(cudaFree(matCV_d));
    free(sml);
    free(bulk_modulus);
    free(cs_porous);
    free(till_rho_0);
}
