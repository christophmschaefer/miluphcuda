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


#include "plasticity.h"
#include "config_parameter.h"
#include "parameter.h"
#include "miluph.h"
#include "pressure.h"
#include "float.h"


#if PURE_REGOLITH
__global__ void plasticity()
{
    register int i, inc, matId, d, e;
    register double alpha_phi, kc, I1, sqrt_J2, rn;
#if DIM == 2
    register double shear, bulk, poissons_ratio, sz;
#endif
    register double S_i[DIM][DIM];

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];
        if (EOS_TYPE_REGOLITH == matEOS[matId]) {

            alpha_phi = matAlphaPhi[matId];
            kc = matCohesionCoefficient[matId];

#if DIM == 2
            shear = matShearmodulus[matId];
            bulk = matBulkmodulus[matId];
            poissons_ratio = (3*bulk - 2*shear) / (2*(3*bulk + shear));
            I1 = (1 + poissons_ratio) * (p.S[stressIndex(i, 0, 0)] + p.S[stressIndex(i, 1, 1)]);
#else
            I1 = p.S[stressIndex(i,0,0)] + p.S[stressIndex(i,1,1)] + p.S[stressIndex(i,2,2)];
#endif

            //Tension cracking treatment
            //Equation 29, Bui et al., 2008
            if (-I1*alpha_phi + kc < 0) {
                for (d = 0; d < DIM; d++) {
                    p.S[stressIndex(i, d, d)] -= (I1 - kc/alpha_phi)/3.0;
                }
            }

#if DIM == 2
            I1 = (1 + poissons_ratio) * (p.S[stressIndex(i, 0, 0)] + p.S[stressIndex(i, 1, 1)]);
#else
            I1 = p.S[stressIndex(i,0,0)] + p.S[stressIndex(i,1,1)] + p.S[stressIndex(i,2,2)];
#endif

            //get S
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    S_i[d][e] = p.S[stressIndex(i, d, e)];
                }
                S_i[d][d] -= I1/3.0;
            }
#if DIM == 2
            sz = poissons_ratio*(S_i[0][0] + S_i[1][1]);
#endif

            sqrt_J2 = 0.0;
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    sqrt_J2 += S_i[d][e]*S_i[d][e];
                }
            }
#if DIM == 2
            sqrt_J2 += sz*sz;
#endif
            sqrt_J2 *= 0.5;
            sqrt_J2 = sqrt(sqrt_J2);

            //stress-scaling
            //Equation 31, Bui et al., 2008
            if (sqrt_J2 > 0) {
                rn = (-I1*alpha_phi + kc) / sqrt_J2;
                rn = min(rn, 1.0);
                for (d = 0; d < DIM; d++) {
                    for (e = 0; e < DIM; e++) {
                        p.S[stressIndex(i, d, e)] = rn*S_i[d][e];
                    }
                    p.S[stressIndex(i, d, d)] += I1/3.0;
                }
            }
        } //end if (EOS_TYPE_REGOLITH)
    }
}
#endif



#if PLASTICITY
__global__ void plasticityModel(void) {
    register int i, inc, d, e;
    register double mises_f, tmp;
    register double I1, J2;
    register double y, y_i, y_d, y_M, y_0, y_0_d, damage, e_melt;
    register double A, B;   // Drucker-Prager constants
    register double mu_i, mu_d;  // coefficients of internal friction
    register int matId;
#if LOW_DENSITY_WEAKENING
    register double rho0, eta, ldw_f, ldw_eta_limit, ldw_alpha, ldw_beta, ldw_gamma;
#endif

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

        // VISCOUS_REGOLITH is treated in timeintegration.cu when \sigma is calculated
        if (matEOS[p_rhs.materialId[i]] == EOS_TYPE_VISCOUS_REGOLITH) {
            continue;
        }

        mises_f = 1.0;

        // compute second invariant of the deviator stress tensor
        J2 = 0.0;
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                tmp = p.S[stressIndex(i, d, e)];
                J2 += tmp*tmp;
            }
        }
        J2 *= 0.5;

        // compute first invariant of the stress tensor
        I1 = -3.0 * p.p[i];

#if LOW_DENSITY_WEAKENING   // reduce strength by reducing the cohesion for low densities
        matId = p_rhs.materialId[i];
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
        // compute weakening factor
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
        if( ldw_f > 1.0  ||  ldw_f < 0.0 ) {
            printf("ERROR. Found low-density weakening factor outside [0,1], with ldw_f = %e...\n", ldw_f);
        }
#endif

#if MOHR_COULOMB_PLASTICITY
        // Mohr-Coulomb yield criterion
        // matInternalFriction = \mu = tan(matFrictionAngle)
        y_0 = matCohesion[p_rhs.materialId[i]];
# if LOW_DENSITY_WEAKENING
        y_0 *= ldw_f;   // reduce cohesion (locally)
# endif

        // follow slope set by friction coefficient for p > 0, and slope = 1 for p < 0 (i.e., zero at -y_0)
        if( p.p[i] > 0.0 ) {
            y = y_0 + matInternalFriction[p_rhs.materialId[i]] * p.p[i];
        } else if( p.p[i] > -y_0 ) {
            y = y_0 + p.p[i];
        } else {
            y = 0.0;
        }

        // additional von Mises limit if set
# if VON_MISES_PLASTICITY
        y = min(y, matYieldStress[p_rhs.materialId[i]]);
# endif
        if (y < 0.0) y = 0.0;

        // negative-pressure cap: limit negative pressures to value at zero of yield strength curve (at -cohesion)
        if( p.p[i] < -y_0 )
            p.p[i] = -y_0;

        // Drucker-Prager-like -> compare to sqrt(J2)
        if (J2 > 0.0) {
            mises_f = y / sqrt(J2);
        }

#elif DRUCKER_PRAGER_PLASTICITY
        A = B = 0.0;
        y_0 = matCohesion[p_rhs.materialId[i]];
# if LOW_DENSITY_WEAKENING
        y_0 *= ldw_f;   // reduce cohesion (locally)
# endif
        // Drucker-Prager constants from Mohr-Coulomb constants -> 3D!
        A = 6. * y_0 * cos(matFrictionAngle[p_rhs.materialId[i]])
                / (sqrt(3.) * (3. - sin(matFrictionAngle[p_rhs.materialId[i]])));
        B = 2. * sin(matFrictionAngle[p_rhs.materialId[i]]) / (sqrt(3.) * (3. - sin(matFrictionAngle[p_rhs.materialId[i]])));

        // yield strength determined by Drucker-Prager condition
        y = A + 3.0*p.p[i]*B;

        // additional von Mises limit if set
# if VON_MISES_PLASTICITY
        y = min(y, matYieldStress[p_rhs.materialId[i]]);
# endif
        if (y < 0.0) y = 0.0;

        // Drucker-Prager-like -> compare to sqrt(J2)
        if (J2 > 0.0) {
            mises_f = y / sqrt(J2);
        }

#elif COLLINS_PLASTICITY
        y_0 = matCohesion[p_rhs.materialId[i]];
        y_M = matYieldStress[p_rhs.materialId[i]];
        mu_i = matInternalFriction[p_rhs.materialId[i]];

        // yield strength of intact material, with constant cohesion for p < 0
        y_i = y_0;
        if (p.p[i] > 0.0) {
            y_i += mu_i * p.p[i]
                / (1 + mu_i * p.p[i]  / (y_M - y_0) );
        }
# if COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY
        e_melt = matMeltEnergy[p_rhs.materialId[i]];

        if (p.e[i] >= e_melt) {
            y_i = 0.0;
        } else if (p.e[i] > 0.0) {
            y_i *= ( 1.0 - p.e[i] / e_melt );
        }
# endif
# if FRAGMENTATION
        y_0_d = matCohesionDamaged[p_rhs.materialId[i]];
#  if LOW_DENSITY_WEAKENING
        // reduce damaged cohesion (locally)
        // intact cohesion is not affected by low-density weakening
        y_0_d *= ldw_f;
#  endif
        mu_d = matInternalFrictionDamaged[p_rhs.materialId[i]];
        damage = p.damage_total[i];
        if (damage > 1.0) damage = 1.0;
        if (damage < 0.0) damage = 0.0;

        // yield strength of damaged material, going to zero with slope = 1 for p < 0 (i.e., the zero is at -y_0_d)
        if( p.p[i] > 0.0 ) {
            y_d = y_0_d + mu_d * p.p[i];
        } else if( p.p[i] > -y_0_d ) {
            y_d = y_0_d + p.p[i];
        } else {
            y_d = 0.0;
        }

        if (y_d < 0.0) y_d = 0.0;

        // the actual yield strength y is a weighted mean of y_i and y_d
        // note: therefore potential melt-energy effects are also included in y
        y = (1.0-damage) * y_i + damage * y_d;

        // always limit the yield strength to the intact value
        if (y > y_i)
            y = y_i;

        // here we apply a "cap on negative pressure release by damage"
        // negative pressure is foremost released by (1-damage), in line with the Grady-Kipp model, but only
        // up to the residual tensile strength the material retains even when fully damaged, assumed to be -y_0_d
        // note: modification by (1-damage) must be done only once (here), otherwise it would be cumulative
        if( p.p[i] < -y_0_d ) {
            if( (1.0-damage)*p.p[i] > -y_0_d ) {
                p.p[i] = -y_0_d;
            } else {
                p.p[i] = (1.0-damage)*p.p[i];
            }
        }
# else
        y = y_i;
# endif   // FRAGMENTATION

        // Drucker-Prager-like -> compare to sqrt(J2)
        if (J2 > 0.0) {
            mises_f = y / sqrt(J2);
        }

#elif COLLINS_PLASTICITY_SIMPLE
        y_0 = matCohesion[p_rhs.materialId[i]];
        y_M = matYieldStress[p_rhs.materialId[i]];
        mu_i = matInternalFriction[p_rhs.materialId[i]];

# if LOW_DENSITY_WEAKENING
        y_0 *= ldw_f;   // reduce cohesion (locally)
# endif

        // Lundborg yield strength curve for p > 0
        // linear decrease to zero with slope = 1 for p < 0 (i.e., zero at -y_0)
        if( p.p[i] > 0.0 ) {
            y = y_0 + mu_i * p.p[i]
                / (1.0 + mu_i * p.p[i]  / (y_M - y_0) );
        } else if( p.p[i] > -y_0 ) {
            y = y_0 + p.p[i];
        } else {
            y = 0.0;
        }

        // let the yield strength decrease to zero for p < 0 following the regular y_i curve,
        // where the zero is at p_0 = -y_0 (y_M-y_0) / (mu_i y_M)
//        if ( p.p[i] > y_0*(y_0-y_M)/(mu_i*y_M) ) {
//            y = y_0 + mu_i * p.p[i]
//                / (1.0 + mu_i * p.p[i]  / (y_M - y_0) );
//        } else {
//            y = 0.0;
//        }

        // negative-pressure cap: limit negative pressures to value at zero of yield strength curve (at -cohesion)
        if( p.p[i] < -y_0 )
            p.p[i] = -y_0;

        // Drucker-Prager-like -> compare to sqrt(J2)
        if (J2 > 0.0) {
            mises_f = y / sqrt(J2);
        }

#elif VON_MISES_PLASTICITY
        y = matYieldStress[p_rhs.materialId[i]];
# if SIRONO_POROSITY
        // shear strength using the Sirono model
        if (matEOS[p_rhs.materialId[i]] == EOS_TYPE_SIRONO) {
            y = sqrt((-1.0) * p.tensile_strength[i] * p.compressive_strength[i]);
            p.shear_strength[i] = y;
        } else {
            p.shear_strength[i] = DBL_MAX;
            y = p.shear_strength[i];
        }
# endif
# if LOW_DENSITY_WEAKENING
        // reduce whole yield/shear strength (locally)
        y *= ldw_f;
# endif
        // von Mises limit like
        if (J2 > 0.0) {
            mises_f = y*y/(3.0*J2);
        }
#endif  // plasticity models

        // finally limit the deviatoric stress tensor
        if (mises_f > 1.0)
            mises_f = 1.0;
        if (mises_f < 0) // actually, this should never happen
            mises_f = 0.0;

        // remember the plastic lowering factor for later usage
        p_rhs.plastic_f[i] = mises_f;
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                p.S[stressIndex(i, d, e)] *= mises_f;
            }
        }
    }
}
#endif



#if JC_PLASTICITY
__global__ void JohnsonCookPlasticity(void) {
    // introduce plastic behaviour by limiting the deviatoric stress
    register int i, inc, d, e;
    register double J2, jc_f, y_0, tmp;
    register double y_jc = 0;
    register double T_star = 0;
    register double B, n, m, edot0, C, Tref, Tmelt;
    /*register double Cp, CV;*/

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

        J2 = 0;
        jc_f = 0;
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                tmp = p.S[stressIndex(i, d, e)];
                J2 += tmp*tmp;
            }
        }

        y_0 = matjc_y0[p_rhs.materialId[i]];
        B = matjc_B[p_rhs.materialId[i]];
        n = matjc_n[p_rhs.materialId[i]];
        m = matjc_m[p_rhs.materialId[i]];
        edot0 = matjc_edot0[p_rhs.materialId[i]];
        C = matjc_C[p_rhs.materialId[i]];
        Tref = matjc_Tref[p_rhs.materialId[i]];
        Tmelt = matjc_Tmelt[p_rhs.materialId[i]];
        /*Cp = matCp[p_rhs.materialId[i]];*/
        /*CV = matCV[p_rhs.materialId[i]];*/

        register double edotp = p.edotp[i];
        register double ep = p.ep[i];
        register double T = p.T[i];

        // T_star has to be different for different cases, otherwise we have complex numbers and nans
        if (T < Tref) {
            T_star = 0;
        } else if (T > Tmelt) {
            T_star = 1;
        } else {
            T_star = (T - Tref) / (Tmelt - Tref);
        }

        // Calculating flow stress according to Johnson and Cook
        if (edotp > 0) {
            y_jc = (y_0 + B*(pow(ep,n))) * (1 + C*log(edotp / edot0)) * (1 - pow(T_star,m));
        } else {
            y_jc = y_0;
        }

        y_jc = y_jc * y_jc;
        J2 = J2 * 1.5;
        if (J2 > y_jc)
            jc_f = y_jc/J2;
        else
            jc_f = 1;
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                p.S[stressIndex(i, d, e)] *= jc_f;
            }
        }

        /* remember for calculation of edotp later on */
        p.jc_f[i] = jc_f;
        p.edotp[i] = 0.0;
    }
}
#endif
