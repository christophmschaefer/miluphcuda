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
    // introduce plastic behaviour by limiting the deviatoric stress
    register int i, inc, d, e;
    register double mises_f, tmp;
    register double I1, J2, sqrt_J2;
    register double y, y_i, y_d, y_M, y_0, y_0_d, damage, e_melt;
    /* drucker prager constants */
    register double A, B;
    double mu_i, mu_d; // coefficients of internal friction

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

        // VISCOUS_REGOLITH is treated in timeintegration.cu when \sigma is calculated
        if (matEOS[p_rhs.materialId[i]] == EOS_TYPE_VISCOUS_REGOLITH) {
            continue;
        }

        mises_f = 1.0;

        /* second invariant of the deviator stress tensor */
        J2 = 0.0;
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                tmp = p.S[stressIndex(i, d, e)];
                J2 += tmp*tmp;
            }
        }
        J2 *= 0.5;
        sqrt_J2 = sqrt(J2);

        /* first invariant of the stress tensor */
        I1 = -3.0 * p.p[i];

#if MOHR_COULOMB_PLASTICITY
        // Mohr-Coulomb yield criterion
        // matInternalFriction = \mu = tan(matFrictionAngle)
        y = matCohesion[p_rhs.materialId[i]] + matInternalFriction[p_rhs.materialId[i]] * p.p[i];
        
        // additional von Mises limit if set
# if VON_MISES_PLASTICITY
        y = min(y, matYieldStress[p_rhs.materialId[i]]);
# endif
        if (y < 0.0) {
            y = 0.0;
        }

        // Drucker-Prager-like -> compare to sqrt(J2)
        if (J2 > 0.0) {
            mises_f = y/sqrt_J2;
        }
#elif DRUCKER_PRAGER_PLASTICITY
        A = B = 0;
        // Drucker-Prager constants from Mohr-Coulomb constants -> 3D!
        A = 6. * matCohesion[p_rhs.materialId[i]] * cos(matFrictionAngle[p_rhs.materialId[i]])
                / (sqrt(3.) * (3. - sin(matFrictionAngle[p_rhs.materialId[i]])));
        B = 2. * sin(matFrictionAngle[p_rhs.materialId[i]]) / (sqrt(3.) * (3. - sin(matFrictionAngle[p_rhs.materialId[i]])));

        // yield strength determined by Drucker-Prager condition
        y = A + 3.0*p.p[i]*B;

        // additional von Mises limit if set
# if VON_MISES_PLASTICITY
        y = min(y, matYieldStress[p_rhs.materialId[i]]);
# endif
        if (y < 0.0) {
            y = 0.0;
        }

        // Drucker-Prager-like -> compare to sqrt(J2)
        if (J2 > 0.0) {
            mises_f = y/sqrt_J2;
        }
#elif COLLINS_PLASTICITY
        y_0 = matCohesion[p_rhs.materialId[i]];
        y_M = matYieldStress[p_rhs.materialId[i]];
        mu_i = matInternalFriction[p_rhs.materialId[i]];

        // yield strength of intact material, with constant cohesion for p<0
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
        mu_d = matInternalFrictionDamaged[p_rhs.materialId[i]];
        damage = p.damage_total[i];
        if (damage > 1.0) damage = 1.0;
        if (damage < 0.0) damage = 0.0;
        
        // yield strength of damaged material, with the cohesion going (linearly) to zero for p<0
        y_d = y_0_d + mu_d * p.p[i];
        if (y_d < 0.0)
            y_d = 0.0;
        
        // the actual yield strength Y is a weighted mean of Y_i and Y_d
        // note: therefore potential melt-energy effects are also included in Y
        y = (1.0-damage) * y_i + damage * y_d;
        
        // always limit the yield strength to the intact value
        if (y > y_i)
            y = y_i;
# else
        y = y_i;
# endif
        // Drucker-Prager-like -> compare to sqrt(J2)
        if (J2 > 0.0) {
            mises_f = y/sqrt_J2;
        }
#elif COLLINS_PLASTICITY_SIMPLE
        y_0 = matCohesion[p_rhs.materialId[i]];
        y_M = matYieldStress[p_rhs.materialId[i]];
        mu_i = matInternalFriction[p_rhs.materialId[i]];

        // unlike for the regular Collins model, here we let the yield strength decrease to zero for p<0,
        // following a linear decline for p<0, where the slope is mu_i, and the zero at -y_0/mu_i
        if( p.p[i] > 0.0 ) {
            y = y_0 + mu_i * p.p[i]
                / (1.0 + mu_i * p.p[i]  / (y_M - y_0) );
        } else if( p.p[i] > -y_0 / mu_i ) {
            y = y_0 + p.p[i] * mu_i;
        } else {
            y = 0.0;
        }

        // let the yield strength decrease to zero for p<0 following the regular Y_i curve,
        // where the zero is at p_0 = -Y_0 (Y_M-Y_0) / (mu_i Y_M)
//        if ( p.p[i] > y_0*(y_0-y_M)/(mu_i*y_M) ) {
//            y = y_0 + mu_i * p.p[i]
//                / (1.0 + mu_i * p.p[i]  / (y_M - y_0) );
//        } else {
//            y = 0.0;
//        }

        // also limit negative pressures to value at zero of yield strength curve
        // (zero is at -y_0/mu_i if assumed linear for p<0)
        if( p.p[i] < -y_0 / mu_i)
            p.p[i] = -y_0 / mu_i;

        // Drucker-Prager-like -> compare to sqrt(J2)
        if (J2 > 0.0) {
            mises_f = y/sqrt_J2;
        }
#else // simple von Mises yield criterion without *any* dependency
        y = matYieldStress[p_rhs.materialId[i]];
# if SIRONO_POROSITY
        // Shear Strength using Sironos Model
        if (matEOS[p_rhs.materialId[i]] == EOS_TYPE_SIRONO) {
            y = sqrt((-1.0) * p.tensile_strength[i] * p.compressive_strength[i]);
            p.shear_strength[i] = y;
        } else {
            p.shear_strength[i] = DBL_MAX;
            y = p.shear_strength[i];
        }
# endif
        // von Mises limit like
        if (J2 > 0.0) {
            mises_f = y*y/(3.0*J2);
        }
#endif
        // finally limit the deviatoric stress tensor
        if (mises_f > 1.0)
            mises_f = 1.0;
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
