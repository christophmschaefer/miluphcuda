/**
 * @author      Christoph Schaefer, Oliver Wandel and Thomas I. Maindl
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
#include "config_parameter.h"
#include "parameter.h"
#include "internal_forces.h"
#include "boundary.h"
#include "pressure.h"
#include "linalg.h"
#include "viscosity.h"




extern __constant__ int isRelaxationRun;
extern __device__ SPH_kernel kernel;

//__launch_bounds__(64, 16)
__global__ void internalForces(int *interactions) {
    register int64_t interactions_index;
    int i, k, inc, j, numInteractions;
    int f, kk;

    double W;
    double tmp;
    double ax, ay;
    double sml;
#if DIM == 3
    double az;
#endif

    int matId;
    int matIdj;
    double sml1;

    double vxj, vyj, vzj, Sj[DIM*DIM];

#if FRAGMENTATION
    double di, di_tensile;  // both are directly the damage (not DIM-root of it)
#endif

#if ARTIFICIAL_VISCOSITY
    double vr; // vr = v_ij * r_ij
    double rr;
    double rhobar; // rhobar = 0.5*(rho_i + rho_j)
    double mu;
    double muijmax;
    double smooth;
    double csbar;
    double alpha, beta;

#if BALSARA_SWITCH
    double fi, fj;
    double curli, curlj;
    const double eps_balsara = 1e-4;
#endif
#endif

#if ARTIFICIAL_STRESS
    double artf = 0;
#endif

    int d;
    int dd;
    int e;
#if SOLID
    double sigma_i[DIM][DIM], sigma_j[DIM][DIM];
    double edot[DIM][DIM], rdot[DIM][DIM];
    double S_i[DIM][DIM];
    double sqrt_J2, I1, alpha_phi, kc;
    double lambda_dot, tr_edot;
#endif

    double dr[DIM];
    double dv[DIM];

    double x, vx;
#if DIM > 1
    double y, vy;
#endif
    int boundia = 0;
#if DIM == 3
    double z, vz;
#endif

    double drhodt;

#if INTEGRATE_ENERGY
    double dedt;
#endif

    double dvx;
#if DIM > 1
    double dvy;
#endif
#if DIM > 2
    double dvz;
#endif

#if NAVIER_STOKES
    double eta;
    double zetaij;
#endif

    double vvnablaW;
    double dWdr;
    double dWdrj;
    double dWdx[DIM];
    double Wj;
    double dWdxj[DIM];
    double pij = 0;
    double r;
    double accels[DIM];
    double accelsj[DIM];
    double accelshearj[DIM];

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {

        matId = p_rhs.materialId[i];
        //do nothing for boundary particles
        if (matId == BOUNDARY_PARTICLE_ID) continue;
        if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[i]] || matId == EOS_TYPE_IGNORE) {
                continue;
        }

        numInteractions = p.noi[i];

        ax = 0;
        ay = 0;
#if DIM > 2
        az = 0;
#endif

#if ARTIFICIAL_VISCOSITY
        alpha = matAlpha[matId];
        beta = matBeta[matId];
        muijmax = 0;
#endif
        sml1 = p.h[i];

        drhodt = 0;
#if INTEGRATE_ENERGY
        dedt = 0;
#endif
#if INTEGRATE_SML
        p.dhdt[i] = 0.0;
#endif

#if SOLID
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                // set rotation rate and strain rate tensor stuff to zero
                edot[d][e] = 0.0;
                rdot[d][e] = 0.0;
            }
        }
#endif
        for (d = 0; d < DIM; d++) {
            accels[d] = 0.0;
            accelsj[d] = 0.0;
            accelshearj[d] = 0.0;
        }
        sml = p.h[i];
#if FRAGMENTATION
        di = p.damage_total[i];
        if (di < 0.0) di = 0.0;
        if (di > 1.0) di = 1.0;
#endif

        x = p.x[i];
#if DIM > 1
        y = p.y[i];
# if DIM > 2
        z = p.z[i];
# endif
#endif
        vx = p.vx[i];
#if DIM > 1
        vy = p.vy[i];
# if DIM > 2
        vz = p.vz[i];
# endif
#endif
        p.dxdt[i] = 0;
        p.ax[i] = 0;
#if DIM > 1
        p.dydt[i] = 0;
        p.ay[i] = 0;
# if DIM > 2
        p.dzdt[i] = 0;
        p.az[i] = 0;
# endif
#endif

#if SOLID
        for (e = 0; e < DIM*DIM; e++) {
            p.dSdt[i*DIM*DIM+e] = 0.0;
        }
#endif
        p.drhodt[i] = 0.0;
#if INTEGRATE_ENERGY
        p.dedt[i] = 0.0;
#endif
#if INTEGRATE_SML
        p.dhdt[i] = 0.0;
#endif
#if FRAGMENTATION
        p.dddt[i] = 0.0;
# if PALPHA_POROSITY
        p.ddamage_porjutzidt[i] = 0.0;
# endif
#endif
#if PALPHA_POROSITY
        p.dalphadt[i] = 0.0;
#endif
        // if particle has no interactions continue and set all derivs to zero
        // but not the accels (these are handled in the tree for gravity)
        if (numInteractions < 1) {
        // finally continue
            continue;
        }

#if BALSARA_SWITCH
        curli = 0;
        for (d = 0; d < DIM; d++) {
            curli += p_rhs.curlv[i*DIM+d]*p_rhs.curlv[i*DIM+d];
        }
        curli = sqrt(curli);
        fi = fabs(p_rhs.divv[i]) / (fabs(p_rhs.divv[i]) + curli + eps_balsara*p.cs[i]/p.h[i]);
#endif

        // THE MAIN SPH LOOP FOR ALL INTERNAL FORCES
        // loop over interaction partners for SPH sums
        for (k = 0; k < numInteractions; k++) {
            matIdj = EOS_TYPE_IGNORE;
            // the interaction parotner
            interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + k;
            j = interactions[interactions_index];

            for (d = 0; d < DIM; d++) {
                accelsj[d] = 0.0;
                dWdxj[d] = 0.0;
                dWdx[d] = 0.0;
            }

            matIdj = p_rhs.materialId[j];
            if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[j]] || matIdj == EOS_TYPE_IGNORE) {
                continue;
            }


            boundia = 0;
            boundia = p_rhs.materialId[j] == BOUNDARY_PARTICLE_ID;
            /*
             * now, if the interaction partner is a BOUNDARY_PARTICLE
             * we need to determine the correct velocity, pressure and stress
             * for it
             */
#if (VARIABLE_SML || INTEGRATE_SML || DEAL_WITH_TOO_MANY_INTERACTIONS)
            sml = 0.5*(p.h[i] + p.h[j]);
#endif
            if (boundia) {
                /* set quantities for boundary particle */
                setQuantitiesFixedVirtualParticles(i, j, &vxj, &vyj, &vzj, &p.rho[j], &p.p[j], Sj);
            } else { /* no boundary particle, just copy */
                vxj = p.vx[j];
#if DIM > 1
                vyj = p.vy[j];
#if DIM > 2
                vzj = p.vz[j];
#endif
#endif
#if SOLID
                for (e = 0; e < DIM*DIM; e++)
                    Sj[e] = p.S[j*DIM*DIM+e];
#endif
            }

            // relative vector
            dr[0] = x - p.x[j];
#if DIM > 1
            dr[1] = y - p.y[j];
#if DIM > 2
            dr[2] = z - p.z[j];
#endif
#endif
            r = 0;
            for (e = 0; e < DIM; e++) {
                r += dr[e]*dr[e];
                dWdx[e] = 0.0;
#if AVERAGE_KERNELS
                dWdxj[e] = 0.0;
#endif
            }
            W = 0.0;
            dWdr = 0.0;
#if AVERAGE_KERNELS
            Wj = 0.0;
            dWdrj = 0.0;
#endif
            r = sqrt(r);

            // get kernel values for this interaction
#if AVERAGE_KERNELS
            kernel(&W, dWdx, &dWdr, dr, p.h[i]);
            kernel(&Wj, dWdxj, &dWdrj, dr, p.h[j]);
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            Wj /= p_rhs.shepard_correction[j];
            for (e = 0; e < DIM; e++) {
                dWdx[e] /= p_rhs.shepard_correction[i];
                dWdxj[e] /= p_rhs.shepard_correction[j];
            }
            dWdr /= p_rhs.shepard_correction[i];
            dWdrj /= p_rhs.shepard_correction[j];

            W = 0.5 * (W + Wj);
            dWdr = 0.5 * (dWdr + dWdrj);
            for (e = 0; e < DIM; e++) {
                dWdx[e] = 0.5 * (dWdx[e] + dWdxj[e]);
            }
# endif // SHEPARD_CORRECTION
#else
            kernel(&W, dWdx, &dWdr, dr, sml);
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            for (e = 0; e < DIM; e++) {
                dWdx[e] /= p_rhs.shepard_correction[i];
            }
            dWdr /= p_rhs.shepard_correction[i];
# endif
#endif

            dv[0] = dvx = vx - vxj;
#if DIM > 1
            dv[1] = dvy = vy - vyj;
#if DIM > 2
            dv[2] = dvz = vz - vzj;
#endif
#endif

            vvnablaW = dvx * dWdx[0];
#if DIM > 1
            vvnablaW += dvy * dWdx[1];
#if DIM > 2
            vvnablaW += dvz * dWdx[2];
#endif
#endif

#if ARTIFICIAL_VISCOSITY || KLEY_VISCOSITY
            rr = 0.0;
            vr = 0.0;
            for (e = 0; e < DIM; e++) {
                rr += dr[e]*dr[e];
                vr += dv[e]*dr[e];
            }
#endif

#if SOLID
            //get sigma_i
            if (matEOS[matId] != EOS_TYPE_REGOLITH) {
                for (d = 0; d < DIM; d++) {
                    for (e = 0; e < DIM; e++) {
                        sigma_i[d][e] = p_rhs.sigma[stressIndex(i, d, e)];
                    }
                }
            } else { // EOS type = regolith
                for (d = 0; d < DIM; d++) {
                    for (e = 0; e < DIM; e++) {
                        sigma_i[d][e] = p.S[stressIndex(i, d, e)];
                    }
                }
            } //material if

            //get sigma_j
            if (matEOS[p_rhs.materialId[j]] != EOS_TYPE_REGOLITH) {
                for (d = 0; d < DIM; d++) {
                    for (e = 0; e < DIM; e++) {
                        sigma_j[d][e] = p_rhs.sigma[stressIndex(j, d, e)];
                    }
                }
            } else { // EOS type = regolith
                for (d = 0; d < DIM; d++) {
                    for (e = 0; e < DIM; e++) {
                        sigma_j[d][e] = Sj[DIM*d+e];
                    }
                }
            } //material if

#endif //SOLID

#if SOLID
            // do calculation of edot and rdot only for real particle interaction partners and not
            // for boundary particle interaction partners

            // we do not need this for VISCOUS_REGOLITH particles since they have
            // a given deviatoric stress which is not integrated

            // calculate rotation rate and strain rate
            // tensor
            // see Benz (1995) or Libersky (1993)
            // Warning: Benz has typos in his paper....
            // edot_ab = 0.5 * (d_b v_a + d_a v_b)
            // rdot_ab = 0.5 * (d_b v_a - d_a v_b)
            if (EOS_TYPE_VISCOUS_REGOLITH != matEOS[matId]) {
                //printf("%d\n", boundia);
                tmp = p.m[j];
# if TENSORIAL_CORRECTION
                // new implementation (after july 2017, modified 2020)
                for (e = 0; e < DIM; e++) {
                    for (f = 0; f < DIM; f++) {
                        for (kk = 0; kk < DIM; kk++) {
                            edot[e][f] += 0.5 * p.m[j]/p.rho[j] *
                                (p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+f*DIM+kk] *
//                                  (-dv[e]) * dr[kk] * dWdr/r
                                  (-dv[e]) * dWdx[kk]
                                  + p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+e*DIM+kk] *
//                                  (-dv[f]) * dr[kk] * dWdr/r);
                                  (-dv[f]) * dWdx[kk]);

                            rdot[e][f] += 0.5 * p.m[j]/p.rho[j] *
                                (p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+f*DIM+kk] *
//                                  (-dv[e]) * dr[kk] * dWdr/r
                                  (-dv[e]) * dWdx[kk]
                                  - p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+e*DIM+kk] *
//                                  (-dv[f]) * dr[kk] * dWdr/r);
                                  (-dv[f]) * dWdx[kk]);
					    }
			    	}
			    }
# else
                tmp = -0.5*tmp/p.rho[i];
                edot[0][0] += tmp*(dvx*dWdx[0] + dvx*dWdx[0]);
#  if DIM > 1
                edot[0][1] += tmp*(dvx*dWdx[1] + dvy*dWdx[0]);
                edot[1][0] += tmp*(dvy*dWdx[0] + dvx*dWdx[1]);
                edot[1][1] += tmp*(dvy*dWdx[1] + dvy*dWdx[1]);
#  endif
#  if DIM == 3
                edot[0][2] += tmp*(dvx*dWdx[2] + dvz*dWdx[0]);
                edot[1][2] += tmp*(dvy*dWdx[2] + dvz*dWdx[1]);
                edot[2][0] += tmp*(dvz*dWdx[0] + dvx*dWdx[2]);
                edot[2][1] += tmp*(dvz*dWdx[1] + dvy*dWdx[2]);
                edot[2][2] += tmp*(dvz*dWdx[2] + dvz*dWdx[2]);
#  endif
                rdot[0][0] += tmp*(dvx*dWdx[0] - dvx*dWdx[0]);
#  if DIM > 1
                rdot[0][1] += tmp*(dvx*dWdx[1] - dvy*dWdx[0]);
                rdot[1][0] += tmp*(dvy*dWdx[0] - dvx*dWdx[1]);
                rdot[1][1] += tmp*(dvy*dWdx[1] - dvy*dWdx[1]);
#  endif
#  if DIM == 3
                rdot[0][2] += tmp*(dvx*dWdx[2] - dvz*dWdx[0]);
                rdot[1][2] += tmp*(dvy*dWdx[2] - dvz*dWdx[1]);
                rdot[2][0] += tmp*(dvz*dWdx[0] - dvx*dWdx[2]);
                rdot[2][1] += tmp*(dvz*dWdx[1] - dvy*dWdx[2]);
                rdot[2][2] += tmp*(dvz*dWdx[2] - dvz*dWdx[2]);
#  endif // DIM == 3
# endif // TENSORIAL_CORRECTION
            } // not EOS_TYPE_VISCOUS_REGOLITH
#endif // SOLID

            pij = 0.0;
#if ARTIFICIAL_VISCOSITY
            // artificial viscosity force only if v_ij * r_ij < 0
            if (vr < 0) {
                csbar = 0.5*(p.cs[i] + p.cs[j]);
                smooth = 0.5*(sml1 + p.h[j]);

                const double eps_artvisc = 1e-2;
                mu = smooth*vr/(rr + smooth*smooth*eps_artvisc);

                if (mu > muijmax) {
                    muijmax = mu;
                }
                rhobar = 0.5*(p.rho[i] + p.rho[j]);
# if BALSARA_SWITCH
                curlj = 0;
                for (d = 0; d < DIM; d++) {
                    curlj += p_rhs.curlv[j*DIM+d]*p_rhs.curlv[j*DIM+d];
                }
                curlj = sqrt(curlj);
                fj = fabs(p_rhs.divv[j]) / (fabs(p_rhs.divv[j]) + curlj + eps_balsara*p.cs[j]/p.h[j]);
                mu *= (fi+fj)/2.;
# endif
                pij = (beta*mu - alpha*csbar) * mu/rhobar;
# if INVISCID_SPH
                pij =  ((2 * mu - csbar) * p.beta[i] * mu) / rhobar;
# endif
            }
#endif // ARTIFICIAL_VISCOSITY


#if NAVIER_STOKES
            eta = (p_rhs.eta[i] + p_rhs.eta[j]) * 0.5 ;
            for (d = 0; d < DIM; d++) {
                accelshearj[d] = 0;
                for (dd = 0; dd < DIM; dd++) {
# if (SPH_EQU_VERSION == 1)
#  if SML_CORRECTION
                    accelshearj[d] += eta * p.m[j] * (p.Tshear[stressIndex(j,d,dd)]/(p.sml_omega[j]*p.rho[j]*p.rho[j])+ p.Tshear[stressIndex(i,d,dd)]/(p.sml_omega[i]*p.rho[i]*p.rho[i])) *dWdx[dd];
#  else
                    accelshearj[d] += eta * p.m[j] * (p.Tshear[stressIndex(j,d,dd)]/(p.rho[j]*p.rho[j]) + p.Tshear[stressIndex(i,d,dd)]/(p.rho[i]*p.rho[i])) *dWdx[dd];
#  endif
# elif (SPH_EQU_VERSION == 2)
#  if SML_CORRECTION
                    accelshearj[d] += eta * p.m[j] * (p.Tshear[stressIndex(j,d,dd)]+p.Tshear[stressIndex(i,d,dd)])/(p.sml_omega[i]*p.rho[i]*p.sml_omega[j]*p.rho[j]) *dWdx[dd];
#  else
                    accelshearj[d] += eta * p.m[j] * (p.Tshear[stressIndex(j,d,dd)]+p.Tshear[stressIndex(i,d,dd)])/(p.rho[i]*p.rho[j]) *dWdx[dd];
#  endif
# endif // SPH_EQU_VERSION
                }
            }
#if KLEY_VISCOSITY //artificial bulk viscosity with f=0.5
            zetaij = 0.0;
            if (vr < 0) { // only for approaching particles
                zetaij = -0.5 * (0.25*(p.h[i] + p.h[j])*(p.h[i]+p.h[j])) * (p.rho[i]+p.rho[j])*0.5 * (p_rhs.divv[i] + p_rhs.divv[j])*0.5;
            }
            for (d = 0; d < DIM; d++) {
# if (SPH_EQU_VERSION == 1)
                accelshearj[d] += zetaij * p.m[j] * (p_rhs.divv[i] + p_rhs.divv[j]) /(p.rho[i]*p.rho[j]) * dWdx[d];
# elif (SPH_EQU_VERSION == 2)
                accelshearj[d] += zetaij * p.m[j] * (p_rhs.divv[i]/(p.rho[i]*p.rho[i]) + p_rhs.divv[j]/(p.rho[j]*p.rho[j])) * dWdx[d];
# endif
            }
#endif // KLEY_VISCOSITY

#endif // NAVIER_STOKES


#if SOLID
# if ARTIFICIAL_STRESS
            artf = fixTensileInstability(i, j);
            artf =  pow(artf, matexponent_tensor[matId]);
# endif
            for (d = 0; d < DIM; d++) {
                accelsj[d] = 0;
                for (dd = 0; dd < DIM; dd++) {
                    // this is stable for rotating rod with two different densities, tested cms July 2018
                    //accelsj[d] = p.m[j] * (sigma_j[d][dd]+sigma_i[d][dd])/(p.rho[i]*p.rho[j]) *dWdx[dd];

                    // the same but with tensorial correction
# if (SPH_EQU_VERSION == 1)
                    // warning! look below, the accelsj for each inner loop are added to accels[d]
                    // this is very confusing
#  if SML_CORRECTION
                    accelsj[d] = p.m[j] * (sigma_j[d][dd]/(p.sml_omega[j]*p.rho[j]*p.rho[j]) + sigma_i[d][dd]/(p.sml_omega[i]*p.rho[i]*p.rho[i])) *dWdx[dd];
#  else
                    accelsj[d] = p.m[j] * (sigma_j[d][dd]/(p.rho[j]*p.rho[j]) + sigma_i[d][dd]/(p.rho[i]*p.rho[i])) *dWdx[dd];
#  endif
# elif (SPH_EQU_VERSION == 2)
#  if SML_CORRECTION
                    accelsj[d] = p.m[j] * (sigma_j[d][dd]+sigma_i[d][dd])/(p.sml_omega[i]*p.sml_omega[j]*p.rho[i]*p.rho[j]) *dWdx[dd];
#  else
                    accelsj[d] = p.m[j] * (sigma_j[d][dd]+sigma_i[d][dd])/(p.rho[i]*p.rho[j]) *dWdx[dd];
#  endif
# else
# error Invalid choice of SPH_EQU_VERSION in parameter.h.
# endif // SPH_EQU_VERSION

                    // the standard formula as also used by Martin Jutzi
                    //accelsj[d] = p.m[j] * (sigma_j[d][dd]/(p.rho[j]*p.rho[j]) + sigma_i[d][dd]/(p.rho[i]*p.rho[i])) *dWdx[dd];

                    // the version as suggested by Libersky, Randles, Carney, Dickinson 1997
                    // unstable for a rotating rod!
/*                    for (e = 0; e < DIM; e++) {
                        accelsj[d] +=  -p.m[j]/(p.rho[i]*p.rho[j]) * (sigma_j[d][dd] -
                                sigma_i[d][dd]) * dWdr/r * dr[e] *
                            p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+dd*DIM+e];
                      } */

// Correction for tensile instability fix according to Monaghan, jcp 159 (2000)
# if ARTIFICIAL_STRESS
                    double arts_rij;
#  if (SPH_EQU_VERSION == 1)
                    arts_rij = p_rhs.R[stressIndex(i,d,dd)]/(p.rho[i]*p.rho[i])
                              + p_rhs.R[stressIndex(j,d,dd)]/(p.rho[j]*p.rho[j]);
#  elif (SPH_EQU_VERSION == 2)
                    arts_rij = (p_rhs.R[stressIndex(i,d,dd)] + p_rhs.R[stressIndex(j,d,dd)])/(p.rho[i]*p.rho[j]);
#  endif
                    // add the special artificial stress
                    accels[d] += p.m[j] * arts_rij * artf * dWdx[dd];
# endif // ARTIFICIAL_STRESS

                    // bs...
                   // accels[d] += p.m[j] * (sigma_j[d][dd]/pow(p.rho[j],2) + sigma_i[d][dd]/pow(p.rho[i],2)) * dWdr/r * (-dr[dd]) * (-dr[d]) * p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+d*DIM+dd];

                    accels[d] += accelsj[d];
                }
            }
#else // NOT SOLID
# if (SPH_EQU_VERSION == 1)
#  if SML_CORRECTION
            for (d = 0; d < DIM; d++) {
                accelsj[d] =  -p.m[j] * (p.p[i]/(p.sml_omega[i]*p.rho[i]*p.rho[i]) + p.p[j]/(p.sml_omega[j]*p.rho[j]*p.rho[j])) * dWdx[d];
                accels[d] += accelsj[d];
            }
#  else
            for (d = 0; d < DIM; d++) {
                accelsj[d] =  -p.m[j] * (p.p[i]/(p.rho[i]*p.rho[i]) + p.p[j]/(p.rho[j]*p.rho[j])) * dWdx[d];
                accels[d] += accelsj[d];
            }
#  endif // SML_CORRECTION
# elif (SPH_EQU_VERSION == 2)
#  if SML_CORRECTION
            for (d = 0; d < DIM; d++) {
                accelsj[d] =  -p.m[j] * ((p.p[i]+p.p[j])/(p.sml_omega[i]*p.rho[i]*p.sml_omega[j]*p.rho[j])) * dWdx[d];
                accels[d] += accelsj[d];
            }
#  else
            for (d = 0; d < DIM; d++) {
                accelsj[d] =  -p.m[j] * ((p.p[i]+p.p[j])/(p.rho[i]*p.rho[j])) * dWdx[d];
                accels[d] += accelsj[d];
            }
#  endif
# endif // SPH_EQU_VERSION
#endif // SOLID

#if NAVIER_STOKES
// add viscous accel to total accel
            for (d = 0; d < DIM; d++) {
                accels[d] += accelshearj[d];
            }
#endif

# if ARTIFICIAL_VISCOSITY
            accels[0] += p.m[j]*(-pij)*dWdx[0];
#  if DIM > 1
            accels[1] += p.m[j]*(-pij)*dWdx[1];
#   if DIM > 2
            accels[2] += p.m[j]*(-pij)*dWdx[2];
#   endif
# endif
# endif


#if SOLID
            // use old version, not Frank Ott's
            //drhodt += p.m[j]*vvnablaW;
            // density integration stuff
            // see Frank Ott's thesis for details
            //drhodt += p.m[i]*vvnablaW;
            // Randles and Libersky's version (1996)
# if TENSORIAL_CORRECTION
#  if 0 // cms 2020-06-10 testing time step size, this part gives a tiny step size due to density
      //                evolution, needs some debugging
      // debugging started 2023-02-15, for the colliding rings, the original version is stable, this one not!
            double divv = 0.0;
            for (d = 0; d < DIM; d++) {
                for (dd = 0; dd < DIM; dd++) {
                    divv += p.m[j]/p.rho[j] * dv[d] * p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+d*DIM+dd] * dWdx[dd];
                }
            }
            drhodt += p.rho[i] * divv;
#  else
            drhodt += p.rho[i]/p.rho[j] * p.m[j] * vvnablaW;
#  endif
# else
            drhodt += p.rho[i]/p.rho[j] * p.m[j] * vvnablaW;
# endif // TENSORIAL CORRECTION

#else // HYDRO now
# if SML_CORRECTION 
            drhodt += p.m[j]*vvnablaW;
# else
            drhodt += p.rho[i]/p.rho[j] * p.m[j] * vvnablaW;
# endif // SML_CORRECTION
#endif // SOLID


#if INTEGRATE_SML
            // minus since vvnablaW is v_i - v_j \nabla W_ij
# if TENSORIAL_CORRECTION
            for (d = 0; d < DIM; d++) {
                for (dd = 0; dd < DIM; dd++) {
                    p.dhdt[i] -= 1./DIM * p.h[i] * p.m[j]/p.rho[j] * dv[d] * dWdx[dd] * p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+d*DIM+dd];
                }
            }
# else
#  if !SML_CORRECTION
            p.dhdt[i] -= 1./DIM * p.h[i] * p.m[j]/p.rho[j] * vvnablaW;
#  endif // SML_CORRECTION
# endif
#endif // INTEGRATE_SML

#if INTEGRATE_ENERGY
# if ARTIFICIAL_VISCOSITY
            if (!isRelaxationRun) {
#  if SML_CORRECTION
                dedt += p.m[j] * vvnablaW;
#  else
                dedt += 0.5 * p.m[j] * pij * vvnablaW;
#  endif // SML_CORRECTION
            }
# endif

# if SOLID
# if 0 // deactivated cms 2019-07-02 SOLID
// new implementation cms 2019-05-23
            for (d = 0; d < DIM; d++) {
                for (dd = 0; dd < DIM; dd++) {
#  if (SPH_EQU_VERSION == 1)
                    dedt += 0.5 * p.m[j] * (p_rhs.sigma[stressIndex(i,d,dd)]/(p.rho[i]*p.rho[i]) + p_rhs.sigma[stressIndex(j,d,dd)]/(p.rho[j]*p.rho[j])) * dv[d] * dWdx[dd];
#  elif (SPH_EQU_VERSION == 2)
                    dedt += 0.5 * p.m[j] * (p_rhs.sigma[stressIndex(i,d,dd)] + p_rhs.sigma[stressIndex(j,d,dd)])/(p.rho[i]*p.rho[j]) * dv[d] * dWdx[dd];
#endif
#if DEBUG_MISC
                    if (isnan(dedt)) {
                        printf("no %d m=%e sigma_i[%d][%d]=%e sigma_j[%d][%d]= %e dv[%d] %e  dWdx[%d] %e  p_i %e  p_j %e rho_i %e rho_j %e pij %e cs_i %e cs_j %e\n", i, p.m[j], d, dd, p_rhs.sigma[stressIndex(i,d,dd)], d, dd,p_rhs.sigma[stressIndex(j,d,dd)], d, dv[d], dd, dWdx[dd], p.p[i], p.p[j], p.rho[i], p.rho[j], pij,p.cs[i], p.cs[j]);
                        assert(0);
                    }
#endif
                }
            }
#endif // 0 deactivation from cms 2019-07-02

# else // dedt for non-solid
            // remember, accelsj  are accelerations by particle j, and dv = v_i - v_j
            dedt += 0.5 * accelsj[0] * -dvx;
#  if DIM > 1
            dedt += 0.5 * accelsj[1] * -dvy;
#  endif
#  if DIM > 2
            dedt += 0.5 * accelsj[2] * -dvz;
#  endif
# endif // SOLID

#endif // INTEGRATE ENERGY

        } // neighbors loop end

        ax = accels[0];
#if DIM > 1
        ay = accels[1];
#endif
#if DIM > 2
        az = accels[2];
#endif
        p.ax[i] = ax;
#if DIM > 1
        p.ay[i] = ay;
#endif
#if DIM > 2
        p.az[i] = az;
#endif

#if SML_CORRECTION
        p.drhodt[i] = 1 / p.sml_omega[i] * drhodt;
        p.dhdt[i] = - p.h[i] / (DIM * p.rho[i]) * p.drhodt[i];
#else
        p.drhodt[i] = drhodt;
#endif // SML_CORRECTION

#if INTEGRATE_DENSITY
        // if the density is calculated via kernel sum, we set drhodt to 0 here again
        if (matdensity_via_kernel_sum[p_rhs.materialId[i]]) {
            p.drhodt[i] = 0.0;
        }
#endif


#if INTEGRATE_ENERGY
# if SOLID
        double ptmp = 0.0;
        double edottmp = 0.0;

#  if COLLINS_PLASTICITY
        // influence of damage on p < 0 already set in plasticity.cu
        ptmp = p.p[i];
#  elif FRAGMENTATION
        if (p.p[i] < 0.0) {
            // reduction of neg. pressure following Grady-Kipp model
            ptmp = (1.0 - di) * p.p[i];
        } else {
            ptmp = p.p[i];
        }
#  else
        ptmp = p.p[i];
#  endif

        dedt -= ptmp / p.rho[i] * p_rhs.divv[i];
        // symmetrize edot
        for (d = 0; d < DIM; d++) {
            for (dd = 0; dd < d; dd++) {
                edottmp = 0.5*(edot[d][dd] + edot[dd][d]);
                edot[d][dd] = edottmp;
                edot[dd][d] = edottmp;
            }
        }
        for (d = 0; d < DIM; d++) {
            for (dd = 0; dd < DIM; dd++) {
                double Stmp = p.S[stressIndex(i,d,dd)];
#  if FRAGMENTATION && DAMAGE_ACTS_ON_S
                // reduction of S following Grady-Kipp model
                Stmp *= (1.0-di);
#  endif
                dedt += Stmp / p.rho[i] * edot[d][dd];
            }
        }
# endif // SOLID
# if SML_CORRECTION
        p.dedt[i] = p.p[i]/(p.rho[i]*p.rho[i] * p.sml_omega[i]) * dedt;
# else
        p.dedt[i] = dedt;
# endif // SML_CORRECTION
#endif // INTEGRATE_ENERGY


#if PALPHA_POROSITY
        if (matEOS[matId] == EOS_TYPE_JUTZI || matEOS[matId] == EOS_TYPE_JUTZI_MURNAGHAN || matEOS[matId] == EOS_TYPE_JUTZI_ANEOS) {
            if (p.alpha_jutzi[i] <= 1.0) {
                p.dalphadt[i] = 0.0;
                p.alpha_jutzi[i] = 1.0;
            } else {
#if INTEGRATE_ENERGY
                p.dalphadt[i] = ((p.dedt[i] * p.delpdele[i] + p.alpha_jutzi[i] * p.drhodt[i] * p.delpdelrho[i])
                              * p.dalphadp[i]) / (p.alpha_jutzi[i] + p.dalphadp[i] * (p.p[i] - p.rho[i] * p.delpdelrho[i]));
#else
                p.dalphadt[i] = ((p.alpha_jutzi[i] * p.drhodt[i] * p.delpdelrho[i])
                              * p.dalphadp[i]) / (p.alpha_jutzi[i] + p.dalphadp[i] * (p.p[i] - p.rho[i] * p.delpdelrho[i]));

#endif
                if (p.dalphadt[i] > 0.0) {
                    p.dalphadt[i] = 0.0;
                }
            }
        } else {
            p.dalphadt[i] = 0.0;
        }
#endif

#if EPSALPHA_POROSITY
        /* calculate the change in epsilon and alpha per time */
        if (matEOS[matId] == EOS_TYPE_EPSILON) {
            double dalpha_epspordeps = 0.0;
            p.depsilon_vdt[i] = 0.0;
            int f;
            double kappa = matporepsilon_kappa[matId];
            double alpha_0 = matporepsilon_alpha_0[matId];
            double eps_e = matporepsilon_epsilon_e[matId];
            double eps_x = matporepsilon_epsilon_x[matId];
            double eps_c = matporepsilon_epsilon_c[matId];
            for (f = 0; f < DIM; f++) {
                p.depsilon_vdt[i] += edot[f][f];
            }
            if (p.alpha_epspor[i] <= 1.0) {
                p.dalpha_epspordt[i] = 0.0;
                p.alpha_epspor[i] = 1.0;
                if (p.depsilon_vdt[i] > 0.0)
                    p.depsilon_vdt[i] = 0.0;
            } else {
                if (p.depsilon_vdt[i] < 0.0) {
                    if (p.epsilon_v[i] >= eps_e) {
                        dalpha_epspordeps = 0.0;
                    } else if (p.epsilon_v[i] < eps_e && p.epsilon_v[i] >= eps_x) {
                        dalpha_epspordeps = alpha_0 * kappa * exp(kappa * (p.epsilon_v[i] - eps_e));
                    } else if (p.epsilon_v[i] < eps_x && p.epsilon_v[i] > eps_c) {
                        dalpha_epspordeps = 2.0 * (1.0 - alpha_0 * exp(kappa * (eps_x - eps_e))) * (eps_c - p.epsilon_v[i]) / (pow((eps_c - eps_x), 2));
                    } else if (p.epsilon_v[i] <= eps_c) {
                        p.alpha_epspor[i] = 1.0;
                        dalpha_epspordeps = 0.0;
                    }
                } else {
                    p.depsilon_vdt[i] = 0.0;
                }
                p.dalpha_epspordt[i] = dalpha_epspordeps * p.depsilon_vdt[i];
            }
        } else {
            p.dalpha_epspordt[i] = 0.0;
            p.depsilon_vdt[i] = 0.0;
        }
#endif

#if SOLID
        // now we can find the change of the stress tensor components
        double shear = matShearmodulus[matId];
        double bulk = matBulkmodulus[matId];
        double young = matYoungModulus[matId];
        int f;
# if SOLID
	    double edotp[DIM][DIM]; // plastic strain rate
# endif
# if SIRONO_POROSITY
        if (matEOS[matId] == EOS_TYPE_SIRONO) {
            shear = 0.5 * p.K[i];
            bulk = p.K[i];
            young = (9.0 * bulk * shear / (3.0 * bulk + shear));
        }
# endif

        if (matEOS[matId] != EOS_TYPE_REGOLITH && matEOS[matId] != EOS_TYPE_VISCOUS_REGOLITH) {
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    // Hooke's law
                    p.dSdt[stressIndex(i,d,e)] = 2.0 * shear * edot[d][e];
# if JC_PLASTICITY
		            edotp[d][e] = (1 - p.jc_f[i]) * edot[d][e];
# else // plasticity via other plasticity model
                    edotp[d][e] = (1 - p_rhs.plastic_f[i]) * edot[d][e];
# endif
                    // rotation terms
                    for (f = 0; f < DIM; f++) {
                        // trace
                        if (d == e) {
                            p.dSdt[stressIndex(i,d,e)] -= 2.0 * shear * edot[f][f] / 3.0;
# if JC_PLASTICITY
		            	    edotp[d][e] += (-1./3)*(1-p.jc_f[i])*edot[f][f];
# else
                            edotp[d][e] += (-1./3)*(1-p_rhs.plastic_f[i])*edot[f][f];
# endif
                        }
                        p.dSdt[stressIndex(i,d,e)] += p.S[stressIndex(i,d,f)] * rdot[e][f];
                        p.dSdt[stressIndex(i,d,e)] += p.S[stressIndex(i,e,f)] * rdot[d][f];
                    }
# if PALPHA_POROSITY && STRESS_PALPHA_POROSITY
                    if (matEOS[matId] == EOS_TYPE_JUTZI || matEOS[matId] == EOS_TYPE_JUTZI_MURNAGHAN || matEOS[matId] == EOS_TYPE_JUTZI_ANEOS) {
                        p.dSdt[stressIndex(i,d,e)] = p.f[i] / p.alpha_jutzi[i] * p.dSdt[stressIndex(i,d,e)]
                                                            - 1.0 / (p.alpha_jutzi[i]*p.alpha_jutzi[i])

#  if 0 // FRAGMENTATION && DAMAGE_ACTS_ON_S
                                                            * (1-di)*p.S[stressIndex(i,d,e)]
#  else
                                                            * p.S[stressIndex(i,d,e)]
#  endif
                                                            * p.dalphadt[i];
                    }
# endif
                }
            }

# if JC_PLASTICITY
            /* calculate plastic strain rate tensor from dSdt */
            double K2 = 0;
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    K2 += 0.5*edotp[d][e]*edotp[d][e];
                }
            }
            p.edotp[i] = 2./3. * sqrt(3*K2);

            /* change of temperature due to plastic deformation */
            double work = 0;
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    work += sigma_i[d][e] * edotp[d][e];
                }
            }
            /* these are the particles that fail the adiabatic assumption */
            if (work < 0) {
                /*  fprintf(stderr, "Warning: work related to plastic strain is negative for particle %d located at \t", i);
                for (d = 0; d < DIM; d++)
                    fprintf(stderr, "x[%d]: %g \t", d, p[i].x[d]);
                fprintf(stderr, "\n"); */
                work = 0;
            }
            /* daniel Thun daniel thun */
            p.dTdt[i] = work / (matCp[p_rhs.materialId[i]] * p.rho[i]);
            if (p.dTdt[i] < 0) {
                //fprintf(stderr, "%d work: %g, Cp: %g, rho: %g\n", i, work, matCp[p_rhs.materialId[i]], p.rho[i]);
            }
            if (p.noi[i] < 1)
                p.dTdt[i] = 0.0;
# else // some other plasticity model at work
            /* calculate plastic strain rate tensor from dSdt */
            double K2 = 0;
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    // a measure for the total deviatoric strain rate
//                    K2 += p.dSdt[stressIndex(i,d,e)]*p.dSdt[stressIndex(i,d,e)];
                    K2 += edotp[d][e]*edotp[d][e];
                }
            }
            // still to double check factor 2./3 here -> reference from LS-DYNA support page on effective plastic strain
            p.edotp[i] = sqrt(2./3.*K2);
            // now consider only the plastic part (with the plasticity factor from this time step and convert to strain
//            p.edotp[i] = (1-p_rhs.plastic_f[i])/(3*shear)*sqrt(3./2.*K2);
# endif

# if ARTIFICIAL_VISCOSITY
            p.muijmax[i] = muijmax;
# endif

            double tensileMax = 0.0;
            tensileMax = calculateMaxEigenvalue(sigma_i);
            p.local_strain[i] = tensileMax/young;

# if FRAGMENTATION
            // calculate damage evolution dd/dt...
            // 1st: get max eigenvalue (max principle stress) of sigma_i
            // 2nd: get local scalar strain out of max tensile stress
            di_tensile = pow(p.d[i], DIM);  // because p.d is DIM-root of damage
            if (di_tensile < 1.0) {
                p.local_strain[i] = tensileMax / ((1.0 - di_tensile) * young);

                // 3rd: calculate dd/dt
                // note: d(d**1/DIM)/dt is calculated
                // speed of a longitudinal elastic wave, see eg. Melosh, Impact Cratering
                // crack growth velocity = 0.4 times c_elast
                double c_g = 0.4 * sqrt((bulk + 4.0 * shear * (1.0 - di_tensile) / 3.0) * 1.0 / p.rho[i]);

                // find number of active flaws
                int n_active = 0;
                for (d = 0; d < p.numFlaws[i]; d++) {
                    if (p_rhs.flaws[i*maxNumFlaws+d] < p.local_strain[i]) {
                        n_active++;
                    }
                }
                p.numActiveFlaws[i] = max(n_active, p.numActiveFlaws[i]);
                p.dddt[i] = n_active * c_g / sml1;

                if (p.dddt[i] < 0.0) {
                    printf("ERROR. Found dd/dt < 0 for:\n");
                    printf("x: %e\t y: %e\t damage_total: %e\t numFlaws: %d\t numActiveFlaws: %d\t dddt: %e\t local_strain: %e\n",
                            p.x[i], p.y[i], p.damage_total[i], p.numFlaws[i], p.numActiveFlaws[i], p.dddt[i], p.local_strain[i]);
                }
            } else {
                // particle already dead
                p.local_strain[i] = 0.0;
                p.numActiveFlaws[i] = p.numFlaws[i];
                p.dddt[i] = 0.0;
                p.d[i] = 1.0;
            }
#  if PALPHA_POROSITY
            if (matEOS[matId] == EOS_TYPE_JUTZI || matEOS[matId] == EOS_TYPE_JUTZI_MURNAGHAN || matEOS[matId] == EOS_TYPE_JUTZI_ANEOS) {
                double deld = 0.01; 	/* variation in the damage to avoid infinity problem */
                double alpha_0 = matporjutzi_alpha_0[matId];
                if (alpha_0 > 1.0) {
                    p.ddamage_porjutzidt[i] = - 1.0/DIM * (pow(1.0 - (p.alpha_jutzi[i] - 1.0) / (alpha_0 - 1.0) + deld, 1.0/DIM - 1.0))
                                            / (pow(1.0 + deld, 1.0/DIM) - pow(deld, 1.0/DIM)) * 1.0/(alpha_0 - 1.0) * p.dalphadt[i];
                }
            }
#  endif
# endif // FRAGMENTATION

        } else if (matEOS[matId] != EOS_TYPE_VISCOUS_REGOLITH) { // if materialtype = regolith
            alpha_phi = matAlphaPhi[matId];
            kc = matCohesionCoefficient[matId];
            tr_edot = 0.0;
            for (d = 0; d < DIM; d++) {
                tr_edot += edot[d][d];
            }
# if DIM == 2
            double poissons_ratio = (3*bulk - 2*shear) / (2*(3*bulk + shear));
            I1 = (1 + poissons_ratio) * (p.S[stressIndex(i, 0, 0)] + p.S[stressIndex(i, 1, 1)]);
# else
            I1 = p.S[stressIndex(i,0,0)] + p.S[stressIndex(i,1,1)] + p.S[stressIndex(i,2,2)];
# endif
            //get S
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    S_i[d][e] = p.S[stressIndex(i, d, e)];
                }
                S_i[d][d] -= I1/3.0;
            }
# if DIM == 2
            double sz = poissons_ratio*(S_i[0][0] + S_i[1][1]);
# endif
            //calculate sqrt(J2)
            sqrt_J2 = 0.0;
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    sqrt_J2 += S_i[d][e]*S_i[d][e];
                }
            }
# if DIM == 2
            sqrt_J2 += sz*sz;
# endif
            sqrt_J2 *= 0.5;
            sqrt_J2 = sqrt(sqrt_J2);

            //calculate lambda_dot
            lambda_dot = 0.0;
            if (!(sqrt_J2 + alpha_phi * I1 - kc < 0)) {
                if (sqrt_J2 > 0.0) {
                    for (d = 0; d < DIM; d++) {
                        for (e = 0; e < DIM; e++) {
                            lambda_dot += S_i[d][e]*edot[d][e];
                        }
                    }
                    lambda_dot *= shear/sqrt_J2;
                }
                lambda_dot += 3*alpha_phi*bulk*tr_edot;
                /*lambda_dot /= 9*alpha_phi*alpha_phi*bulk + shear;*/
                lambda_dot /= shear;
            }

            // do not mess up with the elastic regime
            if (lambda_dot < 0)
                lambda_dot = 0.0;

            //calculate dsigmadt
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    p.dSdt[stressIndex(i,d,e)] = 2*shear*edot[d][e];
                    for (f = 0; f < DIM; f++) {
                        p.dSdt[stressIndex(i,d,e)] += p.S[stressIndex(i,d,f)]*rdot[e][f] + p.S[stressIndex(i,f,e)]*rdot[d][f];
                    }
                    if (sqrt_J2 > 0.0) {
                        p.dSdt[stressIndex(i,d,e)] -= S_i[d][e]*lambda_dot*shear/sqrt_J2;
                    }
                }
                /*p.dSdt[stressIndex(i,d,d)] += tr_edot*(bulk-2*shear/3.0) - 3*lambda_dot*alpha_phi*bulk;*/
                p.dSdt[stressIndex(i,d,d)] += tr_edot*(bulk-2*shear/3.0);
            }
# if FRAGMENTATION
            /* disable fragmentation for regolith, cause there's none */
            p.local_strain[i] = 0.0;
            p.numActiveFlaws[i] = 0;
            p.dddt[i] = 0.0;
# endif
        } else if (matEOS[matId] == EOS_TYPE_VISCOUS_REGOLITH) {
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    p.dSdt[stressIndex(i,d,e)] = 0.0;
                }
            }
        } //end material-if

#endif // SOLID

    } // particle loop end
}



#if VISCOUS_REGOLITH
__global__ void calculatedeviatoricStress(int *interactions)
{
    register int64_t interactions_index;
    register int i, j, inc, d, e, k;
    register int mt;
    int noi;
    int f, kk;
    double dx, dy, dvx, dvy;
    double dv[DIM];
    double dr[DIM];
    double W, dWdx[DIM];
    double Wj, dWdxj[DIM];
    double dWdr;
    double dWdrj;
    double x, y;
#if DIM == 3
    double dz, z, dvz;
#endif
    double edot[DIM][DIM], edottrace;
    double tmp;
    register double srp; // strain-rate parameter
    register double mustar;
    double mumax = 2e3; // FixMe! This is the upper limit of the Mohr-Coulomb yield stress criterion
    int matId;
    double sml;
    double r;

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        mt = p_rhs.materialId[i];
        sml = p.h[i];
        matId = mt;
        if (matEOS[mt] != EOS_TYPE_VISCOUS_REGOLITH) {
            continue;
        }
        noi = p.noi[i];

        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                edot[d][e] = 0.0;
            }
        }
        x = p.x[i];
        y = p.y[i];
#if DIM > 2
        z = p.z[i];
#endif
         /* interaction loop */
        for (k = 0; k < noi; k++) {
            // interacting particle id
            interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + k;
            j = interactions[interactions_index];
            if (EOS_TYPE_IGNORE == matEOS[p_rhs.materialId[j]] || EOS_TYPE_IGNORE == p_rhs.materialId[j]) {
                continue;
            }
            /* relative vector */
            dr[0] = dx = x - p.x[j];
            dr[1] = dy = y - p.y[j];
            dv[0] = dvx = p.vx[i] - p.vx[j];
            dv[1] = dvy = p.vy[i] - p.vy[j];
#if DIM > 2
            dr[2] = dz = z - p.z[j];
            dv[2] = dvz = p.vz[i] - p.vz[j];
#endif
            r = 0;
            for (e = 0; e < DIM; e++) {
                r += dr[e]*dr[e];
            }
            r = sqrt(r);

#if (VARIABLE_SML || INTEGRATE_SML || DEAL_WITH_TOO_MANY_INTERACTIONS)
            sml = 0.5*(p.h[i] + p.h[j]);
#endif

            // get kernel values for this interaction
#if AVERAGE_KERNELS
            kernel(&W, dWdx, &dWdr, dr, p.h[i]);
            kernel(&Wj, dWdxj, &dWdrj, dr, p.h[j]);
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            Wj /= p_rhs.shepard_correction[j];
            for (e = 0; e < DIM; e++) {
                dWdx[e] /= p_rhs.shepard_correction[i];
                dWdxj[e] /= p_rhs.shepard_correction[j];
            }
            dWdr /= p_rhs.shepard_correction[i];
            dWdrj /= p_rhs.shepard_correction[j];

            W = 0.5 * (W + Wj);
            dWdr = 0.5 * (dWdr + dWdrj);
            for (e = 0; e < DIM; e++) {
                dWdx[e] = 0.5 * (dWdx[e] + dWdxj[e]);
            }
# endif // SHEPARD_CORRECTION
#else
            kernel(&W, dWdx, &dWdr, dr, sml);
# if SHEPARD_CORRECTION
            W /= p_rhs.shepard_correction[i];
            for (e = 0; e < DIM; e++) {
                dWdx[e] /= p_rhs.shepard_correction[i];
            }
            dWdr /= p_rhs.shepard_correction[i];
# endif
#endif




          //  printf("W %e dWdx %e dWdy %e dWdz %e i %d j %d dx %e dy %e dz %e\n", W, dWdx, dWdy, dWdz, i, j, dx, dy, dz);
            // do calculation of edot and rdot only for real particle interaction partners and not
            // for boundary particle interaction partners

            // calculate rotation rate and strain rate
            // tensor
            // see Benz (1995) or Libersky (1993)
            // Warning: Benz has typos in his paper....
            // edot_ab = 0.5 * (d_b v_a + d_a v_b)
            // rdot_ab = 0.5 * (d_b v_a - d_a v_b)
            tmp = p.m[j];
#if TENSORIAL_CORRECTION
            tmp = -0.5*tmp/p.rho[j]*dWdr/r;
            // new implementation (after july 2017)
            for (e = 0; e < DIM; e++) {
                for (f = 0; f < DIM; f++) {
                    for (kk = 0; kk < DIM; kk++) {
                        edot[e][f] += 0.5 * p.m[j]/p.rho[j] *
                            (p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+f*DIM+kk] *
                              (-dv[e]) * dr[kk] * dWdr/r
                              + p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+e*DIM+kk] *
                              (-dv[f]) * dr[kk] * dWdr/r);
                    }
                }
            }
#else
            tmp = -0.5*tmp/p.rho[i];
            edot[0][0] += tmp*(dvx*dWdx[0] + dvx*dWdx[0]);
            edot[0][1] += tmp*(dvx*dWdx[1] + dvy*dWdx[0]);
            edot[1][0] += tmp*(dvy*dWdx[0] + dvx*dWdx[1]);
            edot[1][1] += tmp*(dvy*dWdx[1] + dvy*dWdx[1]);
#if DIM > 2
            edot[0][2] += tmp*(dvx*dWdx[2] + dvz*dWdx[0]);
            edot[1][2] += tmp*(dvy*dWdx[2] + dvz*dWdx[1]);
            edot[2][0] += tmp*(dvz*dWdx[0] + dvx*dWdx[2]);
            edot[2][1] += tmp*(dvz*dWdx[1] + dvy*dWdx[2]);
            edot[2][2] += tmp*(dvz*dWdx[2] + dvz*dWdx[2]);
#endif
#endif // TENSORIAL_CORRECTION


#if 0
            if (isnan(dvx) || isnan(dvy) || isnan(dvz)) {
//                printf("ACCELS %e %e %e\n", ax, ay, az);
                printf("MATERIAL IDs %d %d\n", matId, p_rhs.materialId[j]);
                printf("ilocations --- jlocations %e %e %e  --- %e %e %e\n", p.x[i], p.y[i], p.z[i], p.x[j], p.y[j], p.z[j]);
                printf("dWdx  --- %e %e %e\n", dWdx, dWdy, dWdz);
                printf("DX %e %e %e\n", dx, dy, dz);
                printf("DENSITIES: i %e j %e\n", p.rho[i], p.rho[j]);
                printf("PRESSURES: i %e j %e\n", p.p[i], p.p[j]);
                printf(" EDOTI");
                for (e = 0; e < DIM; e++) {
                    for (d = 0; d < DIM; d++) {
                        printf("EDOTI[%d][%d] = %e", e, d, edot[e][d]);
                    }
               }
                printf("TMP: %e \n" , tmp);
                assert(1);
            }
#endif

        } /* interaction loop end */

        edottrace = 0.0;
        /* trace of the strain rate tensor */
        for (d = 0; d < DIM; d++) {
            edottrace += edot[d][d];
        }
        /* remove edottrace from edot to make traceless tensor */
        edot[0][0] -= 1./3 * edottrace;
        edot[1][1] -= 1./3 * edottrace;
#if DIM > 2
        edot[2][2] -= 1./3 * edottrace;
#endif
        // now let's calculate S from edot and cohesion and internal friction
        // formulae (26) and (29) from Ulrich et al. 2013
        srp = 0.0;
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                srp += edot[d][e]*edot[d][e];
            }
        }
        //printf("srp: %e\n", srp);
        srp = sqrt(4*srp);
        mustar = matCohesion[mt] + p.p[i]*tan(matFrictionAngle[mt]);
        if (srp > 0) {
            mustar /= srp;
            if (mustar > mumax) {
                mustar = mumax;
            }
        } else {
            mustar = mumax;
        }
        //printf("%d %e %e %e %e %e\n", mt, srp, mustar,  matCohesion[mt], p.p[i], matFrictionAngle[mt]);

        /* deviatoric stress */
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                p.S[stressIndex(i, d, e)] = 2*edot[d][e]*mustar;
            }
        }

    } // loop over particles
}
#endif // VISCOUS_REGOLITH
