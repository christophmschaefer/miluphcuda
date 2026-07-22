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
 */



/* coupled heun and rk4 integrator */
// Heun is used for the SPH particles
// RK4 for the NBODYs
// following the idea from Daniel Thun in Append A of https://www.aanda.org/articles/aa/pdf/2018/08/aa32804-18.pdf
// note: this integrator is designed to calculate the nbody orbit with high precision while keeping
//       the hydro computational effort low (meaning with a higher timestep)
// designed for circumbinary disks


// authors: Evita Vavilina and cms



#include "coupled_heun_rk4_sph_nbody.h"
#include "config_parameter.h"
#include "timeintegration.h"
#include "parameter.h"
#include "memory_handling.h"
#include "miluph.h"
#include "pressure.h"
#include "rhs.h"
#include "gravity.h"
#include "damage.h"
#include <float.h>


extern __device__ double endTimeD, currentTimeD;
extern __device__ double substep_currentTimeD;
extern __device__ double dt;
extern __device__ double dtmax;
extern __device__ int blockCount;
extern __device__ double emin_d;
extern __device__ double Smin_d;
extern __device__ double rhomin_d;
extern __device__ double damagemin_d;
extern __device__ double alphamin_d;
extern __device__ double betamin_d;
extern __device__ double alpha_epspormin_d;
extern __device__ double epsilon_vmin_d;
extern __device__ int pressureChangeSmallEnough;
extern __device__ double maxpressureDiff;


extern double L_ini;


__global__ void CorrectorStep_heun()
{
    register int i;
#if SOLID
    register int j;
    register int k;
#endif

    // particle loop
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        p.x[i] = p.x[i] + dt/2 * (predictor.dxdt[i] + p.dxdt[i]);
#if DIM > 1
        p.y[i] = p.y[i] + dt/2 * (predictor.dydt[i] + p.dydt[i]);
        p.vy[i] = p.vy[i] + dt/2 * (predictor.ay[i] + p.ay[i]);
        p.ay[i] = 0.5*(predictor.ay[i] + p.ay[i]);
#endif
        p.vx[i] = p.vx[i] + dt/2 * (predictor.ax[i] + p.ax[i]);
        p.ax[i] = 0.5*(predictor.ax[i] + p.ax[i]);
#if DIM == 3
        p.z[i] = p.z[i] + dt/2 * (predictor.dzdt[i] + p.dzdt[i]);
        p.vz[i] = p.vz[i] + dt/2 * (predictor.az[i] + p.az[i]);
        p.az[i] = 0.5*(predictor.az[i] + p.az[i]);
#endif
#if INTEGRATE_ENERGY
        p.e[i] = p.e[i] + dt/2 * (predictor.dedt[i] + p.dedt[i]);
        p.dedt[i] = 0.5*(predictor.dedt[i] + p.dedt[i]);
#endif
#if FRAGMENTATION
        p.d[i] = p.d[i] + dt/2 * (predictor.dddt[i] + p.dddt[i]);
        p.dddt[i] = 0.5*(predictor.dddt[i] + p.dddt[i]);
#endif
#if INTEGRATE_SML
        p.h[i] = p.h[i] + dt/2 * (predictor.dhdt[i] + p.dhdt[i]);
        p.dhdt[i] = 0.5 * (predictor.dhdt[i] + p.dhdt[i]);
#else
        p.h[i] = predictor.h[i];
#endif
#if JC_PLASTICITY
        p.T[i] = p.T[i] + dt/2 * (predictor.dTdt[i] + p.dTdt[i]);
        p.dTdt[i] = 0.5*(predictor.dTdt[i] + p.dTdt[i]);
#endif
#if FRAGMENTATION
# if PALPHA_POROSITY
//        if (p.drhodt[i] > 0 && predictor.p[i] > predictor.pold[i]) {
		if (predictor.p[i] > predictor.pold[i]) {
            p.damage_porjutzi[i] = p.damage_porjutzi[i] + dt/2 *(predictor.ddamage_porjutzidt[i] + p.ddamage_porjutzidt[i]);
        } else {
            p.d[i] = p.d[i];
            p.damage_porjutzi[i] = p.damage_porjutzi[i];
        }
# endif
        p.numActiveFlaws[i] = predictor.numActiveFlaws[i];
#endif
        // change drhodt after PALPHA_POROSITY
#if INTEGRATE_DENSITY
        p.rho[i] = p.rho[i] + dt/2 * (predictor.drhodt[i] + p.drhodt[i]);
        p.drhodt[i] = 0.5*(predictor.drhodt[i]+ p.drhodt[i]);
#else
        p.rho[i] = p.rho[i];
#endif
#if SIRONO_POROSITY
        p.rho_0prime[i] = p.rho_0prime[i];
        p.rho_c_plus[i] = p.rho_c_plus[i];
        p.rho_c_minus[i] = p.rho_c_minus[i];
        p.compressive_strength[i] = p.compressive_strength[i];
        p.tensile_strength[i] = p.tensile_strength[i];
        p.shear_strength[i] = p.shear_strength[i];
        p.K[i] = p.K[i];
        p.flag_rho_0prime[i] = p.flag_rho_0prime[i];
        p.flag_plastic[i] = p.flag_plastic[i];
#endif
#if EPSALPHA_POROSITY
        p.alpha_epspor[i] = p.alpha_epspor[i] + dt/2 * (predictor.dalpha_epspordt[i] + p.dalpha_epspordt[i]);
        p.dalpha_epspordt[i] = 0.5*(predictor.dalpha_epspordt[i] + p.dalpha_epspordt[i]);
        p.epsilon_v[i] = p.epsilon_v[i] + dt/2 * (predictor.depsilon_vdt[i] + p.depsilon_vdt[i]);
        p.depsilon_vdt[i] = 0.5*(predictor.depsilon_vdt[i] + p.depsilon_vdt[i]);
#endif
#if INVISCID_SPH
        p.beta[i] = p.beta[i] + dt/2 * (predictor.dbetadt[i] + p.dbetadt[i]);
        p.dbetadt[i] = 0.5 * (predictor.dbetadt[i] + p.dbetadt[i]);
#endif
#if SOLID
        for (j = 0; j < DIM; j++) {
            for (k = 0; k < DIM; k++) {
                p.S[stressIndex(i,j,k)] = p.S[stressIndex(i,j,k)] + dt/2  *
                    (predictor.dSdt[stressIndex(i,j,k)] + p.dSdt[stressIndex(i,j,k)]);
                p.dSdt[stressIndex(i,j,k)] = 0.5*(predictor.dSdt[stressIndex(i,j,k)] +
                        p.dSdt[stressIndex(i,j,k)]);
            }
        }
#if PALPHA_POROSITY
        /* check if we have compaction and change alpha accordingly */
//        if (p.drhodt[i] > 0 && predictor.p[i] > predictor.pold[i]) {
		if (predictor.p[i] > predictor.pold[i]) {
            p.alpha_jutzi[i] = p.alpha_jutzi[i] + dt/2 * ( p.dalphadt[i] + predictor.dalphadt[i]);
        } else {
            p.alpha_jutzi[i] = p.alpha_jutzi[i];
        }
        // next line, Oli & Christoph personal communication 2019-06-24
        p.p[i] = predictor.p[i];
        p.pold[i] = predictor.pold[i];
        p.alpha_jutzi_old[i] = p.alpha_jutzi_old[i];
#endif
        p.ep[i] = p.ep[i] + dt/2 * (predictor.edotp[i] + p.edotp[i]);
        p.edotp[i] = 0.5*(predictor.edotp[i] + p.edotp[i]);
#endif
    }
}

__global__ void PredictorStep_heun()
{
    register int i;
#if SOLID
    register int j;
    register int k;
#endif

    // particle loop
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        predictor.x[i] = p.x[i] + dt * p.dxdt[i];
        predictor.vx[i] = p.vx[i] + dt * p.ax[i];

#if DIM > 1
        predictor.y[i] = p.y[i] + dt * p.dydt[i];
        predictor.vy[i] = p.vy[i] + dt * p.ay[i];
#endif
#if DIM > 2
        predictor.z[i] = p.z[i] + dt * p.dzdt[i];
        predictor.vz[i] = p.vz[i] + dt * p.az[i];
#endif
#if INTEGRATE_DENSITY
        predictor.rho[i] = p.rho[i] + dt * p.drhodt[i];
#else
        predictor.rho[i] = p.rho[i];
#endif
        predictor.drhodt[i] = p.drhodt[i];
#if INTEGRATE_ENERGY
        predictor.e[i] = p.e[i] + dt * p.dedt[i];
#endif

#if INTEGRATE_SML
        predictor.h[i] = p.h[i] + dt * p.dhdt[i];
#else
        predictor.h[i] = p.h[i];
#endif

#if PALPHA_POROSITY
        predictor.alpha_jutzi[i] = p.alpha_jutzi[i] + dt * p.dalphadt[i];
        predictor.alpha_jutzi_old[i] = p.alpha_jutzi_old[i];
#endif

#if EPSALPHA_POROSITY
        predictor.alpha_epspor[i] = p.alpha_epspor[i] + dt * p.dalpha_epspordt[i];
        predictor.epsilon_v[i] = p.epsilon_v[i] + dt * p.depsilon_vdt[i];
#endif

#if FRAGMENTATION
        predictor.d[i] = p.d[i] + dt * p.dddt[i];
        predictor.numActiveFlaws[i] = p.numActiveFlaws[i];
# if PALPHA_POROSITY
        predictor.damage_porjutzi[i] = p.damage_porjutzi[i] + dt * p.ddamage_porjutzidt[i];
        // cms 2019-06-25
        // pressuremaxchange function will compare predictor.p - predictor.pold and
        // calc_pressure in integrator will overwrite predictor.p to the value *after* the test time
        // step
        //predictor.p[i] = p.p[i];
        predictor.pold[i] = p.p[i];
# endif
#endif
#if SIRONO_POROSITY
        predictor.rho_0prime[i] = p.rho_0prime[i];
        predictor.rho_c_plus[i] = p.rho_c_plus[i];
        predictor.rho_c_minus[i] = p.rho_c_minus[i];
        predictor.compressive_strength[i] = p.compressive_strength[i];
        predictor.tensile_strength[i] = p.tensile_strength[i];
        predictor.shear_strength[i] = p.shear_strength[i];
        predictor.K[i] = p.K[i];
        predictor.flag_rho_0prime[i] = p.flag_rho_0prime[i];
        predictor.flag_plastic[i] = p.flag_plastic[i];
#endif
#if JC_PLASTICITY
        predictor.T[i] = p.T[i] + dt * p.dTdt[i];
#endif
#if INVISCID_SPH
        predictor.beta[i] = p.beta[i] + dt * p.dbetadt[i];
#endif
#if SOLID
        predictor.ep[i] = p.ep[i] + dt * p.edotp[i];
        for (j = 0; j < DIM; j++) {
            for (k = 0; k < DIM; k++) {
                predictor.S[stressIndex(i,j,k)] = p.S[stressIndex(i,j,k)] + dt * p.dSdt[stressIndex(i,j,k)];
            }
        }
#endif
    }

}

#if PALPHA_POROSITY
/* check the pressure change to avoid large deviation from the crush-curve */
__global__ void pressureChangeCheck_heun(double *maxpressureDiffPerBlock)
{
    __shared__ double sharedMaxpressureDiff[NUM_THREADS_PC_INTEGRATOR];
    double localMaxpressureDiff = 0.0;
    double tmp = 0;
    int i, j, k, m;
    maxpressureDiff = 0.0;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        if (matEOS[p_rhs.materialId[i]] == EOS_TYPE_JUTZI || matEOS[p_rhs.materialId[i]] == EOS_TYPE_JUTZI_MURNAGHAN || matEOS[p_rhs.materialId[i]] == EOS_TYPE_JUTZI_ANEOS) {
        /* when this function is called, p points to predictor_device */
            // only consider pressure if the particle is compressed
            if (p.drhodt[i] > 0) {
                tmp = (p.p[i] - p.pold[i]);
            } else {
                tmp = 0;
            }
            localMaxpressureDiff = max(tmp, localMaxpressureDiff);
        }
    }
    i = threadIdx.x;
    sharedMaxpressureDiff[i] = localMaxpressureDiff;
    for (j = NUM_THREADS_PC_INTEGRATOR / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;
            sharedMaxpressureDiff[i] = localMaxpressureDiff = max(localMaxpressureDiff, sharedMaxpressureDiff[k]);
        }
    }
    // write block result to global memory
    if (i == 0) {
        k = blockIdx.x;
        maxpressureDiffPerBlock[k] = localMaxpressureDiff;
        m = gridDim.x - 1;
        if (m == atomicInc((unsigned int *)&blockCount, m)) {
            // last block, so combine all block results
            for (j = 0; j <= m; j++) {
                localMaxpressureDiff = max(localMaxpressureDiff, maxpressureDiffPerBlock[j]);
            }
            maxpressureDiff = localMaxpressureDiff;
            // reset block count
            blockCount = 0;
        }
        if (maxpressureDiff > max_abs_pressure_change) {
            printf("%e\n", maxpressureDiff);
            pressureChangeSmallEnough = FALSE;
            dt = 0.25 * dt;
            dt = min(dt, endTimeD - currentTimeD);
        } else {
            pressureChangeSmallEnough = TRUE;
        }
    }

}
#endif

__global__ void setTimestep_heun(double *forcesPerBlock, double *courantPerBlock, double *dtSPerBlock, double *dtePerBlock, double *dtrhoPerBlock, double *dtdamagePerBlock, double *dtalphaPerBlock, double *dtartviscPerBlock, double *dtbetaPerBlock, double *dtalpha_epsporPerBlock, double *dtepsilon_vPerBlock)
{

#define SAFETY_FIRST 0.1

    __shared__ double sharedForces[NUM_THREADS_LIMITTIMESTEP];
    __shared__ double sharedCourant[NUM_THREADS_LIMITTIMESTEP];
    __shared__ double sharedArtVisc[NUM_THREADS_LIMITTIMESTEP];
    __shared__ double sharedS[NUM_THREADS_LIMITTIMESTEP];
    __shared__ double sharede[NUM_THREADS_LIMITTIMESTEP];
    __shared__ double sharedrho[NUM_THREADS_LIMITTIMESTEP];
    __shared__ double shareddamage[NUM_THREADS_LIMITTIMESTEP];
    __shared__ double sharedalpha[NUM_THREADS_LIMITTIMESTEP];
    __shared__ double sharedbeta[NUM_THREADS_LIMITTIMESTEP];
    __shared__ double sharedalpha_epspor[NUM_THREADS_LIMITTIMESTEP];
    __shared__ double sharedepsilon_v[NUM_THREADS_LIMITTIMESTEP];

    int i, j, k, m;
    int d, dd;
    int index;
    int hasEnergy;
    double forces = DBL_MAX, courant = DBL_MAX;
    double dtx = DBL_MAX;
    double dtS = DBL_MAX;
    double dtrho = DBL_MAX;
    double dte = DBL_MAX;
    double dtdamage = DBL_MAX;
    double dtalpha = DBL_MAX;
    double dtbeta = DBL_MAX;
    double dtalpha_epspor = DBL_MAX;
    double dtepsilon_v = DBL_MAX;
    double temp;
    double sml;
    int matId;
#if SOLID
    double myS, dS;
#endif
    double ax, ay;
#if DIM == 3
    double az;
#endif
    double dtartvisc = DBL_MAX;

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        matId = p_rhs.materialId[i];
#if INTEGRATE_ENERGY
        hasEnergy = 0;

        switch  (matEOS[matId]) {
            case (EOS_TYPE_TILLOTSON):
                hasEnergy = 1;
                break;
            case (EOS_TYPE_JUTZI):
                hasEnergy = 1;
                break;
			case (EOS_TYPE_JUTZI_ANEOS):
				hasEnergy = 1;
				break;
            case (EOS_TYPE_SIRONO):
                hasEnergy = 1;
                break;
            case (EOS_TYPE_EPSILON):
                hasEnergy = 1;
                break;
            case (EOS_TYPE_ANEOS):
                hasEnergy = 1;
                break;
            default:
                hasEnergy = 0;
                break;
        }
#endif
        ax = p.ax[i];
#if DIM > 1
        ay = p.ay[i];
#endif
#if DIM == 3
        az = p.az[i];
#endif
        temp = ax*ax;
#if DIM > 1
        temp += + ay*ay;
#endif
#if DIM == 3
        temp += az*az;
#endif

        sml = p.h[i];
        temp = sqrt(sml / sqrt(temp));
        forces = min(forces, temp);
        temp = sml / p.cs[i];
        courant = min(courant, temp);

#if ARTIFICIAL_VISCOSITY
        temp = COURANT_FACT * sml / (p.cs[i] + 1.2 * (matAlpha[matId]) * p.cs[i] + matBeta[matId] * p.muijmax[i]);
        dtartvisc = min(dtartvisc, temp);
#endif
#if INVISCID_SPH
        if (p.dbetadt[i] != 0) {
            temp = SAFETY_FIRST * (fabs(p.beta[i])+betamin_d)/fabs(p.dbetadt[i]);
            dtbeta = min(temp, dtbeta);
        }

#endif
#if SOLID
        myS = 0;
        dS = 0;

        for (d = 0; d < DIM; d++) {
            for (dd = 0; dd < DIM; dd++) {
                index = i*DIM*DIM+d*DIM+dd;
                myS = fabs(p.S[index]);
                dS = fabs(p.dSdt[index]);
                if (dS > 0) {
                    temp = SAFETY_FIRST * (myS+Smin_d)/dS;
                    dtS = min(temp, dtS);
                }
            }
        }
#endif
#if INTEGRATE_DENSITY
        if (p.drhodt[i] != 0) {
            temp = SAFETY_FIRST * (fabs(p.rho[i])+rhomin_d)/fabs(p.drhodt[i]);
            dtrho = min(temp, dtrho);
        }
#endif
#if INTEGRATE_ENERGY
        if (p.dedt[i] != 0 && hasEnergy) {
            temp = SAFETY_FIRST * (fabs(p.e[i])+emin_d)/fabs(p.dedt[i]);
            dte = min(temp, dte);
        }
#endif

#if PALPHA_POROSITY
        if (p.dalphadt[i] != 0) {
            temp = 1.0e-2 / fabs(p.dalphadt[i]);
            dtalpha = min(temp, dtalpha);
        }
#endif

#if EPSALPHA_POROSITY
        if (p.dalpha_epspordt[i] != 0) {
            temp = 1.0e-1 / fabs(p.dalpha_epspordt[i]);
            dtalpha_epspor = min(temp, dtalpha_epspor);
        }

        if (p.depsilon_vdt[i] != 0) {
            temp = SAFETY_FIRST * (fabs(p.epsilon_v[i])+epsilon_vmin_d)/fabs(p.depsilon_vdt[i]);
            dtepsilon_v = min(temp, dtepsilon_v);
        }
#endif

#if FRAGMENTATION
        if (p.dddt[i] != 0) {
            temp = SAFETY_FIRST * (fabs(p.d[i])+damagemin_d)/fabs(p.dddt[i]);
            dtdamage = min(temp, dtdamage);
        }
#endif
    }

    i = threadIdx.x;
    sharedForces[i] = forces;
    sharedCourant[i] = courant;
    sharedS[i] = dtS;
    sharede[i] = dte;
    sharedrho[i] = dtrho;
    shareddamage[i] = dtdamage;
    sharedalpha[i] = dtalpha;
    sharedalpha_epspor[i] = dtalpha_epspor;
    sharedepsilon_v[i] = dtepsilon_v;
#if ARTIFICIAL_VISCOSITY
    sharedArtVisc[i] = dtartvisc;
#endif
#if INVISCID_SPH
    sharedbeta[i] = dtbeta;
#endif
    for (j = NUM_THREADS_LIMITTIMESTEP / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;
            sharedForces[i] = forces = min(forces, sharedForces[k]);
            sharedCourant[i] = courant = min(courant, sharedCourant[k]);
            sharedS[i] = dtS = min(dtS, sharedS[k]);
            sharede[i] = dte = min(dte, sharede[k]);
            sharedrho[i] = dtrho = min(dtrho, sharedrho[k]);
            shareddamage[i] = dtdamage = min(dtdamage, shareddamage[k]);
            sharedalpha[i] = dtalpha = min(dtalpha, sharedalpha[k]);
            sharedalpha_epspor[i] = dtalpha_epspor = min(dtalpha_epspor, sharedalpha_epspor[k]);
            sharedepsilon_v[i] = dtepsilon_v = min(dtepsilon_v, sharedepsilon_v[k]);
#if ARTIFICIAL_VISCOSITY
            sharedArtVisc[i] = dtartvisc = min(dtartvisc, sharedArtVisc[k]);
#endif
#if INVISCID_SPH
            sharedbeta[i] = dtbeta = min(dtbeta, sharedbeta[k]);
#endif
        }
    }
    // write block result to global memory
    if (i == 0) {
        k = blockIdx.x;
        forcesPerBlock[k] = forces;
        courantPerBlock[k] = courant;
        dtSPerBlock[k] = dtS;
        dtePerBlock[k] = dte;
        dtrhoPerBlock[k] = dtrho;
        dtdamagePerBlock[k] = dtdamage;
        dtalphaPerBlock[k] = dtalpha;
        dtalpha_epsporPerBlock[k] = dtalpha_epspor;
        dtepsilon_vPerBlock[k] = dtepsilon_v;
#if ARTIFICIAL_VISCOSITY
        dtartviscPerBlock[k] = dtartvisc;
#endif
#if INVISCID_SPH
        dtbetaPerBlock[k] = dtbeta;
#endif
        m = gridDim.x - 1;
        if (m == atomicInc((unsigned int *)&blockCount, m)) {
            // last block, so combine all block results
            for (j = 0; j <= m; j++) {
                forces = min(forces, forcesPerBlock[j]);
                courant = min(courant, courantPerBlock[j]);
                dtS = min(dtS, dtSPerBlock[j]);
                dte = min(dte, dtePerBlock[j]);
                dtrho = min(dtrho, dtrhoPerBlock[j]);
                dtdamage = min(dtdamage, dtdamagePerBlock[j]);
                dtalpha = min(dtalpha, dtalphaPerBlock[j]);
                dtalpha_epspor = min(dtalpha_epspor, dtalpha_epsporPerBlock[j]);
                dtepsilon_v = min(dtepsilon_v, dtepsilon_vPerBlock[j]);
#if ARTIFICIAL_VISCOSITY
                dtartvisc = min(dtartvisc, dtartviscPerBlock[j]);
#endif
#if INVISCID_SPH
                dtbeta = min(dtbeta, dtbetaPerBlock[j]);
#endif
            }
            // set new timestep
            dt = dtx = min(COURANT_FACT*courant, FORCES_FACT*forces);
#if SOLID
            dt = min(dt, dtS);
#endif
#if INTEGRATE_ENERGY
            dt = min(dt, dte);
#endif
#if INTEGRATE_DENSITY
            dt = min(dt, dtrho);
#endif
#if FRAGMENTATION
            dt = min(dt, dtdamage);
#endif
#if PALPHA_POROSITY
         //   dt = min(dt, dtalpha);
#endif
#if EPSALPHA_POROSITY
            dt = min(dt, dtalpha_epspor);
            dt = min(dt, dtepsilon_v);
#endif
#if ARTIFICIAL_VISCOSITY
            dt = min(dt, dtartvisc);
#endif
#if INVISCID_SPH
            dt = min(dt, dtbeta);
#endif
            dt = min(dt, endTimeD - currentTimeD);
            if (dt > dtmax) dt = dtmax;
            printf("Time Step Information: dt(v and x): %.17e ", dtx);
#if INTEGRATE_DENSITY
            printf("rho: %.17e ", dtrho);
#endif
#if INTEGRATE_ENERGY
            printf("e: %.17e ", dte);
#endif
#if SOLID
            printf("S: %.17e ", dtS);
#endif
#if FRAGMENTATION
            printf("damage: %.17e ", dtdamage);
#endif
#if PALPHA_POROSITY
            printf("distention: %.17e ", dtalpha);
#endif
#if EPSALPHA_POROSITY
            printf("distention: %.17e ", dtalpha_epspor);
            printf("epsilon: %.17e ", dtepsilon_v);
#endif
            printf("\n");
            printf("time: %.17e timestep set to %.17e, integrating until %.17e \n", currentTimeD, dt, endTimeD);
#if !PALPHA_POROSITY
            currentTimeD += dt;
#endif
			// reset block count
			blockCount = 0;
		}
	}
}




void heun_rk4()
{

    double *courantPerBlock, *forcesPerBlock;
    double *dtSPerBlock, *dtePerBlock, *dtrhoPerBlock;
    double *dtdamagePerBlock;
    double *dtalphaPerBlock;
    double *dtbetaPerBlock;
    double *dtartviscPerBlock;
    double *dtalpha_epsporPerBlock;
    double *dtepsilon_vPerBlock;
    double *maxpressureDiffPerBlock;
    int pressureChangeSmallEnough_host;


    cudaVerify(cudaMalloc((void**)&courantPerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&forcesPerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&dtSPerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&dtePerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&dtrhoPerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&dtdamagePerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&dtalphaPerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&dtbetaPerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&maxpressureDiffPerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&dtartviscPerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&dtalpha_epsporPerBlock, sizeof(double)*numberOfMultiprocessors));
    cudaVerify(cudaMalloc((void**)&dtepsilon_vPerBlock, sizeof(double)*numberOfMultiprocessors));

    int lastTimestep = startTimestep + numberOfTimesteps;
    int timestep;
    double maxpressureDiff_host;
    double maxpressureDiff_previous;
    int maxpressureDiff_cnt;
    double substep_currentTime;
    currentTime = startTime;
    double endTime = startTime;

    int allocate_immutables = 1;
    // alloc mem for one rhs
    allocate_particles_memory(&predictor_device, allocate_immutables);
    copy_particles_immutables_device_to_device(&predictor_device, &p_device);
    /* tell the gpu the current time */
    cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
    cudaVerify(cudaMemcpyToSymbol(predictor, &predictor_device, sizeof(struct Particle)));



#if GRAVITATING_POINT_MASSES
    int rkstep;
    int with_feedback;

    // alloc mem for multiple rhs and copy immutables
    for (rkstep = 0; rkstep < 4; rkstep++) {
        allocate_pointmass_memory(&rk4_pointmass_device[rkstep], allocate_immutables);
    }
    cudaVerify(cudaMemcpyToSymbol(rk4_pointmass, &rk4_pointmass_device, sizeof(struct Pointmass) * 4));
#endif


    for (timestep = startTimestep; timestep < lastTimestep; timestep++) {
        fprintf(stdout, "calculating step %d\n", timestep);
        printf("\nstep %d / %d\n", timestep, lastTimestep);
        endTime += timePerStep;
        fprintf(stdout, " currenttime: %e \t endtime: %e\n", currentTime, endTime);
        /* tell the gpu the time step */
        if (timePerStep > param.maxtimestep) {
            fprintf(stdout, "timestep was larger than maxtimestep given by -M, reducing to %e\n", param.maxtimestep);
            cudaVerify(cudaMemcpyToSymbol(dt, &param.maxtimestep, sizeof(double)));
        } else {
            cudaVerify(cudaMemcpyToSymbol(dt, &timePerStep, sizeof(double)));
        }


        // checking for changes in angular momentum
        if (param.angular_momentum_check > 0) {
            double L_current = calculate_angular_momentum();
            double L_change_relative;
            if (L_ini > 0) {
                L_change_relative = fabs((L_ini - L_current)/L_ini);
            }
            if (param.verbose) {
                fprintf(stdout, "Checking angular momentum conservation.\n");
                fprintf(stdout, "Initial angular momentum of the particles: %.17e\n", L_ini);
                fprintf(stdout, "Current angular momentum of the particles: %.17e\n", L_current);
                fprintf(stdout, "Relative change: %.17e\n", L_change_relative);
            }
            if (L_change_relative > param.angular_momentum_check) {
                fprintf(stderr, "Conservation of angular momentum violated. Exiting.\n");
                exit(111);
            }
        }


        /* tell the gpu the end time */
        cudaVerify(cudaMemcpyToSymbol(endTimeD, &endTime, sizeof(double)));

		while (currentTime < endTime) {
			cudaVerify(cudaDeviceSynchronize());
			// calculate first right hand side with p_device
	        cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
            cudaVerify(cudaDeviceSynchronize());
            cudaVerify(cudaMemcpyFromSymbol(&currentTime, currentTimeD, sizeof(double)));
            substep_currentTime = currentTime;
            cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));
#if GRAVITATING_POINT_MASSES
	        cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
#endif
            rightHandSide();
            cudaVerify(cudaDeviceSynchronize());
            cudaVerifyKernel((setTimestep_heun<<<numberOfMultiprocessors, NUM_THREADS_LIMITTIMESTEP>>>(
                              forcesPerBlock, courantPerBlock,
                              dtSPerBlock, dtePerBlock, dtrhoPerBlock, dtdamagePerBlock,
                              dtalphaPerBlock, dtartviscPerBlock, dtbetaPerBlock, dtalpha_epsporPerBlock, dtepsilon_vPerBlock)));
            cudaVerify(cudaDeviceSynchronize());
            /* get the time and the time step from the gpu */
            cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
#if GRAVITATING_POINT_MASSES
            // calculate disk feedback if wanted
            backreaction_from_disk_to_point_masses(TRUE);
#endif
			cudaVerify(cudaDeviceSynchronize());
            pressureChangeSmallEnough_host = FALSE;
            maxpressureDiff_cnt = 0;
            maxpressureDiff_host = 0;
            maxpressureDiff_previous = 0;
            while (pressureChangeSmallEnough_host == FALSE) {
	            // do the predictor step (writes to predictor)
                printf("First step with time step: %e at time: %e.\n", dt_host, currentTime);
	            cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
#if GRAVITATING_POINT_MASSES
                // fix this for rk4
	            cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
#endif
    	        cudaVerifyKernel((PredictorStep_heun<<<numberOfMultiprocessors, NUM_THREADS_PC_INTEGRATOR>>>()));
			    cudaVerify(cudaDeviceSynchronize());


#define DISABLE_PRESSURE_CHECK_POROSITY_FOR_THE_TIME_BEING 0


#if !DISABLE_PRESSURE_CHECK_POROSITY_FOR_THE_TIME_BEING
#if PALPHA_POROSITY
                /* check if the step was too large */
                /* check the pressure at predictor step */
		        cudaVerify(cudaMemcpyToSymbol(p, &predictor_device, sizeof(struct Particle)));
				cudaVerifyKernel((calculatePressure<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
    			cudaVerify(cudaDeviceSynchronize());
			    cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
				printf("before pressure change check: dt_host: %e\n", dt_host);
				cudaVerifyKernel((pressureChangeCheck_heun<<<numberOfMultiprocessors, NUM_THREADS_PC_INTEGRATOR>>>(maxpressureDiffPerBlock)));
    			cudaVerify(cudaDeviceSynchronize());
                cudaVerify(cudaMemcpyFromSymbol(&pressureChangeSmallEnough_host, pressureChangeSmallEnough, sizeof(int)));
                cudaVerify(cudaMemcpyFromSymbol(&maxpressureDiff_host, maxpressureDiff, sizeof(double)));
#else
                pressureChangeSmallEnough_host = TRUE;
#endif // PALPHA_POROSITY
#else
                pressureChangeSmallEnough_host = TRUE;
#endif // !DISABLE_PRESSURE_CHECK_POROSITY_FOR_THE_TIME_BEING


			    cudaVerify(cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double)));
				printf("after pressure change check: dt_host: %e\n", dt_host);
				printf("pressureChangeSmallEnough_host: %d\n", pressureChangeSmallEnough_host);
				if (pressureChangeSmallEnough_host == FALSE) {
                    /* redo predictor step with smaller timestep, derivatives are in p_device */
					printf("Reducing timestep due to Pressure Check function to: %.17e\n", dt_host);
                    if (fabs(maxpressureDiff_host -maxpressureDiff_previous) < 1e-3) {
                        maxpressureDiff_cnt++;
                    }
                    maxpressureDiff_previous = maxpressureDiff_host;
                    if (maxpressureDiff_cnt > 1) {
                        printf("Cannot reduce timestep anymore, continuing with dt %.17e and maxpressurediff %.17e", dt_host, maxpressureDiff_host);
                        pressureChangeSmallEnough_host = TRUE;
                    }
				}
                if (pressureChangeSmallEnough_host == TRUE) {
                    /* okay, step seems good, let's do the corrector step */
                    substep_currentTime = currentTime + dt_host;
					currentTime += dt_host;
					printf("Timestep okay, continuing with NBODY integration rk4.\n");
                        // -> rk4 integration with back reaction if wanted
#if GRAVITATING_POINT_MASSES
	                cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
                    rk4_nbodies();
#endif
					cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
    	            if (param.selfgravity) {
        	            copy_gravitational_accels_device_to_device(&predictor_device, &p_device);
                    }

                    /* now righthandside with predictor variables */
		            cudaVerify(cudaMemcpyToSymbol(p, &predictor_device, sizeof(struct Particle)));
#if GRAVITATING_POINT_MASSES
	                cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
#endif
				    rightHandSide();
                    /* now the corrector step  with the original values of p_device and the derivatives of p_device and predictor_device */
		            cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
#if GRAVITATING_POINT_MASSES
	                cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
#endif
    	            cudaVerifyKernel((CorrectorStep_heun<<<numberOfMultiprocessors, NUM_THREADS_PC_INTEGRATOR>>>()));
		    //step was successful --> do something (e.g. look for min/max pressure...)
                    afterIntegrationStep();

				}
			}
		} // current time < end time loop
		// write results
#if FRAGMENTATION
        cudaVerify(cudaDeviceSynchronize());
        cudaVerifyKernel((damageLimit<<<numberOfMultiprocessors*4, NUM_THREADS_PC_INTEGRATOR>>>()));
        cudaVerify(cudaDeviceSynchronize());
#endif
        copyToHostAndWriteToFile(timestep, lastTimestep);

	} // timestep loop

	// free memory

    int free_immutables = 1;
#if GRAVITATING_POINT_MASSES
    for (rkstep = 0; rkstep < 4; rkstep++) {
        free_pointmass_memory(&rk4_pointmass_device[rkstep], free_immutables);
        }
#endif

    free_particles_memory(&predictor_device, free_immutables);
#if GRAVITATING_POINT_MASSES
    free_pointmass_memory(&predictor_pointmass_device, free_immutables);
#endif
	cudaVerify(cudaFree(courantPerBlock));
	cudaVerify(cudaFree(forcesPerBlock));
    cudaVerify(cudaFree(dtSPerBlock));
	cudaVerify(cudaFree(dtePerBlock));
	cudaVerify(cudaFree(dtrhoPerBlock));
	cudaVerify(cudaFree(dtdamagePerBlock));
    cudaVerify(cudaFree(dtalphaPerBlock));
    cudaVerify(cudaFree(dtbetaPerBlock));
    cudaVerify(cudaFree(dtalpha_epsporPerBlock));
    cudaVerify(cudaFree(dtepsilon_vPerBlock));
    cudaVerify(cudaFree(dtartviscPerBlock));
    cudaVerify(cudaFree(maxpressureDiffPerBlock));
}
