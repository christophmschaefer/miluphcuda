/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com
 *
 * @section     LICENSE
 * Copyright (c) 2026 Christoph Schaefer
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

/*
 * Kick-drift-kick (leapfrog) integrator with one rhs per step, see kdk.h.
 *
 * The rates of SPH depend on the velocities (artificial viscosity, strain rates) and on the other integrated
 * quantities, so the rhs at the end of a step is evaluated at the predicted state v* = v^n + dt a^n,
 * Q* = Q^n + dt dQ/dt^n (as in Phantom, Price et al. 2018, PASA 35, e031). The second kick is then applied as
 *     q^{n+1} = q* + dt/2 (dq/dt^{n+1} - dq/dt^n)
 * to the state left by the rhs, which equals v^{n+1/2} + dt/2 a^{n+1} for the velocities. Written this way, all
 * changes that the rhs makes to the state itself (stress limiting by the plasticity models, damage limit, return
 * mapping of alpha, boundary conditions, smoothing length corrections) are kept, as with the other integrators,
 * where the rhs at the start of a step modifies the state from which the step starts.
 *
 * The derivatives of the end of a step are those of the start of the next step (first same as last), so one
 * rhs per step remains. For the second kick, the derivatives of the start of the step are copied to kdk_old,
 * which holds only these arrays. Positions are changed only by the drift.
 */

#include "kdk.h"
#include "miluph.h"
#include "timeintegration.h"
#include "config_parameter.h"
#include "parameter.h"
#include "porosity.h"
#include "pressure.h"
#include "rhs.h"
#include "damage.h"
#include "fast_integration.h"
#include <float.h>
#include <string.h>

extern __device__ double endTimeD, currentTimeD;
extern __device__ double substep_currentTimeD;
extern __device__ double dt;

extern double L_ini;

/* derivatives at the start of the step, for the second kick. Only the arrays listed in kdk_old_arrays() are
   allocated, all other pointers of the structs are NULL. */
__constant__ struct Particle kdk_old;
struct Particle kdk_old_device;
#if GRAVITATING_POINT_MASSES
__constant__ struct Pointmass kdk_old_pointmass;
struct Pointmass kdk_old_pointmass_device;
#endif

/* time step limits */
enum {
    KDK_LIMIT_COURANT,
    KDK_LIMIT_FORCES,
    KDK_LIMIT_DAMAGE,
    KDK_LIMIT_ALPHA,
    KDK_NUM_LIMITS
};
static const char *kdk_limit_names[KDK_NUM_LIMITS] = {"courant", "forces", "damage", "alpha"};



/* first kick, drift, and prediction of the state at the end of the step (at which the rhs is evaluated) */
__global__ void kdkKickDriftPredict(void)
{
    int i;
#if SOLID
    int k;
#endif
#if XSPH
    int matId;
#endif

#if GRAVITATING_POINT_MASSES
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i += blockDim.x * gridDim.x) {
        pointmass.x[i] += dt * (pointmass.vx[i] + 0.5 * dt * pointmass.ax[i]);
        pointmass.vx[i] += dt * pointmass.ax[i];
# if DIM > 1
        pointmass.y[i] += dt * (pointmass.vy[i] + 0.5 * dt * pointmass.ay[i]);
        pointmass.vy[i] += dt * pointmass.ay[i];
# endif
# if DIM > 2
        pointmass.z[i] += dt * (pointmass.vz[i] + 0.5 * dt * pointmass.az[i]);
        pointmass.vz[i] += dt * pointmass.az[i];
# endif
    }
#endif

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += blockDim.x * gridDim.x) {
        /* drift with v^{n+1/2} = v^n + dt/2 a^n */
        p.x[i] += dt * (p.vx[i] + 0.5 * dt * p.ax[i]);
#if DIM > 1
        p.y[i] += dt * (p.vy[i] + 0.5 * dt * p.ay[i]);
#endif
#if DIM > 2
        p.z[i] += dt * (p.vz[i] + 0.5 * dt * p.az[i]);
#endif
#if XSPH
        /* XSPH correction of the last rhs, as in setlocationchanges() in velocity.cu */
        matId = p_rhs.materialId[i];
        if (matId != EOS_TYPE_IGNORE && !(matId >= 0 && matEOS[matId] == EOS_TYPE_IGNORE)) {
            p.x[i] += dt * 0.5 * p.xsphvx[i];
# if DIM > 1
            p.y[i] += dt * 0.5 * p.xsphvy[i];
# endif
# if DIM > 2
            p.z[i] += dt * 0.5 * p.xsphvz[i];
# endif
        }
#endif

        /* predicted state v* = v^n + dt a^n, Q* = Q^n + dt dQ/dt^n */
        p.vx[i] += dt * p.ax[i];
#if DIM > 1
        p.vy[i] += dt * p.ay[i];
#endif
#if DIM > 2
        p.vz[i] += dt * p.az[i];
#endif
#if INTEGRATE_DENSITY
        p.rho[i] += dt * p.drhodt[i];
#endif
#if INTEGRATE_ENERGY
        p.e[i] += dt * p.dedt[i];
#endif
#if INTEGRATE_SML
        p.h[i] += dt * p.dhdt[i];
#endif
#if PALPHA_POROSITY
        p.alpha_jutzi[i] += dt * p.dalphadt[i];
#endif
#if FRAGMENTATION
        p.d[i] += dt * p.dddt[i];
# if PALPHA_POROSITY
        p.damage_porjutzi[i] += dt * p.ddamage_porjutzidt[i];
# endif
#endif
#if JC_PLASTICITY
        p.T[i] += dt * p.dTdt[i];
#endif
#if INVISCID_SPH
        p.beta[i] += dt * p.dbetadt[i];
#endif
#if EPSALPHA_POROSITY
        p.alpha_epspor[i] += dt * p.dalpha_epspordt[i];
        p.epsilon_v[i] += dt * p.depsilon_vdt[i];
#endif
#if SOLID
        for (k = 0; k < DIM*DIM; k++) {
            p.S[i*DIM*DIM + k] += dt * p.dSdt[i*DIM*DIM + k];
        }
        p.ep[i] += dt * p.edotp[i];
        p.eps_tot[i] += dt * p.deps_totdt[i];
#endif
    }
}



/* second kick: q^{n+1} = q* + dt/2 (dq/dt^{n+1} - dq/dt^n), with the derivatives of the start of the step in kdk_old */
__global__ void kdkKick(void)
{
    int i;
#if SOLID
    int k;
#endif

#if GRAVITATING_POINT_MASSES
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i += blockDim.x * gridDim.x) {
        pointmass.vx[i] += 0.5 * dt * (pointmass.ax[i] - kdk_old_pointmass.ax[i]);
# if DIM > 1
        pointmass.vy[i] += 0.5 * dt * (pointmass.ay[i] - kdk_old_pointmass.ay[i]);
# endif
# if DIM > 2
        pointmass.vz[i] += 0.5 * dt * (pointmass.az[i] - kdk_old_pointmass.az[i]);
# endif
    }
#endif

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += blockDim.x * gridDim.x) {
        p.vx[i] += 0.5 * dt * (p.ax[i] - kdk_old.ax[i]);
#if DIM > 1
        p.vy[i] += 0.5 * dt * (p.ay[i] - kdk_old.ay[i]);
#endif
#if DIM > 2
        p.vz[i] += 0.5 * dt * (p.az[i] - kdk_old.az[i]);
#endif
#if INTEGRATE_DENSITY
        p.rho[i] += 0.5 * dt * (p.drhodt[i] - kdk_old.drhodt[i]);
#endif
#if INTEGRATE_ENERGY
        p.e[i] += 0.5 * dt * (p.dedt[i] - kdk_old.dedt[i]);
#endif
#if INTEGRATE_SML
        p.h[i] += 0.5 * dt * (p.dhdt[i] - kdk_old.dhdt[i]);
#endif
#if PALPHA_POROSITY
        p.alpha_jutzi[i] += 0.5 * dt * (p.dalphadt[i] - kdk_old.dalphadt[i]);
#endif
#if FRAGMENTATION
        p.d[i] += 0.5 * dt * (p.dddt[i] - kdk_old.dddt[i]);
# if PALPHA_POROSITY
        p.damage_porjutzi[i] += 0.5 * dt * (p.ddamage_porjutzidt[i] - kdk_old.ddamage_porjutzidt[i]);
# endif
#endif
#if JC_PLASTICITY
        p.T[i] += 0.5 * dt * (p.dTdt[i] - kdk_old.dTdt[i]);
#endif
#if INVISCID_SPH
        p.beta[i] += 0.5 * dt * (p.dbetadt[i] - kdk_old.dbetadt[i]);
#endif
#if EPSALPHA_POROSITY
        p.alpha_epspor[i] += 0.5 * dt * (p.dalpha_epspordt[i] - kdk_old.dalpha_epspordt[i]);
        p.epsilon_v[i] += 0.5 * dt * (p.depsilon_vdt[i] - kdk_old.depsilon_vdt[i]);
#endif
#if SOLID
        for (k = 0; k < DIM*DIM; k++) {
            p.S[i*DIM*DIM + k] += 0.5 * dt * (p.dSdt[i*DIM*DIM + k] - kdk_old.dSdt[i*DIM*DIM + k]);
        }
        p.ep[i] += 0.5 * dt * (p.edotp[i] - kdk_old.edotp[i]);
        p.eps_tot[i] += 0.5 * dt * (p.deps_totdt[i] - kdk_old.deps_totdt[i]);
#endif
    }
}



/* time step limits from the derivatives of the current state, one result per block and limit in limitsPerBlock */
__global__ void kdkLimitTimestep(double *limitsPerBlock)
{
    __shared__ double shared[KDK_NUM_LIMITS][NUM_THREADS_LIMITTIMESTEP];
    double lim[KDK_NUM_LIMITS];
    double tmp, signal;
    int i, j, l, matId;

    for (l = 0; l < KDK_NUM_LIMITS; l++) {
        lim[l] = DBL_MAX;
    }

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += blockDim.x * gridDim.x) {
        matId = p_rhs.materialId[i];
        /* CFL condition for particles that interact, as limitTimestepCourant() in rk2adaptive.cu, with the
           signal speed of the artificial viscosity as in setTimestep() in predictor_corrector.cu */
        if (p.noi[i] > 0 && matId >= 0) {
#if SOLID
            signal = sqrt(p.cs[i]*p.cs[i] + 4./3*matShearmodulus[matId]/p.rho[i]);
#else
            signal = p.cs[i];
#endif
#if ARTIFICIAL_VISCOSITY
            signal += 1.2 * matAlpha[matId] * p.cs[i] + matBeta[matId] * p.muijmax[i];
#endif
            if (signal > 0.0) {
                lim[KDK_LIMIT_COURANT] = min(lim[KDK_LIMIT_COURANT], KDK_COURANT_FACT * p.h[i] / signal);
            }
        }
#if KDK_USE_FORCES_LIMIT
        tmp = p.ax[i]*p.ax[i];
# if DIM > 1
        tmp += p.ay[i]*p.ay[i];
# endif
# if DIM > 2
        tmp += p.az[i]*p.az[i];
# endif
        if (tmp > 0.0) {
            lim[KDK_LIMIT_FORCES] = min(lim[KDK_LIMIT_FORCES], FORCES_FACT * sqrt(p.h[i] / sqrt(tmp)));
        }
#endif
#if KDK_USE_DAMAGE_LIMIT && FRAGMENTATION
        /* as limitTimestepDamage() in rk2adaptive.cu */
        if (p.dddt[i] > 0.0) {
            tmp = 0.7 * (p.d[i] + KDK_MAX_DAMAGE_CHANGE) / p.dddt[i];
            tmp = min(tmp, KDK_MAX_DAMAGE_CHANGE / p.dddt[i]);
            lim[KDK_LIMIT_DAMAGE] = min(lim[KDK_LIMIT_DAMAGE], tmp);
        }
#endif
#if PALPHA_POROSITY && !PALPHA_RETURN_MAPPING
        if (p.dalphadt[i] != 0.0) {
            lim[KDK_LIMIT_ALPHA] = min(lim[KDK_LIMIT_ALPHA], KDK_MAX_ALPHA_CHANGE / fabs(p.dalphadt[i]));
        }
#endif
    }

    /* reduce to one result per block */
    i = threadIdx.x;
    for (l = 0; l < KDK_NUM_LIMITS; l++) {
        shared[l][i] = lim[l];
    }
    for (j = NUM_THREADS_LIMITTIMESTEP / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            for (l = 0; l < KDK_NUM_LIMITS; l++) {
                shared[l][i] = lim[l] = min(lim[l], shared[l][i + j]);
            }
        }
    }
    if (i == 0) {
        for (l = 0; l < KDK_NUM_LIMITS; l++) {
            limitsPerBlock[blockIdx.x * KDK_NUM_LIMITS + l] = lim[l];
        }
    }
}



/* time step for the next step and the limit that sets it, from the per-block results of kdkLimitTimestep() */
static double kdk_timestep(double *limitsPerBlock_d, double *limitsPerBlock, double dt_upper, int *limit)
{
    int b, l;
    double dt_new = dt_upper;

    cudaVerifyKernel((kdkLimitTimestep<<<numberOfMultiprocessors, NUM_THREADS_LIMITTIMESTEP>>>(limitsPerBlock_d)));
    cudaVerify(cudaDeviceSynchronize());
    cudaVerify(cudaMemcpy(limitsPerBlock, limitsPerBlock_d, numberOfMultiprocessors * KDK_NUM_LIMITS * sizeof(double),
                cudaMemcpyDeviceToHost));
    *limit = -1;
    for (b = 0; b < numberOfMultiprocessors; b++) {
        for (l = 0; l < KDK_NUM_LIMITS; l++) {
            if (limitsPerBlock[b * KDK_NUM_LIMITS + l] < dt_new) {
                dt_new = limitsPerBlock[b * KDK_NUM_LIMITS + l];
                *limit = l;
            }
        }
    }
    return dt_new;
}



enum {
    KDK_OLD_ALLOCATE,
    KDK_OLD_COPY,
    KDK_OLD_FREE
};

/* allocates, fills (from p_device and pointmass_device) or frees the arrays of kdk_old_device and
   kdk_old_pointmass_device. The list has to contain exactly the derivatives used by kdkKick(). */
static void kdk_old_arrays(int op)
{
#define KDK_OLD_ARRAY(dst, src, field, size) \
    do { \
        if (op == KDK_OLD_ALLOCATE) \
            cudaVerify(cudaMalloc((void**)&(dst).field, size)); \
        else if (op == KDK_OLD_COPY) \
            cudaVerify(cudaMemcpy((dst).field, (src).field, size, cudaMemcpyDeviceToDevice)); \
        else \
            cudaVerify(cudaFree((dst).field)); \
    } while (0)
#define KDK_OLD(field, size) KDK_OLD_ARRAY(kdk_old_device, p_device, field, size)

    if (op == KDK_OLD_ALLOCATE) {
        memset(&kdk_old_device, 0, sizeof(struct Particle));
    }
    KDK_OLD(ax, memorySizeForParticles);
#if DIM > 1
    KDK_OLD(ay, memorySizeForParticles);
#endif
#if DIM > 2
    KDK_OLD(az, memorySizeForParticles);
#endif
#if INTEGRATE_DENSITY
    KDK_OLD(drhodt, memorySizeForParticles);
#endif
#if INTEGRATE_ENERGY
    KDK_OLD(dedt, memorySizeForParticles);
#endif
#if INTEGRATE_SML
    KDK_OLD(dhdt, memorySizeForParticles);
#endif
#if PALPHA_POROSITY
    KDK_OLD(dalphadt, memorySizeForParticles);
#endif
#if FRAGMENTATION
    KDK_OLD(dddt, memorySizeForParticles);
# if PALPHA_POROSITY
    KDK_OLD(ddamage_porjutzidt, memorySizeForParticles);
# endif
#endif
#if JC_PLASTICITY
    KDK_OLD(dTdt, memorySizeForParticles);
#endif
#if INVISCID_SPH
    KDK_OLD(dbetadt, memorySizeForParticles);
#endif
#if EPSALPHA_POROSITY
    KDK_OLD(dalpha_epspordt, memorySizeForParticles);
    KDK_OLD(depsilon_vdt, memorySizeForParticles);
#endif
#if SOLID
    KDK_OLD(dSdt, memorySizeForStress);
    KDK_OLD(edotp, memorySizeForParticles);
    KDK_OLD(deps_totdt, memorySizeForParticles);
#endif

#if GRAVITATING_POINT_MASSES
# define KDK_OLD_PM(field) KDK_OLD_ARRAY(kdk_old_pointmass_device, pointmass_device, field, memorySizeForPointmasses)
    if (op == KDK_OLD_ALLOCATE) {
        memset(&kdk_old_pointmass_device, 0, sizeof(struct Pointmass));
    }
    KDK_OLD_PM(ax);
# if DIM > 1
    KDK_OLD_PM(ay);
# endif
# if DIM > 2
    KDK_OLD_PM(az);
# endif
# undef KDK_OLD_PM
#endif

#undef KDK_OLD
#undef KDK_OLD_ARRAY

    if (op == KDK_OLD_ALLOCATE) {
        cudaVerify(cudaMemcpyToSymbol(kdk_old, &kdk_old_device, sizeof(struct Particle)));
#if GRAVITATING_POINT_MASSES
        cudaVerify(cudaMemcpyToSymbol(kdk_old_pointmass, &kdk_old_pointmass_device, sizeof(struct Pointmass)));
#endif
    }
}



#if FAST_INTEGRATION_SCHEME
/* check_fast_scheme(), returns the number of materials that switched */
static int kdk_check_fast_scheme(void)
{
    int m, before = 0, after = 0;
    int *switched = (int *) malloc(numberOfMaterials * sizeof(int));

    assert(switched != NULL);
    cudaVerify(cudaMemcpy(switched, matFastSwitched_d, numberOfMaterials * sizeof(int), cudaMemcpyDeviceToHost));
    for (m = 0; m < numberOfMaterials; m++) {
        before += switched[m];
    }
    check_fast_scheme();
    cudaVerify(cudaMemcpy(switched, matFastSwitched_d, numberOfMaterials * sizeof(int), cudaMemcpyDeviceToHost));
    for (m = 0; m < numberOfMaterials; m++) {
        after += switched[m];
    }
    free(switched);
    return after - before;
}
#endif



void kdk(void)
{
    int lastTimestep = startTimestep + numberOfTimesteps;
    int timestep;
    int limit;
    int first_step = TRUE;
    unsigned int nsteps = 0, nsteps_total = 0;
    double endTime = startTime;
    double dt_upper;
    double substep_currentTime;
    double *limitsPerBlock_d, *limitsPerBlock;

    assert(param.maxtimestep > 0.0);

    cudaVerify(cudaMalloc((void**)&limitsPerBlock_d, numberOfMultiprocessors * KDK_NUM_LIMITS * sizeof(double)));
    limitsPerBlock = (double *) malloc(numberOfMultiprocessors * KDK_NUM_LIMITS * sizeof(double));
    assert(limitsPerBlock != NULL);

    /* memory for the derivatives at the start of a step */
    kdk_old_arrays(KDK_OLD_ALLOCATE);

    /* the rhs is always evaluated on p_device */
    cudaVerify(cudaMemcpyToSymbol(p, &p_device, sizeof(struct Particle)));
#if GRAVITATING_POINT_MASSES
    cudaVerify(cudaMemcpyToSymbol(pointmass, &pointmass_device, sizeof(struct Pointmass)));
#endif

    currentTime = startTime;
    cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
    substep_currentTime = currentTime;
    cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));

    /* derivatives of the initial state, the rhs also commits its distention for PALPHA_RETURN_MAPPING */
#if FAST_INTEGRATION_SCHEME
    kdk_check_fast_scheme();
#endif
    rightHandSide();
    cudaVerify(cudaDeviceSynchronize());
#if PALPHA_POROSITY && PALPHA_RETURN_MAPPING
    /* from now on, the rhs is evaluated at predicted states, the history is committed at the end of each step */
    palpha_commit_pending = FALSE;
#endif

    for (timestep = startTimestep; timestep < lastTimestep; timestep++) {
        endTime += timePerStep;
        assert(endTime > currentTime);
        cudaVerify(cudaMemcpyToSymbol(endTimeD, &endTime, sizeof(double)));
        fprintf(stdout, "\n\nStart integrating output step %d / %d from time %g to %g...\n",
                timestep+1, lastTimestep, currentTime, endTime);
        nsteps = 0;

        // checking for changes in angular momentum
        if (param.angular_momentum_check > 0) {
            double L_current = calculate_angular_momentum();
            double L_change_relative;
            if (L_ini > 0) {
                L_change_relative = fabs((L_ini - L_current)/L_ini);
            }
            if (param.verbose) {
                fprintf(stdout, "Checking angular momentum conservation.\n");
                fprintf(stdout, "Initial angular momentum: %.17e\n", L_ini);
                fprintf(stdout, "Current angular momentum: %.17e\n", L_current);
                fprintf(stdout, "Relative change: %.17e\n", L_change_relative);
            }
            if (L_change_relative > param.angular_momentum_check) {
                fprintf(stderr, "Conservation of angular momentum violated. Exiting.\n");
                exit(111);
            }
        }

        while (currentTime < endTime) {
#if FAST_INTEGRATION_SCHEME
            /* a switch changes the material parameters, the derivatives of the current state have to follow */
            if (kdk_check_fast_scheme() > 0) {
                if (param.verbose)
                    fprintf(stdout, "fast integration scheme switched, recomputing the derivatives\n");
                rightHandSide();
                cudaVerify(cudaDeviceSynchronize());
            }
#endif
            /* time step from the derivatives of the current state */
            dt_upper = fmin(param.maxtimestep, endTime - currentTime);
            if (first_step && param.firsttimestep > 0.0)
                dt_upper = fmin(dt_upper, param.firsttimestep);
            dt_host = kdk_timestep(limitsPerBlock_d, limitsPerBlock, dt_upper, &limit);
            if (dt_host < KDK_SMALLEST_DT && dt_host < endTime - currentTime) {
                fprintf(stderr, "Timestep %e is below KDK_SMALLEST_DT. Stopping here.\n", dt_host);
                exit(1);
            }
            cudaVerify(cudaMemcpyToSymbol(dt, &dt_host, sizeof(double)));

            /* first kick, drift, and prediction of the end of the step */
            kdk_old_arrays(KDK_OLD_COPY);
            cudaVerifyKernel((kdkKickDriftPredict<<<numberOfMultiprocessors * 4, NUM_THREADS_KDK_INTEGRATOR>>>()));
            cudaVerify(cudaDeviceSynchronize());

            /* derivatives at the end of the step */
            substep_currentTime = currentTime + dt_host;
            cudaVerify(cudaMemcpyToSymbol(substep_currentTimeD, &substep_currentTime, sizeof(double)));
            rightHandSide();
            cudaVerify(cudaDeviceSynchronize());

            /* second kick */
            cudaVerifyKernel((kdkKick<<<numberOfMultiprocessors * 4, NUM_THREADS_KDK_INTEGRATOR>>>()));
            cudaVerify(cudaDeviceSynchronize());
#if PALPHA_POROSITY && PALPHA_RETURN_MAPPING
            /* distention of the final state of the step, committed as history */
            palpha_return_mapping(TRUE);
#endif

            /* land exactly on the output time */
            if (dt_host >= endTime - currentTime)
                currentTime = endTime;
            else
                currentTime += dt_host;
            cudaVerify(cudaMemcpyToSymbol(currentTimeD, &currentTime, sizeof(double)));
            first_step = FALSE;
            nsteps++;
            if (param.verbose) {
                fprintf(stdout, "kdk: t = %.15e  dt = %.15e  limited by %s  time to next output: %g\n",
                        currentTime, dt_host, limit >= 0 ? kdk_limit_names[limit] : "max or output time",
                        endTime - currentTime);
            }

            afterIntegrationStep();
#if PALPHA_POROSITY && PALPHA_RETURN_MAPPING
            /* the next rhs is evaluated at a predicted state, which is not committed */
            palpha_commit_pending = FALSE;
#endif
        }
        nsteps_total += nsteps;
        fprintf(stdout, "Finished integrating output step %d / %d with %u timesteps (%u in total).\n",
                timestep+1, lastTimestep, nsteps, nsteps_total);

        // write results
#if FRAGMENTATION
        cudaVerifyKernel((damageLimit<<<numberOfMultiprocessors*4, NUM_THREADS_KDK_INTEGRATOR>>>()));
        cudaVerify(cudaDeviceSynchronize());
#endif
        copyToHostAndWriteToFile(timestep, lastTimestep);
    }

    kdk_old_arrays(KDK_OLD_FREE);
    cudaVerify(cudaFree(limitsPerBlock_d));
    free(limitsPerBlock);
}



void print_kdk_settings(void)
{
    fprintf(stdout, "\nkdk integrator settings:\n");
    fprintf(stdout, "    KDK_COURANT_FACT: %g (always used)\n", KDK_COURANT_FACT);
    fprintf(stdout, "    KDK_USE_FORCES_LIMIT: %d (FORCES_FACT: %g)\n", KDK_USE_FORCES_LIMIT, FORCES_FACT);
#if FRAGMENTATION
    fprintf(stdout, "    KDK_USE_DAMAGE_LIMIT: %d (KDK_MAX_DAMAGE_CHANGE: %g)\n", KDK_USE_DAMAGE_LIMIT,
            KDK_MAX_DAMAGE_CHANGE);
#endif
#if PALPHA_POROSITY && !PALPHA_RETURN_MAPPING
    fprintf(stdout, "    KDK_MAX_ALPHA_CHANGE: %g\n", KDK_MAX_ALPHA_CHANGE);
#endif
    fprintf(stdout, "    KDK_SMALLEST_DT: %g\n", KDK_SMALLEST_DT);
}
