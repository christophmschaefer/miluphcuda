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
#include "memory_handling.h"
#include "aneos.h"


/* allocate memory on the device for pointmasses */
int allocate_pointmass_memory(struct Pointmass *a, int allocate_immutables)
{
    int rc = 0;

	cudaVerify(cudaMalloc((void**)&a->x, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&a->vx, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&a->ax, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&a->feedback_ax, memorySizeForPointmasses));
#if DIM > 1
	cudaVerify(cudaMalloc((void**)&a->y, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&a->vy, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&a->ay, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&a->feedback_ay, memorySizeForPointmasses));
# if DIM > 2
	cudaVerify(cudaMalloc((void**)&a->z, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&a->vz, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&a->az, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&a->feedback_az, memorySizeForPointmasses));
# endif
#endif
	cudaVerify(cudaMalloc((void**)&a->m, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&a->rmin, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&a->rmax, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&a->feels_particles, integermemorySizeForPointmasses));

    return rc;
}



/* allocate memory on the device for particles */
int allocate_particles_memory(struct Particle *a, int allocate_immutables)
{
    int rc = 0;

#if TENSORIAL_CORRECTION
    // also moved to p_device only
//	cudaVerify(cudaMalloc((void**)&a->tensorialCorrectionMatrix, memorySizeForStress));
    // not needed anymore, let's save memory --- tschakka!
/*    if (allocate_immutables) {
        cudaVerify(cudaMalloc((void**)&a->tensorialCorrectiondWdrr, MAX_NUM_INTERACTIONS * maxNumberOfParticles * sizeof(double)));
    } */
#endif

#if INTEGRATE_ENERGY
	cudaVerify(cudaMalloc((void**)&a->dedt, memorySizeForParticles));
#endif

#if ARTIFICIAL_VISCOSITY
	cudaVerify(cudaMalloc((void**)&a->muijmax, memorySizeForParticles));
#endif

	cudaVerify(cudaMalloc((void**)&a->drhodt, memorySizeForParticles));

#if SOLID
	cudaVerify(cudaMalloc((void**)&a->S, memorySizeForStress));
	cudaVerify(cudaMalloc((void**)&a->dSdt, memorySizeForStress));
	cudaVerify(cudaMalloc((void**)&a->local_strain, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->ep, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->edotp, memorySizeForParticles));
#endif

#if NAVIER_STOKES
	cudaVerify(cudaMalloc((void**)&a->Tshear, memorySizeForStress));
#endif

#if INVISCID_SPH
	cudaVerify(cudaMalloc((void**)&a->beta, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->beta_old, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->divv_old, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->dbetadt, memorySizeForParticles));
#endif

#if FRAGMENTATION
	memorySizeForActivationThreshold = maxNumberOfParticles * MAX_NUM_FLAWS * sizeof(double);
	cudaVerify(cudaMalloc((void**)&a->d, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->damage_total, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->dddt, memorySizeForParticles));

	cudaVerify(cudaMalloc((void**)&a->numFlaws, memorySizeForInteractions));
	cudaVerify(cudaMalloc((void**)&a->numActiveFlaws, memorySizeForInteractions));
    if (allocate_immutables) {
	    cudaVerify(cudaMalloc((void**)&a->flaws, memorySizeForActivationThreshold));
    }
# if PALPHA_POROSITY
	cudaVerify(cudaMalloc((void**)&a->damage_porjutzi, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->ddamage_porjutzidt, memorySizeForParticles));
# endif
#endif

    if (allocate_immutables) {
        cudaVerify(cudaMalloc((void**)&a->h0, memorySizeForParticles));
    }

#if GHOST_BOUNDARIES
	cudaVerify(cudaMalloc((void**)&a->real_partner, memorySizeForInteractions));
#endif

#if PALPHA_POROSITY
	cudaVerify(cudaMalloc((void**)&a->pold, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->alpha_jutzi, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->alpha_jutzi_old, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->dalphadt, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->dp, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->dalphadp, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->dalphadrho, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->delpdelrho, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->delpdele, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->f, memorySizeForParticles));
#endif

#if SIRONO_POROSITY
    cudaVerify(cudaMalloc((void**)&a->compressive_strength, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->tensile_strength, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->shear_strength, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->K, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->rho_0prime, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->rho_c_plus, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->rho_c_minus, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->flag_rho_0prime, memorySizeForInteractions));
    cudaVerify(cudaMalloc((void**)&a->flag_plastic, memorySizeForInteractions));
#endif

#if EPSALPHA_POROSITY
    cudaVerify(cudaMalloc((void**)&a->alpha_epspor, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->dalpha_epspordt, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->epsilon_v, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->depsilon_vdt, memorySizeForParticles));
#endif

    cudaVerify(cudaMalloc((void**)&a->x0, memorySizeForTree));
#if DIM > 1
    cudaVerify(cudaMalloc((void**)&a->y0, memorySizeForTree));
#if DIM > 2
    cudaVerify(cudaMalloc((void**)&a->z0, memorySizeForTree));
#endif
#endif
	cudaVerify(cudaMalloc((void**)&a->x, memorySizeForTree));
#if DIM > 1
	cudaVerify(cudaMalloc((void**)&a->y, memorySizeForTree));
#endif
	cudaVerify(cudaMalloc((void**)&a->vx, memorySizeForParticles));
#if DIM > 1
	cudaVerify(cudaMalloc((void**)&a->vy, memorySizeForParticles));
#endif
	cudaVerify(cudaMalloc((void**)&a->dxdt, memorySizeForParticles));
#if DIM > 1
 	cudaVerify(cudaMalloc((void**)&a->dydt, memorySizeForParticles));
#endif

#if XSPH
	cudaVerify(cudaMalloc((void**)&a->xsphvx, memorySizeForParticles));
#if DIM > 1
	cudaVerify(cudaMalloc((void**)&a->xsphvy, memorySizeForParticles));
#endif
#endif
	cudaVerify(cudaMalloc((void**)&a->ax, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->g_ax, memorySizeForParticles));
#if DIM > 1
	cudaVerify(cudaMalloc((void**)&a->ay, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->g_ay, memorySizeForParticles));
#endif
	cudaVerify(cudaMalloc((void**)&a->m, memorySizeForTree));
	cudaVerify(cudaMalloc((void**)&a->h, memorySizeForParticles));
#if INTEGRATE_SML
	cudaVerify(cudaMalloc((void**)&a->dhdt, memorySizeForParticles));
#endif

#if SML_CORRECTION
	cudaVerify(cudaMalloc((void**)&a->sml_omega, memorySizeForParticles));
#endif

	cudaVerify(cudaMalloc((void**)&a->rho, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->p, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->e, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->cs, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->noi, memorySizeForInteractions));
	cudaVerify(cudaMalloc((void**)&a->depth, memorySizeForInteractions));
#if MORE_OUTPUT
	cudaVerify(cudaMalloc((void**)&a->p_min, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->p_max, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->rho_min, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->rho_max, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->e_min, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->e_max, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->cs_min, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&a->cs_max, memorySizeForParticles));
#endif
// moved to p_device only, so we don't need mem here anymore
//	cudaVerify(cudaMalloc((void**)&a->materialId, memorySizeForInteractions));

#if JC_PLASTICITY
	cudaVerify(cudaMalloc((void**)&a->T, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->dTdt, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->jc_f, memorySizeForParticles));
#endif

#if DIM > 2
	cudaVerify(cudaMalloc((void**)&a->z, memorySizeForTree));
	cudaVerify(cudaMalloc((void**)&a->dzdt, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->vz, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->az, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&a->g_az, memorySizeForParticles));
#if XSPH
	cudaVerify(cudaMalloc((void**)&a->xsphvz, memorySizeForParticles));
#endif
#endif
	cudaVerify(cudaMemset(a->ax, 0, memorySizeForParticles));
	cudaVerify(cudaMemset(a->g_ax, 0, memorySizeForParticles));
#if DIM > 1
	cudaVerify(cudaMemset(a->ay, 0, memorySizeForParticles));
	cudaVerify(cudaMemset(a->g_ay, 0, memorySizeForParticles));
#if DIM == 3
	cudaVerify(cudaMemset(a->az, 0, memorySizeForParticles));
	cudaVerify(cudaMemset(a->g_az, 0, memorySizeForParticles));
#endif
#endif

    return rc;
}



int copy_gravitational_accels_device_to_device(struct Particle *dst, struct Particle *src)
{
    int rc = 0;
    cudaVerify(cudaMemcpy(dst->g_ax, src->g_ax, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#if DIM > 1
    cudaVerify(cudaMemcpy(dst->g_ay, src->g_ay, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#if DIM > 2
    cudaVerify(cudaMemcpy(dst->g_az, src->g_az, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif
#endif

    return rc;
}



int copy_pointmass_derivatives_device_to_device(struct Pointmass *dst, struct Pointmass *src)
{
    int rc = 0;
    cudaVerify(cudaMemcpy(dst->ax, src->ax, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->vx, src->vx, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->feedback_ax, src->feedback_ax, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
#if DIM > 1
    cudaVerify(cudaMemcpy(dst->ay, src->ay, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->vy, src->vy, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->feedback_ay, src->feedback_ay, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
# if DIM > 2
    cudaVerify(cudaMemcpy(dst->az, src->az, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->vz, src->vz, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->feedback_az, src->feedback_az, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
# endif
#endif

    return rc;
}



int copy_particles_derivatives_device_to_device(struct Particle *dst, struct Particle *src)
{
    int rc = 0;

    cudaVerify(cudaMemcpy(dst->ax, src->ax, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->g_ax, src->g_ax, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->dxdt, src->dxdt, memorySizeForParticles, cudaMemcpyDeviceToDevice));

#if DIM > 1
    cudaVerify(cudaMemcpy(dst->ay, src->ay, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->g_ay, src->g_ay, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->dydt, src->dydt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#if DIM > 2
    cudaVerify(cudaMemcpy(dst->az, src->az, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->g_az, src->g_az, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->dzdt, src->dzdt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif
#endif

    cudaVerify(cudaMemcpy(dst->drhodt, src->drhodt, memorySizeForParticles, cudaMemcpyDeviceToDevice));

#if INTEGRATE_SML
    cudaVerify(cudaMemcpy(dst->dhdt, src->dhdt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif

#if SML_CORRECTION
    cudaVerify(cudaMemcpy(dst->sml_omega, src->sml_omega, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif

#if PALPHA_POROSITY
    cudaVerify(cudaMemcpy(dst->dalphadt, src->dalphadt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#if FRAGMENTATION
    cudaVerify(cudaMemcpy(dst->ddamage_porjutzidt, src->ddamage_porjutzidt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif
#endif

#if EPSALPHA_POROSITY
    cudaVerify(cudaMemcpy(dst->dalpha_epspordt, src->dalpha_epspordt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->depsilon_vdt, src->depsilon_vdt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif

#if INTEGRATE_ENERGY
    cudaVerify(cudaMemcpy(dst->dedt, src->dedt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif

#if SOLID
    cudaVerify(cudaMemcpy(dst->dSdt, src->dSdt, memorySizeForStress, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->edotp, src->edotp, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif

#if INVISCID_SPH
	cudaVerify(cudaMemcpy(dst->dbetadt, src->dbetadt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif

#if JC_PLASTICITY
    cudaVerify(cudaMemcpy(dst->dTdt, src->dTdt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif

#if FRAGMENTATION
    cudaVerify(cudaMemcpy(dst->dddt, src->dddt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->numActiveFlaws, src->numActiveFlaws, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
#endif

    return rc;
}



int copy_pointmass_immutables_device_to_device(struct Pointmass *dst, struct Pointmass *src)
{
    int rc = 0;

    cudaVerify(cudaMemcpy((*dst).m, (*src).m, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy((*dst).feels_particles, (*src).feels_particles, integermemorySizeForPointmasses, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy((*dst).rmin, (*src).rmin, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy((*dst).rmax, (*src).rmax, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));

    return rc;
}



int copy_particles_immutables_device_to_device(struct Particle *dst, struct Particle *src)
{
    int rc = 0;

    cudaVerify(cudaMemcpy((*dst).x0, (*src).x0, memorySizeForTree, cudaMemcpyDeviceToDevice));
#if DIM > 1
    cudaVerify(cudaMemcpy((*dst).y0, (*src).y0, memorySizeForTree, cudaMemcpyDeviceToDevice));
#endif
#if DIM > 2
    cudaVerify(cudaMemcpy((*dst).z0, (*src).z0, memorySizeForTree, cudaMemcpyDeviceToDevice));
#endif
    cudaVerify(cudaMemcpy((*dst).m, (*src).m, memorySizeForTree, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy((*dst).h, (*src).h, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy((*dst).cs, (*src).cs, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    //cudaVerify(cudaMemcpy((*dst).materialId, (*src).materialId, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
#if FRAGMENTATION
	cudaVerify(cudaMemcpy(dst->numFlaws, src->numFlaws, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
    //cudaVerify(cudaMemcpy(dst->flaws, src->flaws, memorySizeForActivationThreshold, cudaMemcpyDeviceToDevice));
#endif

    return rc;
}



int copy_pointmass_variables_device_to_device(struct Pointmass *dst, struct Pointmass *src)
{
    int rc = 0;
    cudaVerify(cudaMemcpy(dst->x, src->x, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
    // mass is variable
    cudaVerify(cudaMemcpy(dst->m, src->m, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->vx, src->vx, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
#if DIM > 1
    cudaVerify(cudaMemcpy(dst->y, src->y, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->vy, src->vy, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
# if DIM > 2
    cudaVerify(cudaMemcpy(dst->z, src->z, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->vz, src->vz, memorySizeForPointmasses, cudaMemcpyDeviceToDevice));
# endif
#endif

    return rc;
}



int copy_particles_variables_device_to_device(struct Particle *dst, struct Particle *src)
{
    int rc = 0;

    cudaVerify(cudaMemcpy(dst->x, src->x, memorySizeForTree, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->x0, src->x0, memorySizeForTree, cudaMemcpyDeviceToDevice));
    // materialId moved to p_device aka p_rhs only
    //cudaVerify(cudaMemcpy((*dst).materialId, (*src).materialId, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
#if DIM > 1
    cudaVerify(cudaMemcpy(dst->y, src->y, memorySizeForTree, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->y0, src->y0, memorySizeForTree, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->vy, src->vy, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif
#if DIM > 2
    cudaVerify(cudaMemcpy(dst->z0, src->z0, memorySizeForTree, cudaMemcpyDeviceToDevice));
#endif

    cudaVerify(cudaMemcpy(dst->vx, src->vx, memorySizeForParticles, cudaMemcpyDeviceToDevice));

    cudaVerify(cudaMemcpy(dst->rho, src->rho, memorySizeForParticles, cudaMemcpyDeviceToDevice));

    cudaVerify(cudaMemcpy(dst->h, src->h, memorySizeForParticles, cudaMemcpyDeviceToDevice));

#if INTEGRATE_ENERGY
    cudaVerify(cudaMemcpy(dst->e, src->e, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif

#if PALPHA_POROSITY
    cudaVerify(cudaMemcpy(dst->alpha_jutzi, src->alpha_jutzi, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->alpha_jutzi_old, src->alpha_jutzi, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->dalphadp, src->dalphadp, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->dalphadrho, src->dalphadrho, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->dp, src->dp, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->delpdelrho, src->delpdelrho, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->delpdele, src->delpdele, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->f, src->f, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->p, src->p, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->pold, src->pold, memorySizeForParticles, cudaMemcpyDeviceToDevice));
# if FRAGMENTATION
    cudaVerify(cudaMemcpy(dst->damage_porjutzi, src->damage_porjutzi, memorySizeForParticles, cudaMemcpyDeviceToDevice));
# endif
#endif

#if MORE_OUTPUT
    cudaVerify(cudaMemcpy(dst->p_min, src->p_min, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->p_max, src->p_max, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->rho_min, src->rho_min, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->rho_max, src->rho_max, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->e_min, src->e_min, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->e_max, src->e_max, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->cs_min, src->cs_min, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->cs_max, src->cs_max, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif

#if SIRONO_POROSITY
    cudaVerify(cudaMemcpy(dst->compressive_strength, src->compressive_strength, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->tensile_strength, src->tensile_strength, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->shear_strength, src->shear_strength, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->K, src->K, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->rho_0prime, src->rho_0prime, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->rho_c_plus, src->rho_c_plus, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->rho_c_minus, src->rho_c_minus, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->flag_rho_0prime, src->flag_rho_0prime, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->flag_plastic, src->flag_plastic, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
#endif

#if EPSALPHA_POROSITY
    cudaVerify(cudaMemcpy(dst->alpha_epspor, src->alpha_epspor, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->dalpha_epspordt, src->dalpha_epspordt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->epsilon_v, src->epsilon_v, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->depsilon_vdt, src->depsilon_vdt, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif

#if DIM > 2
    cudaVerify(cudaMemcpy(dst->z, src->z, memorySizeForTree, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->vz, src->vz, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif
#if SOLID
    cudaVerify(cudaMemcpy(dst->S, src->S, memorySizeForStress, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->ep, src->ep, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif
#if NAVIER_STOKES
    cudaVerify(cudaMemcpy(dst->Tshear, src->Tshear, memorySizeForStress, cudaMemcpyDeviceToDevice));
#endif

#if INVISCID_SPH
    cudaVerify(cudaMemcpy(dst->beta, src->beta, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->beta_old, src->beta_old, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->divv_old, src->divv_old, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif

#if JC_PLASTICITY
    cudaVerify(cudaMemcpy(dst->T, src->T, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->jc_f, src->jc_f, memorySizeForParticles, cudaMemcpyDeviceToDevice));
#endif

#if FRAGMENTATION
    cudaVerify(cudaMemcpy(dst->d, src->d, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->damage_total, src->damage_total, memorySizeForParticles, cudaMemcpyDeviceToDevice));
    cudaVerify(cudaMemcpy(dst->numActiveFlaws, src->numActiveFlaws, memorySizeForInteractions, cudaMemcpyDeviceToDevice));
#endif

    return rc;
}



/* free runge-kutta memory for pointmasses on the device */
int free_pointmass_memory(struct Pointmass *a, int free_immutables)
{
    int rc = 0;
	cudaVerify(cudaFree(a->x));
	cudaVerify(cudaFree(a->vx));
	cudaVerify(cudaFree(a->ax));
	cudaVerify(cudaFree(a->feedback_ax));
	cudaVerify(cudaFree(a->m));
	cudaVerify(cudaFree(a->feels_particles));
	cudaVerify(cudaFree(a->rmin));
	cudaVerify(cudaFree(a->rmax));
#if DIM > 1
	cudaVerify(cudaFree(a->y));
	cudaVerify(cudaFree(a->vy));
	cudaVerify(cudaFree(a->ay));
	cudaVerify(cudaFree(a->feedback_ay));
# if DIM > 2
	cudaVerify(cudaFree(a->z));
	cudaVerify(cudaFree(a->vz));
	cudaVerify(cudaFree(a->az));
	cudaVerify(cudaFree(a->feedback_az));
# endif
#endif

    return rc;
}



/* free runge-kutta memory on the device */
int free_particles_memory(struct Particle *a, int free_immutables)
{
    int rc = 0;

	cudaVerify(cudaFree(a->x));
	cudaVerify(cudaFree(a->x0));
	cudaVerify(cudaFree(a->dxdt));
	cudaVerify(cudaFree(a->vx));
	cudaVerify(cudaFree(a->ax));
	cudaVerify(cudaFree(a->g_ax));
	cudaVerify(cudaFree(a->m));
#if DIM > 1
	cudaVerify(cudaFree(a->dydt));
	cudaVerify(cudaFree(a->y));
	cudaVerify(cudaFree(a->y0));
	cudaVerify(cudaFree(a->vy0));
	cudaVerify(cudaFree(a->vy));
	cudaVerify(cudaFree(a->ay));
	cudaVerify(cudaFree(a->g_ay));
#endif

#if XSPH
	cudaVerify(cudaFree(a->xsphvx));
#if DIM > 1
	cudaVerify(cudaFree(a->xsphvy));
#endif
#endif
	cudaVerify(cudaFree(a->h));
	cudaVerify(cudaFree(a->rho));
	cudaVerify(cudaFree(a->p));
	cudaVerify(cudaFree(a->e));
	cudaVerify(cudaFree(a->cs));
	cudaVerify(cudaFree(a->noi));
	cudaVerify(cudaFree(a->depth));
#if MORE_OUTPUT
	cudaVerify(cudaFree(a->p_min));
	cudaVerify(cudaFree(a->p_max));
	cudaVerify(cudaFree(a->rho_min));
	cudaVerify(cudaFree(a->rho_max));
	cudaVerify(cudaFree(a->e_min));
	cudaVerify(cudaFree(a->e_max));
	cudaVerify(cudaFree(a->cs_min));
	cudaVerify(cudaFree(a->cs_max));
#endif
    // materialId only on p_device
	//cudaVerify(cudaFree(a->materialId));
#if DIM > 2
	cudaVerify(cudaFree(a->z));
	cudaVerify(cudaFree(a->z0));
	cudaVerify(cudaFree(a->dzdt));
	cudaVerify(cudaFree(a->vz));
#if XSPH
	cudaVerify(cudaFree(a->xsphvz));
#endif
	cudaVerify(cudaFree(a->az));
	cudaVerify(cudaFree(a->g_az));
#endif


#if ARTIFICIAL_VISCOSITY
	cudaVerify(cudaFree(a->muijmax));
#endif
#if (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH || INTEGRATE_ENERGY)
	cudaVerify(cudaFree(a->divv));
	cudaVerify(cudaFree(a->curlv));
#endif

#if INVISCID_SPH
	cudaVerify(cudaFree(a->beta));
	cudaVerify(cudaFree(a->beta_old));
	cudaVerify(cudaFree(a->divv_old));
	cudaVerify(cudaFree(a->dbetadt));
#endif

#if TENSORIAL_CORRECTION
	//cudaVerify(cudaFree(a->tensorialCorrectionMatrix));
    /*
    if (free_immutables) {
	    cudaVerify(cudaFree(a->tensorialCorrectiondWdrr));
    } */
#endif

#if INTEGRATE_ENERGY
	cudaVerify(cudaFree(a->dedt));
#endif
#if GHOST_BOUNDARIES
	cudaVerify(cudaFree(a->real_partner));
#endif

	cudaVerify(cudaFree(a->drhodt));

#if INTEGRATE_SML
	cudaVerify(cudaFree(a->dhdt));
#endif

#if SML_CORRECTION
    cudaVerify(cudaFree(a->sml_omega));
#endif

#if SOLID
	cudaVerify(cudaFree(a->S));
	cudaVerify(cudaFree(a->dSdt));
	cudaVerify(cudaFree(a->local_strain));
    cudaVerify(cudaFree(a->ep));
    cudaVerify(cudaFree(a->edotp));
#endif
#if NAVIER_STOKES
	cudaVerify(cudaFree(a->Tshear));
#endif

#if JC_PLASTICITY
	cudaVerify(cudaFree(a->T));
	cudaVerify(cudaFree(a->dTdt));
	cudaVerify(cudaFree(a->jc_f));
#endif

#if PALPHA_POROSITY
	cudaVerify(cudaFree(a->pold));
	cudaVerify(cudaFree(a->alpha_jutzi));
	cudaVerify(cudaFree(a->alpha_jutzi_old));
	cudaVerify(cudaFree(a->dalphadt));
	cudaVerify(cudaFree(a->f));
	cudaVerify(cudaFree(a->dalphadp));
	cudaVerify(cudaFree(a->dp));
	cudaVerify(cudaFree(a->delpdelrho));
	cudaVerify(cudaFree(a->delpdele));
	cudaVerify(cudaFree(a->dalphadrho));
#endif

#if SIRONO_POROSITY
    cudaVerify(cudaFree(a->compressive_strength));
    cudaVerify(cudaFree(a->tensile_strength));
    cudaVerify(cudaFree(a->shear_strength));
    cudaVerify(cudaFree(a->K));
    cudaVerify(cudaFree(a->rho_0prime));
    cudaVerify(cudaFree(a->rho_c_plus));
    cudaVerify(cudaFree(a->rho_c_minus));
    cudaVerify(cudaFree(a->flag_rho_0prime));
    cudaVerify(cudaFree(a->flag_plastic));
#endif

#if EPSALPHA_POROSITY
    cudaVerify(cudaFree(a->alpha_epspor));
    cudaVerify(cudaFree(a->dalpha_epspordt));
    cudaVerify(cudaFree(a->epsilon_v));
    cudaVerify(cudaFree(a->depsilon_vdt));
#endif

#if FRAGMENTATION
	cudaVerify(cudaFree(a->d));
	cudaVerify(cudaFree(a->damage_total));
	cudaVerify(cudaFree(a->dddt));
	cudaVerify(cudaFree(a->numFlaws));
	cudaVerify(cudaFree(a->numActiveFlaws));
    if (free_immutables) {
	    cudaVerify(cudaFree(a->flaws));
    }
    if (free_immutables) {
	    cudaVerify(cudaFree(a->h0));
    }
# if PALPHA_POROSITY
	cudaVerify(cudaFree(a->damage_porjutzi));
	cudaVerify(cudaFree(a->ddamage_porjutzidt));
# endif
#endif

    return rc;
}



/* allocate memory for tree and basic particle struct */
int init_allocate_memory(void)
{
    int rc = 0;
    // where does the 2.5 come from?
	// numberOfNodes = ceil(2.5 * maxNumberOfParticles);
	numberOfNodes = ceil(3.5 * maxNumberOfParticles);

    if (numberOfNodes < 1024*numberOfMultiprocessors)
        numberOfNodes = 1024*numberOfMultiprocessors;

	if (param.verbose) {
		fprintf(stdout, "Allocating memory for %d nodes of tree...\n", numberOfNodes);
	}

	numberOfChildren = 8; // always 8 children per node
	numberOfRealParticles = maxNumberOfParticles;
	if (param.verbose) {
		fprintf(stdout, "Allocating memory for %d particles...\n", numberOfRealParticles);
	}

#define WARPSIZE 32
    
    while ((numberOfNodes & (WARPSIZE-1)) != 0) numberOfNodes++;

	if (param.verbose) {
		fprintf(stdout, "After checking with WARPSIZE of %d, allocating memory for %d nodes of tree...\n", WARPSIZE, numberOfNodes); 
	}

	if (param.verbose) {
        fprintf(stdout, "\nAllocating memory for %d particles...\n", numberOfParticles);
	    fprintf(stdout, "Allocating memory for %d pointmasses...\n", numberOfPointmasses);
        fprintf(stdout, "Number of nodes of tree: %d\n", numberOfNodes);
		fprintf(stdout, "Allocating memory for maximum of %d children.\n", numberOfChildren * (numberOfNodes-numberOfRealParticles));
    }

	memorySizeForParticles = maxNumberOfParticles * sizeof(double);
	memorySizeForPointmasses = numberOfPointmasses * sizeof(double);
	integermemorySizeForPointmasses = numberOfPointmasses * sizeof(int);
	memorySizeForTree = numberOfNodes * sizeof(double);
	memorySizeForStress = maxNumberOfParticles * DIM * DIM * sizeof(double);
	memorySizeForChildren = numberOfChildren * (numberOfNodes-numberOfRealParticles) * sizeof(int);
	memorySizeForInteractions = maxNumberOfParticles * sizeof(int);

	if (param.verbose) {
		fprintf(stdout, "Memory size for particles: %d bytes\n", memorySizeForParticles);
		fprintf(stdout, "Memory size for pointmasses: %d bytes\n", memorySizeForPointmasses);
		fprintf(stdout, "Memory size for tree: %d bytes\n", memorySizeForTree);
		fprintf(stdout, "Memory size for stress: %d bytes\n", memorySizeForStress);
		fprintf(stdout, "Memory size for children: %d bytes\n", memorySizeForChildren);
		fprintf(stdout, "Memory size for interactions: %d bytes\n", memorySizeForInteractions);
	}

	cudaVerify(cudaMalloc((void**)&p_device.x0, memorySizeForTree));


    cudaVerify(cudaMallocHost((void**)&p_host.x, memorySizeForTree));
	cudaVerify(cudaMallocHost((void**)&p_host.vx, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.ax, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.g_ax, memorySizeForParticles));
#if DIM > 1
    cudaVerify(cudaMallocHost((void**)&p_host.y, memorySizeForTree));
	cudaVerify(cudaMallocHost((void**)&p_host.vy, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.ay, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.g_ay, memorySizeForParticles));
#endif
#if DIM > 2
    cudaVerify(cudaMallocHost((void**)&p_host.z, memorySizeForTree));
    cudaVerify(cudaMallocHost((void**)&p_host.vz, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.az, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.g_az, memorySizeForParticles));
#endif
    cudaVerify(cudaMallocHost((void**)&p_host.m, memorySizeForTree));
    cudaVerify(cudaMallocHost((void**)&p_host.h, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.rho, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.p, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.e, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.cs, memorySizeForParticles));

#if GRAVITATING_POINT_MASSES
	cudaVerify(cudaMallocHost((void**)&pointmass_host.x, memorySizeForPointmasses));
	cudaVerify(cudaMallocHost((void**)&pointmass_host.vx, memorySizeForPointmasses));
	cudaVerify(cudaMallocHost((void**)&pointmass_host.ax, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.x, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.vx, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.ax, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.feedback_ax, memorySizeForPointmasses));
#if DIM > 1
	cudaVerify(cudaMallocHost((void**)&pointmass_host.y, memorySizeForPointmasses));
	cudaVerify(cudaMallocHost((void**)&pointmass_host.vy, memorySizeForPointmasses));
	cudaVerify(cudaMallocHost((void**)&pointmass_host.ay, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.y, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.vy, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.ay, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.feedback_ay, memorySizeForPointmasses));
#if DIM > 2
	cudaVerify(cudaMallocHost((void**)&pointmass_host.z, memorySizeForPointmasses));
	cudaVerify(cudaMallocHost((void**)&pointmass_host.vz, memorySizeForPointmasses));
	cudaVerify(cudaMallocHost((void**)&pointmass_host.az, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.z, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.vz, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.az, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.feedback_az, memorySizeForPointmasses));
#endif
#endif
	cudaVerify(cudaMallocHost((void**)&pointmass_host.rmin, memorySizeForPointmasses));
	cudaVerify(cudaMallocHost((void**)&pointmass_host.rmax, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.rmin, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.rmax, memorySizeForPointmasses));
	cudaVerify(cudaMallocHost((void**)&pointmass_host.m, memorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.m, memorySizeForPointmasses));
	cudaVerify(cudaMallocHost((void**)&pointmass_host.feels_particles, integermemorySizeForPointmasses));
	cudaVerify(cudaMalloc((void**)&pointmass_device.feels_particles, integermemorySizeForPointmasses));
#endif

#if MORE_OUTPUT
	cudaVerify(cudaMallocHost((void**)&p_host.p_min, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.p_max, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.rho_min, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.rho_max, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.e_min, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.e_max, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.cs_min, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.cs_max, memorySizeForParticles));
#endif

	cudaVerify(cudaMallocHost((void**)&p_host.noi, memorySizeForInteractions));
	cudaVerify(cudaMallocHost((void**)&p_host.depth, memorySizeForInteractions));
	cudaVerify(cudaMallocHost((void**)&interactions_host, memorySizeForInteractions*MAX_NUM_INTERACTIONS));
	cudaVerify(cudaMallocHost((void**)&p_host.materialId, memorySizeForInteractions));
	cudaVerify(cudaMallocHost((void**)&childList_host, memorySizeForChildren));

#if ARTIFICIAL_VISCOSITY
	cudaVerify(cudaMalloc((void**)&p_device.muijmax, memorySizeForParticles));
#endif

#if (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH || INTEGRATE_ENERGY)
	cudaVerify(cudaMalloc((void**)&p_device.divv, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.curlv, memorySizeForParticles*DIM));
#endif

#if INVISCID_SPH
	cudaVerify(cudaMalloc((void**)&p_device.beta, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.beta_old, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.divv_old, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.dbetadt, memorySizeForParticles));
#endif

#if TENSORIAL_CORRECTION
	cudaVerify(cudaMalloc((void**)&p_device.tensorialCorrectionMatrix, memorySizeForStress));
	//cudaVerify(cudaMalloc((void**)&p_device.tensorialCorrectiondWdrr, MAX_NUM_INTERACTIONS * maxNumberOfParticles * sizeof(double)));
#endif

#if SHEPARD_CORRECTION
	cudaVerify(cudaMalloc((void**)&p_device.shepard_correction, memorySizeForParticles));
#endif

#if INTEGRATE_ENERGY
	cudaVerify(cudaMallocHost((void**)&p_host.dedt, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.dedt, memorySizeForParticles));
#endif

	cudaVerify(cudaMallocHost((void**)&p_host.drhodt, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.drhodt, memorySizeForParticles));

#if SOLID
	cudaVerify(cudaMallocHost((void**)&p_host.S, memorySizeForStress));
	cudaVerify(cudaMallocHost((void**)&p_host.dSdt, memorySizeForStress));
	cudaVerify(cudaMalloc((void**)&p_device.S, memorySizeForStress));
	cudaVerify(cudaMalloc((void**)&p_device.dSdt, memorySizeForStress));
	cudaVerify(cudaMallocHost((void**)&p_host.local_strain, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.local_strain, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**) &p_device.sigma, memorySizeForStress));
    cudaVerify(cudaMalloc((void**)&p_device.plastic_f, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.ep, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.ep, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.edotp, memorySizeForParticles));
#endif

#if NAVIER_STOKES
	cudaVerify(cudaMallocHost((void**)&p_host.Tshear, memorySizeForStress));
	cudaVerify(cudaMalloc((void**)&p_device.Tshear, memorySizeForStress));
	cudaVerify(cudaMalloc((void**)&p_device.eta, memorySizeForParticles));
#endif

#if ARTIFICIAL_STRESS
	cudaVerify(cudaMalloc((void**) &p_device.R, memorySizeForStress));
#endif

#if JC_PLASTICITY
	cudaVerify(cudaMalloc((void**)&p_device.T, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.T, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.dTdt, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.jc_f, memorySizeForParticles));
#endif

#if FRAGMENTATION
	memorySizeForActivationThreshold = maxNumberOfParticles * MAX_NUM_FLAWS * sizeof(double);
	cudaVerify(cudaMallocHost((void**)&p_host.d, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.dddt, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.d, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.damage_total, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.dddt, memorySizeForParticles));

	cudaVerify(cudaMallocHost((void**)&p_host.numFlaws, memorySizeForInteractions));
	cudaVerify(cudaMalloc((void**)&p_device.numFlaws, memorySizeForInteractions));
	cudaVerify(cudaMallocHost((void**)&p_host.numActiveFlaws, memorySizeForInteractions));
	cudaVerify(cudaMalloc((void**)&p_device.numActiveFlaws, memorySizeForInteractions));
	cudaVerify(cudaMallocHost((void**)&p_host.flaws, memorySizeForActivationThreshold));
	cudaVerify(cudaMalloc((void**)&p_device.flaws, memorySizeForActivationThreshold));
# if PALPHA_POROSITY
    cudaVerify(cudaMallocHost((void**)&p_host.damage_porjutzi, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.ddamage_porjutzidt, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.damage_porjutzi, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.ddamage_porjutzidt, memorySizeForParticles));
# endif
#endif

	cudaVerify(cudaMalloc((void**)&p_device.h0, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.h0, memorySizeForParticles));

#if GHOST_BOUNDARIES
	cudaVerify(cudaMalloc((void**)&p_device.real_partner, memorySizeForInteractions));
#endif

#if PALPHA_POROSITY
	cudaVerify(cudaMallocHost((void**)&p_host.alpha_jutzi, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.alpha_jutzi_old, memorySizeForParticles));
	cudaVerify(cudaMallocHost((void**)&p_host.pold, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.dalphadt, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.pold, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.alpha_jutzi, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.alpha_jutzi_old, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.dalphadt, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.dalphadp, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.dp, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.dalphadrho, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.f, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.delpdelrho, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.delpdele, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.cs_old, memorySizeForParticles));
#endif

#if SIRONO_POROSITY
    cudaVerify(cudaMallocHost((void**)&p_host.compressive_strength, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.tensile_strength, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.shear_strength, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.rho_0prime, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.rho_c_plus, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.rho_c_minus, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.K, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.flag_rho_0prime, memorySizeForInteractions));
    cudaVerify(cudaMallocHost((void**)&p_host.flag_plastic, memorySizeForInteractions));
    cudaVerify(cudaMalloc((void**)&p_device.compressive_strength, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.tensile_strength, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.shear_strength, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.K, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.rho_0prime, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.rho_c_plus, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.rho_c_minus, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.flag_rho_0prime, memorySizeForInteractions));
    cudaVerify(cudaMalloc((void**)&p_device.flag_plastic, memorySizeForInteractions));
#endif

#if EPSALPHA_POROSITY
    cudaVerify(cudaMallocHost((void**)&p_host.alpha_epspor, memorySizeForParticles));
    cudaVerify(cudaMallocHost((void**)&p_host.epsilon_v, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.alpha_epspor, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.dalpha_epspordt, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.epsilon_v, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.depsilon_vdt, memorySizeForParticles));
#endif

	cudaVerify(cudaMalloc((void**)&p_device.x, memorySizeForTree));
	cudaVerify(cudaMalloc((void**)&p_device.g_x, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.g_local_cellsize, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.vx, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.dxdt, memorySizeForParticles));

#if DIM > 1
	cudaVerify(cudaMalloc((void**)&p_device.y, memorySizeForTree));
	cudaVerify(cudaMalloc((void**)&p_device.g_y, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.vy, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.dydt, memorySizeForParticles));
    cudaVerify(cudaMalloc((void**)&p_device.y0, memorySizeForTree));
    cudaVerify(cudaMalloc((void**)&p_device.vy0, memorySizeForTree));
    cudaVerify(cudaMallocHost((void**)&p_host.vy0, memorySizeForTree));
#endif

    cudaVerify(cudaMalloc((void**)&p_device.x0, memorySizeForTree));
    cudaVerify(cudaMalloc((void**)&p_device.vx0, memorySizeForTree));
    cudaVerify(cudaMallocHost((void**)&p_host.vx0, memorySizeForTree));
#if DIM > 2
    cudaVerify(cudaMalloc((void**)&p_device.z0, memorySizeForTree));
    cudaVerify(cudaMalloc((void**)&p_device.vz0, memorySizeForTree));
    cudaVerify(cudaMallocHost((void**)&p_host.vz0, memorySizeForTree));
#endif

#if XSPH
	cudaVerify(cudaMalloc((void**)&p_device.xsphvx, memorySizeForParticles));
#if DIM > 1
	cudaVerify(cudaMalloc((void**)&p_device.xsphvy, memorySizeForParticles));
#endif
#endif
	cudaVerify(cudaMalloc((void**)&p_device.ax, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.g_ax, memorySizeForParticles));

#if DIM > 1
	cudaVerify(cudaMalloc((void**)&p_device.ay, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.g_ay, memorySizeForParticles));
#endif

	cudaVerify(cudaMalloc((void**)&p_device.m, memorySizeForTree));
	cudaVerify(cudaMalloc((void**)&p_device.h, memorySizeForParticles));

#if INTEGRATE_SML
	cudaVerify(cudaMalloc((void**)&p_device.dhdt, memorySizeForParticles));
#endif

#if SML_CORRECTION
	cudaVerify(cudaMalloc((void**)&p_device.sml_omega, memorySizeForParticles));
#endif

	cudaVerify(cudaMalloc((void**)&p_device.rho, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.p, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.e, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.cs, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.depth, memorySizeForInteractions));
	cudaVerify(cudaMalloc((void**)&p_device.noi, memorySizeForInteractions));
	cudaVerify(cudaMalloc((void**)&p_device.materialId, memorySizeForInteractions));
	cudaVerify(cudaMalloc((void**)&p_device.materialId0, memorySizeForInteractions));
	cudaVerify(cudaMalloc((void**)&p_device.deactivate_me_flag, memorySizeForInteractions));

#if MORE_OUTPUT
	cudaVerify(cudaMalloc((void**)&p_device.p_min, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.p_max, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.rho_min, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.rho_max, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.e_min, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.e_max, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.cs_min, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.cs_max, memorySizeForParticles));
#endif

	cudaVerify(cudaMalloc((void**)&interactions, memorySizeForInteractions*MAX_NUM_INTERACTIONS));
	cudaVerify(cudaMalloc((void**)&childListd, memorySizeForChildren));
#if DIM > 2
	cudaVerify(cudaMalloc((void**)&p_device.z, memorySizeForTree));
	cudaVerify(cudaMalloc((void**)&p_device.g_z, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.dzdt, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.vz, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.az, memorySizeForParticles));
	cudaVerify(cudaMalloc((void**)&p_device.g_az, memorySizeForParticles));
#if XSPH
	cudaVerify(cudaMalloc((void**)&p_device.xsphvz, memorySizeForParticles));
#endif
#endif

	cudaVerify(cudaMemset(p_device.ax, 0, memorySizeForParticles));
	cudaVerify(cudaMemset(p_device.g_ax, 0, memorySizeForParticles));
#if DIM > 1
	cudaVerify(cudaMemset(p_device.ay, 0, memorySizeForParticles));
	cudaVerify(cudaMemset(p_device.g_ay, 0, memorySizeForParticles));
#endif
#if DIM > 2
	cudaVerify(cudaMemset(p_device.az, 0, memorySizeForParticles));
	cudaVerify(cudaMemset(p_device.g_az, 0, memorySizeForParticles));
#endif

    return rc;
}



int copy_particle_data_to_device()
{
    int rc = 0;

	if (param.verbose)
        fprintf(stdout, "\nCopying particle data to device...\n");

	cudaVerify(cudaMemcpy(p_device.x0, p_host.x, memorySizeForTree, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.x, p_host.x, memorySizeForTree, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.vx, p_host.vx, memorySizeForParticles, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.vx0, p_host.vx0, memorySizeForParticles, cudaMemcpyHostToDevice));
#if DIM > 1
	cudaVerify(cudaMemcpy(p_device.y0, p_host.y, memorySizeForTree, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.y, p_host.y, memorySizeForTree, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.vy, p_host.vy, memorySizeForParticles, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.vy0, p_host.vy0, memorySizeForParticles, cudaMemcpyHostToDevice));
#endif
#if DIM > 2
	cudaVerify(cudaMemcpy(p_device.z0, p_host.z, memorySizeForTree, cudaMemcpyHostToDevice));
#endif

#if GRAVITATING_POINT_MASSES
	cudaVerify(cudaMemcpy(pointmass_device.x, pointmass_host.x, memorySizeForPointmasses, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(pointmass_device.vx, pointmass_host.vx, memorySizeForPointmasses, cudaMemcpyHostToDevice));
# if DIM > 1
	cudaVerify(cudaMemcpy(pointmass_device.y, pointmass_host.y, memorySizeForPointmasses, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(pointmass_device.vy, pointmass_host.vy, memorySizeForPointmasses, cudaMemcpyHostToDevice));
#  if DIM > 2
	cudaVerify(cudaMemcpy(pointmass_device.z, pointmass_host.z, memorySizeForPointmasses, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(pointmass_device.vz, pointmass_host.vz, memorySizeForPointmasses, cudaMemcpyHostToDevice));
#  endif
# endif
	cudaVerify(cudaMemcpy(pointmass_device.rmin, pointmass_host.rmin, memorySizeForPointmasses, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(pointmass_device.rmax, pointmass_host.rmax, memorySizeForPointmasses, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(pointmass_device.m, pointmass_host.m, memorySizeForPointmasses, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(pointmass_device.feels_particles, pointmass_host.feels_particles, integermemorySizeForPointmasses, cudaMemcpyHostToDevice));
#endif

	cudaVerify(cudaMemcpy(p_device.h, p_host.h, memorySizeForParticles, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.cs, p_host.cs, memorySizeForParticles, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.m, p_host.m, memorySizeForTree, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.rho, p_host.rho, memorySizeForParticles, cudaMemcpyHostToDevice));
#if INTEGRATE_ENERGY
	cudaVerify(cudaMemcpy(p_device.e, p_host.e, memorySizeForParticles, cudaMemcpyHostToDevice));
#endif
#if SOLID
	cudaVerify(cudaMemcpy(p_device.S, p_host.S, memorySizeForStress, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.ep, p_host.ep, memorySizeForParticles, cudaMemcpyHostToDevice));
#endif
#if NAVIER_STOKES
	cudaVerify(cudaMemcpy(p_device.Tshear, p_host.Tshear, memorySizeForStress, cudaMemcpyHostToDevice));
#endif
#if PALPHA_POROSITY
	cudaVerify(cudaMemcpy(p_device.alpha_jutzi, p_host.alpha_jutzi, memorySizeForParticles, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.alpha_jutzi_old, p_host.alpha_jutzi_old, memorySizeForParticles, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.p, p_host.p, memorySizeForParticles, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.pold, p_host.pold, memorySizeForParticles, cudaMemcpyHostToDevice));
#endif
#if MORE_OUTPUT
    cudaVerify(cudaMemcpy(p_device.p_min, p_host.p_min, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.p_max, p_host.p_max, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.rho_min, p_host.rho_min, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.rho_max, p_host.rho_max, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.e_min, p_host.e_min, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.e_max, p_host.e_max, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.cs_min, p_host.cs_min, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.cs_max, p_host.cs_max, memorySizeForParticles, cudaMemcpyHostToDevice));
#endif
#if SIRONO_POROSITY
    cudaVerify(cudaMemcpy(p_device.compressive_strength, p_host.compressive_strength, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.tensile_strength, p_host.tensile_strength, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.shear_strength, p_host.shear_strength, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.rho_0prime, p_host.rho_0prime, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.rho_c_plus, p_host.rho_c_plus, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.rho_c_minus, p_host.rho_c_minus, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.K, p_host.K, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.flag_rho_0prime, p_host.flag_rho_0prime, memorySizeForInteractions, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.flag_plastic, p_host.flag_plastic, memorySizeForInteractions, cudaMemcpyHostToDevice));
#endif
#if EPSALPHA_POROSITY
    cudaVerify(cudaMemcpy(p_device.alpha_epspor, p_host.alpha_epspor, memorySizeForParticles, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.epsilon_v, p_host.epsilon_v, memorySizeForParticles, cudaMemcpyHostToDevice));
#endif
    cudaVerify(cudaMemcpy(p_device.h0, p_host.h0, memorySizeForParticles, cudaMemcpyHostToDevice));
#if JC_PLASTICITY
	cudaVerify(cudaMemcpy(p_device.T, p_host.T, memorySizeForParticles, cudaMemcpyHostToDevice));
#endif
#if FRAGMENTATION
	cudaVerify(cudaMemcpy(p_device.d, p_host.d, memorySizeForParticles, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.numFlaws, p_host.numFlaws, memorySizeForInteractions, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.numActiveFlaws, p_host.numActiveFlaws, memorySizeForInteractions, cudaMemcpyHostToDevice));
    cudaVerify(cudaMemcpy(p_device.flaws, p_host.flaws, memorySizeForActivationThreshold, cudaMemcpyHostToDevice));
# if PALPHA_POROSITY
    cudaVerify(cudaMemcpy(p_device.damage_porjutzi, p_host.damage_porjutzi, memorySizeForParticles, cudaMemcpyHostToDevice));
# endif
#endif
	cudaVerify(cudaMemcpy(p_device.noi, p_host.noi, memorySizeForInteractions, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.materialId, p_host.materialId, memorySizeForInteractions, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.materialId0, p_host.materialId, memorySizeForInteractions, cudaMemcpyHostToDevice));
#if DIM > 2
	cudaVerify(cudaMemcpy(p_device.z, p_host.z, memorySizeForTree, cudaMemcpyHostToDevice));
	cudaVerify(cudaMemcpy(p_device.vz, p_host.vz, memorySizeForParticles, cudaMemcpyHostToDevice));
#endif
	cudaVerify(cudaMemset((void *) childListd, -1, memorySizeForChildren));

    return rc;
}



int free_memory()
{
    int rc = 0;

	/* free device memory */
	if (param.verbose)
        fprintf(stdout, "Freeing memory...\n");
	cudaVerify(cudaFree(p_device.x));
	cudaVerify(cudaFree(p_device.g_x));
	cudaVerify(cudaFree(p_device.g_local_cellsize));
	cudaVerify(cudaFree(p_device.depth));
	cudaVerify(cudaFree(p_device.x0));
	cudaVerify(cudaFree(p_device.dxdt));
	cudaVerify(cudaFree(p_device.vx));
	cudaVerify(cudaFree(p_device.vx0));
	cudaVerify(cudaFreeHost(p_host.vx0));
	cudaVerify(cudaFree(p_device.ax));
	cudaVerify(cudaFree(p_device.g_ax));
	cudaVerify(cudaFree(p_device.m));

#if DIM > 1
	cudaVerify(cudaFree(p_device.vy0));
	cudaVerify(cudaFreeHost(p_host.vy0));
#if DIM > 2
	cudaVerify(cudaFree(p_device.vz0));
	cudaVerify(cudaFreeHost(p_host.vz0));
#endif
#endif
#if DIM > 1
	cudaVerify(cudaFree(p_device.y));
	cudaVerify(cudaFree(p_device.g_y));
	cudaVerify(cudaFree(p_device.y0));
	cudaVerify(cudaFree(p_device.vy));
	cudaVerify(cudaFree(p_device.dydt));
	cudaVerify(cudaFree(p_device.ay));
	cudaVerify(cudaFree(p_device.g_ay));
#endif

#if GRAVITATING_POINT_MASSES
	cudaVerify(cudaFree(pointmass_device.x));
	cudaVerify(cudaFree(pointmass_device.vx));
	cudaVerify(cudaFree(pointmass_device.ax));
	cudaVerify(cudaFree(pointmass_device.feedback_ax));
# if DIM > 1
	cudaVerify(cudaFree(pointmass_device.y));
	cudaVerify(cudaFree(pointmass_device.vy));
	cudaVerify(cudaFree(pointmass_device.ay));
	cudaVerify(cudaFree(pointmass_device.feedback_ay));
#  if DIM > 2
	cudaVerify(cudaFree(pointmass_device.z));
	cudaVerify(cudaFree(pointmass_device.vz));
	cudaVerify(cudaFree(pointmass_device.az));
	cudaVerify(cudaFree(pointmass_device.feedback_az));
#  endif
# endif
	cudaVerify(cudaFree(pointmass_device.m));
	cudaVerify(cudaFree(pointmass_device.feels_particles));
	cudaVerify(cudaFree(pointmass_device.rmin));
	cudaVerify(cudaFree(pointmass_device.rmax));

	cudaVerify(cudaFreeHost(pointmass_host.x));
	cudaVerify(cudaFreeHost(pointmass_host.vx));
	cudaVerify(cudaFreeHost(pointmass_host.ax));
# if DIM > 1
	cudaVerify(cudaFreeHost(pointmass_host.y));
	cudaVerify(cudaFreeHost(pointmass_host.vy));
	cudaVerify(cudaFreeHost(pointmass_host.ay));
#  if DIM > 2
	cudaVerify(cudaFreeHost(pointmass_host.z));
	cudaVerify(cudaFreeHost(pointmass_host.vz));
	cudaVerify(cudaFreeHost(pointmass_host.az));
#  endif
# endif
	cudaVerify(cudaFreeHost(pointmass_host.m));
	cudaVerify(cudaFreeHost(pointmass_host.feels_particles));
	cudaVerify(cudaFreeHost(pointmass_host.rmin));
	cudaVerify(cudaFreeHost(pointmass_host.rmax));
#endif

#if XSPH
	cudaVerify(cudaFree(p_device.xsphvx));
#if DIM > 1
	cudaVerify(cudaFree(p_device.xsphvy));
#endif
#endif
	cudaVerify(cudaFree(p_device.h));
	cudaVerify(cudaFree(p_device.rho));
	cudaVerify(cudaFree(p_device.p));
	cudaVerify(cudaFree(p_device.e));
	cudaVerify(cudaFree(p_device.cs));
	cudaVerify(cudaFree(p_device.noi));
#if MORE_OUTPUT
	cudaVerify(cudaFree(p_device.p_min));
    cudaVerify(cudaFree(p_device.p_max));
    cudaVerify(cudaFree(p_device.rho_min));
    cudaVerify(cudaFree(p_device.rho_max));
	cudaVerify(cudaFree(p_device.e_min));
    cudaVerify(cudaFree(p_device.e_max));
    cudaVerify(cudaFree(p_device.cs_min));
    cudaVerify(cudaFree(p_device.cs_max));
#endif
#if ARTIFICIAL_VISCOSITY
	cudaVerify(cudaFree(p_device.muijmax));
#endif
#if INVISCID_SPH
	cudaVerify(cudaFree(p_device.beta));
	cudaVerify(cudaFree(p_device.beta_old));
	cudaVerify(cudaFree(p_device.divv_old));
#endif
	cudaVerify(cudaFree(interactions));
	cudaVerify(cudaFree(p_device.materialId));
	cudaVerify(cudaFree(p_device.materialId0));
	cudaVerify(cudaFree(p_device.deactivate_me_flag));
	cudaVerify(cudaFree(childListd));
#if DIM > 2
	cudaVerify(cudaFree(p_device.z));
	cudaVerify(cudaFree(p_device.g_z));
	cudaVerify(cudaFree(p_device.z0));
	cudaVerify(cudaFree(p_device.dzdt));
	cudaVerify(cudaFree(p_device.vz));
#if XSPH
	cudaVerify(cudaFree(p_device.xsphvz));
#endif
	cudaVerify(cudaFree(p_device.az));
	cudaVerify(cudaFree(p_device.g_az));
#endif

#if TENSORIAL_CORRECTION
	cudaVerify(cudaFree(p_device.tensorialCorrectionMatrix));
	//cudaVerify(cudaFree(p_device.tensorialCorrectiondWdrr));
#endif

#if SHEPARD_CORRECTION
	cudaVerify(cudaFree(p_device.shepard_correction));
#endif

#if INTEGRATE_ENERGY
	cudaVerify(cudaFreeHost(p_host.dedt));
	cudaVerify(cudaFree(p_device.dedt));
#endif

	cudaVerify(cudaFreeHost(p_host.drhodt));
	cudaVerify(cudaFree(p_device.drhodt));

#if INTEGRATE_SML
	cudaVerify(cudaFree(p_device.dhdt));
#endif
#if SML_CORRECTION
	cudaVerify(cudaFree(p_device.sml_omega));
#endif

#if NAVIER_STOKES
	cudaVerify(cudaFree(p_device.Tshear));
	cudaVerify(cudaFreeHost(p_host.Tshear));
	cudaVerify(cudaFree(p_device.eta));
#endif
#if SOLID
	cudaVerify(cudaFree(p_device.S));
    cudaVerify(cudaFreeHost(p_host.ep));
	cudaVerify(cudaFree(p_device.dSdt));
	cudaVerify(cudaFreeHost(p_host.S));
	cudaVerify(cudaFreeHost(p_host.dSdt));
	cudaVerify(cudaFree(p_device.local_strain));
	cudaVerify(cudaFreeHost(p_host.local_strain));
    cudaVerify(cudaFree(p_device.plastic_f));
	cudaVerify(cudaFree(p_device.sigma));
    cudaVerify(cudaFree(p_device.ep));
    cudaVerify(cudaFree(p_device.edotp));
#endif
#if ARTIFICIAL_STRESS
	cudaVerify(cudaFree(p_device.R));
#endif

#if JC_PLASTICITY
	cudaVerify(cudaFree(p_device.T));
	cudaVerify(cudaFree(p_device.dTdt));
	cudaVerify(cudaFree(p_device.jc_f));
#endif

#if GHOST_BOUNDARIES
	cudaVerify(cudaFree(p_device.real_partner));
#endif

#if FRAGMENTATION
	cudaVerify(cudaFreeHost(p_host.d));
	cudaVerify(cudaFree(p_device.d));
	cudaVerify(cudaFree(p_device.damage_total));
	cudaVerify(cudaFree(p_device.dddt));
	cudaVerify(cudaFreeHost(p_host.dddt));
	cudaVerify(cudaFreeHost(p_host.numFlaws));
	cudaVerify(cudaFree(p_device.numFlaws));
	cudaVerify(cudaFreeHost(p_host.numActiveFlaws));
	cudaVerify(cudaFree(p_device.numActiveFlaws));
	cudaVerify(cudaFreeHost(p_host.flaws));
	cudaVerify(cudaFree(p_device.flaws));
# if PALPHA_POROSITY
	cudaVerify(cudaFree(p_device.damage_porjutzi));
	cudaVerify(cudaFree(p_device.cs_old));
	cudaVerify(cudaFree(p_device.ddamage_porjutzidt));
# endif
#endif


#if PALPHA_POROSITY
	cudaVerify(cudaFree(p_device.alpha_jutzi));
	cudaVerify(cudaFree(p_device.alpha_jutzi_old));
	cudaVerify(cudaFree(p_device.pold));
	cudaVerify(cudaFree(p_device.dalphadt));
	cudaVerify(cudaFree(p_device.dalphadp));
	cudaVerify(cudaFree(p_device.dp));
	cudaVerify(cudaFree(p_device.dalphadrho));
	cudaVerify(cudaFree(p_device.f));
	cudaVerify(cudaFree(p_device.delpdelrho));
	cudaVerify(cudaFree(p_device.delpdele));
#endif

#if SIRONO_POROSITY
    cudaVerify(cudaFree(p_device.compressive_strength));
    cudaVerify(cudaFree(p_device.tensile_strength));
    cudaVerify(cudaFree(p_device.shear_strength));
    cudaVerify(cudaFree(p_device.K));
    cudaVerify(cudaFree(p_device.rho_0prime));
    cudaVerify(cudaFree(p_device.rho_c_plus));
    cudaVerify(cudaFree(p_device.rho_c_minus));
    cudaVerify(cudaFree(p_device.flag_rho_0prime));
    cudaVerify(cudaFree(p_device.flag_plastic));
#endif

#if EPSALPHA_POROSITY
    cudaVerify(cudaFree(p_device.alpha_epspor));
    cudaVerify(cudaFree(p_device.dalpha_epspordt));
    cudaVerify(cudaFree(p_device.epsilon_v));
    cudaVerify(cudaFree(p_device.depsilon_vdt));
#endif

	cudaVerify(cudaFreeHost(p_host.x));
	cudaVerify(cudaFreeHost(p_host.vx));
	cudaVerify(cudaFreeHost(p_host.ax));
    cudaVerify(cudaFreeHost(p_host.g_ax));
#if DIM > 1
	cudaVerify(cudaFreeHost(p_host.y));
	cudaVerify(cudaFreeHost(p_host.vy));
	cudaVerify(cudaFreeHost(p_host.ay));
    cudaVerify(cudaFreeHost(p_host.g_ay));
#endif
	cudaVerify(cudaFreeHost(p_host.m));
	cudaVerify(cudaFreeHost(p_host.h));
	cudaVerify(cudaFreeHost(p_host.rho));
	cudaVerify(cudaFreeHost(p_host.p));
	cudaVerify(cudaFreeHost(p_host.e));
	cudaVerify(cudaFreeHost(p_host.cs));
	cudaVerify(cudaFreeHost(p_host.noi));
	cudaVerify(cudaFreeHost(interactions_host));
	cudaVerify(cudaFreeHost(p_host.depth));
	cudaVerify(cudaFreeHost(p_host.materialId));
	cudaVerify(cudaFreeHost(childList_host));
#if MORE_OUTPUT
	cudaVerify(cudaFreeHost(p_host.p_min));
	cudaVerify(cudaFreeHost(p_host.p_max));
	cudaVerify(cudaFreeHost(p_host.rho_min));
	cudaVerify(cudaFreeHost(p_host.rho_max));
	cudaVerify(cudaFreeHost(p_host.e_min));
	cudaVerify(cudaFreeHost(p_host.e_max));
	cudaVerify(cudaFreeHost(p_host.cs_min));
	cudaVerify(cudaFreeHost(p_host.cs_max));
#endif
#if INVISCID_SPH
	cudaVerify(cudaFreeHost(p_host.beta));
	cudaVerify(cudaFreeHost(p_host.beta_old));
	cudaVerify(cudaFreeHost(p_host.divv_old));
#endif
#if PALPHA_POROSITY
	cudaVerify(cudaFreeHost(p_host.alpha_jutzi));
	cudaVerify(cudaFreeHost(p_host.alpha_jutzi_old));
	cudaVerify(cudaFreeHost(p_host.dalphadt));
	cudaVerify(cudaFreeHost(p_host.pold));
# if FRAGMENTATION
    cudaVerify(cudaFreeHost(p_host.damage_porjutzi));
    cudaVerify(cudaFreeHost(p_host.ddamage_porjutzidt));
# endif
#endif

#if SIRONO_POROSITY
    cudaVerify(cudaFreeHost(p_host.compressive_strength));
    cudaVerify(cudaFreeHost(p_host.tensile_strength));
    cudaVerify(cudaFreeHost(p_host.shear_strength));
    cudaVerify(cudaFreeHost(p_host.rho_0prime));
    cudaVerify(cudaFreeHost(p_host.rho_c_plus));
    cudaVerify(cudaFreeHost(p_host.rho_c_minus));
    cudaVerify(cudaFreeHost(p_host.K));
    cudaVerify(cudaFreeHost(p_host.flag_rho_0prime));
    cudaVerify(cudaFreeHost(p_host.flag_plastic));
#endif

#if EPSALPHA_POROSITY
    cudaVerify(cudaFreeHost(p_host.alpha_epspor));
    cudaVerify(cudaFreeHost(p_host.epsilon_v));
#endif

#if JC_PLASTICITY
	cudaVerify(cudaFreeHost(p_host.T));
#endif
#if DIM > 2
	cudaVerify(cudaFreeHost(p_host.z));
	cudaVerify(cudaFreeHost(p_host.vz));
	cudaVerify(cudaFreeHost(p_host.az));
    cudaVerify(cudaFreeHost(p_host.g_az));
#endif

    free_aneos_memory();

    return rc;
}
