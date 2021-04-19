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

#ifndef _RK2ADAPTIVE_H
#define _RK2ADAPTIVE_H

#include "miluph.h"
#include "timeintegration.h"
#include "parameter.h"


/* rk2_adaptive integration parameters */

/* specify which quantities are used for error estimate, where positions are always used (for particles) */
/* for pointmasses, no error checking is done by default, but can be set for velocities */
#define RK2_USE_VELOCITY_ERROR 0
#define RK2_USE_DENSITY_ERROR 1
#define RK2_USE_ENERGY_ERROR 0
#define RK2_USE_VELOCITY_ERROR_POINTMASSES 0  // use velocity error checking for pointmasses

/* specific parameters */
#define RK2_LOCATION_SAFETY 0.1  // this times the sml defines the min length to consider in error check for positions
#define MIN_VEL_CHANGE_RK2 10.0  // defines min vel to consider in error check for velocities
#define RK2_TINY_DENSITY 1e-2
#define RK2_TINY_ENERGY 10.0
#define RK2_TIMESTEP_SAFETY 0.9
#define SMALLEST_DT_ALLOWED 1e-16
#define RK2_MAX_ALPHA_CHANGE 1e-4
#define RK2_MAX_DAMAGE_CHANGE 1e-2


void rk2Adaptive();

__global__ void integrateFirstStep(void);
__global__ void integrateSecondStep(void);
__global__ void integrateThirdStep(void);

/**
 * @brief Limit timestep.
 * @details Limits the timestep by:
 *     - CFL condition, via dt ~ sml/cs
 *     - local forces/acceleration, via dt ~ sqrt(sml/a)
 */
__global__ void limitTimestep(double *forcesPerBlock, double *courantPerBlock);

__global__ void checkError(
		double *maxPosAbsErrorPerBlock
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
        , double *maxVelAbsErrorPerBlock
#endif
#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
		, double *maxDensityAbsErrorPerBlock
#endif
#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
		, double *maxEnergyAbsErrorPerBlock
#endif
#if PALPHA_POROSITY
        , double *maxPressureAbsChangePerBlock
#endif
);

__global__ void damageMaxTimeStep(double *maxDamageTimeStepPerBlock);
__global__ void alphaMaxTimeStep(double *maxalphaDiffPerBlock);


#if RK2_USE_VELOCITY_ERROR_POINTMASSES && !GRAVITATING_POINT_MASSES
# error You set RK2_USE_VELOCITY_ERROR_POINTMASSES but not GRAVITATING_POINT_MASSES...
#endif

#endif
