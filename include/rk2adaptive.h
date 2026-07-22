/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com, Christoph Burger
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


/* rk2_adaptive integrator settings */

/* pre-timestep checks to limit timestep in advance (all for particles only)
 * recommended:
 *   - use COURANT_LIMIT
 *   - for sims with FRAGMENTATION: use DAMAGE_LIMIT
 */
#define RK2_USE_COURANT_LIMIT 1  // CFL condition, with dt ~ sml/cs
#define RK2_USE_FORCES_LIMIT 0   // local forces/acceleration, with dt ~ sqrt(sml/a)
#define RK2_USE_DAMAGE_LIMIT 1   // rate of damage change

/* specify quantities for post-timestep error estimate, where positions are always used (for particles)
 * for pointmasses, no error checking is done by default, but can be set for velocities
 * recommended:
 *   - use DENSITY_ERROR
 *   - for sims with PALPHA_POROSITY: use LIMIT_ALPHA_CHANGE
 */
#define RK2_USE_VELOCITY_ERROR 0
#define RK2_USE_DENSITY_ERROR 1
#define RK2_USE_ENERGY_ERROR 0
#define RK2_USE_VELOCITY_ERROR_POINTMASSES 0  // use velocity error checking for pointmasses
#define RK2_LIMIT_PRESSURE_CHANGE 0   // special check for PALPHA_POROSITY for crush curve convergence
#define RK2_LIMIT_ALPHA_CHANGE 1   // special check for PALPHA_POROSITY for crush curve convergence

/* important parameters
 * recommended:
 *   - LOCATION_SAFETY: around 0.1
 *   - TIMESTEP_SAFETY: around 0.9
 *   - MAX_DAMAGE_CHANGE: 0.1 - 0.2  (for RK2_USE_DAMAGE_LIMIT)
 *   - MAX_ALPHA_CHANGE: 1e-3 - 1e-2  (for RK2_LIMIT_ALPHA_CHANGE)
 */
#define RK2_LOCATION_SAFETY 0.1    // this times the sml defines the min length to consider in error check for positions
#define MIN_VEL_CHANGE_RK2 10.0    // defines min vel to consider in error check for velocities
#define RK2_TINY_DENSITY 1e-2    // small density eps compared to typical densities
#define RK2_TINY_ENERGY 10.0     // small energy eps compared to typical energies
#define RK2_TIMESTEP_SAFETY 0.9    // safety factor for setting next timestep
#define SMALLEST_DT_ALLOWED 1e-16  // simulation aborts if timestep falls below
#define RK2_MAX_DAMAGE_CHANGE 0.15   // max allowed (absolute) damage (DIM-root of tensile) change per timestep (for RK2_USE_DAMAGE_LIMIT)
//#define RK2_MAX_ALPHA_CHANGE 2e-3    // max allowed (absolute) distention change per timestep (for RK2_LIMIT_ALPHA_CHANGE)
#define RK2_MAX_ALPHA_CHANGE 1e-2    // max allowed (absolute) distention change per timestep (for RK2_LIMIT_ALPHA_CHANGE)


/**
 * @brief Runge Kutta 2nd order integrator with adaptive timestep.
 * @details Embedded Runge Kutta 2/3 integrator, with several pre- and
 * post-timestep error checks. See Schaefer et al. (2016) for details.
 */
void rk2Adaptive();


/**
 * @brief Limits the timestep by the CFL condition, with dt ~ sml/cs.
 */
__global__ void limitTimestepCourant(double *courantPerBlock);

/**
 * @brief Limits the timestep based on local forces/acceleration, with dt ~ sqrt(sml/a).
 */
__global__ void limitTimestepForces(double *forcesPerBlock);

/**
 * @brief Limits the timestep based on the rate of damage change.
 */
__global__ void limitTimestepDamage(double *maxDamageTimeStepPerBlock);


__global__ void integrateFirstStep(void);
__global__ void integrateSecondStep(void);
__global__ void integrateThirdStep(void);


/**
 * @brief Post-timestep error checks for rk2_adaptive.
 * @details Contains error estimates/checks for positions, velocities, density,
 * energy, and pressure and distention change for p-alpha porosity.
 */
__global__ void checkError(double *maxPosAbsErrorPerBlock
#if RK2_USE_VELOCITY_ERROR || RK2_USE_VELOCITY_ERROR_POINTMASSES
                        , double *maxVelAbsErrorPerBlock
#endif
#if RK2_USE_DENSITY_ERROR && INTEGRATE_DENSITY
                        , double *maxDensityAbsErrorPerBlock
#endif
#if RK2_USE_ENERGY_ERROR && INTEGRATE_ENERGY
                        , double *maxEnergyAbsErrorPerBlock
#endif
#if RK2_LIMIT_PRESSURE_CHANGE && PALPHA_POROSITY
                        , double *maxPressureAbsChangePerBlock
#endif
#if RK2_LIMIT_ALPHA_CHANGE && PALPHA_POROSITY
                        , double *maxAlphaDiffPerBlock
#endif
);

void print_rk2_adaptive_settings();


#if RK2_USE_VELOCITY_ERROR_POINTMASSES && !GRAVITATING_POINT_MASSES
# error You set RK2_USE_VELOCITY_ERROR_POINTMASSES but not GRAVITATING_POINT_MASSES...
#endif

#endif
