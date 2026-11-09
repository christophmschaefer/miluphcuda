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

#ifndef _KDK_H
#define _KDK_H

#include "miluph.h"
#include "timeintegration.h"
#include "parameter.h"


/* kdk integrator settings
 *
 * The time step is set before each step from the derivatives of the current state (there is no error estimate and
 * no rejected step). The CFL condition, with the shear wave speed for SOLID and the signal speed of the artificial
 * viscosity, is always used with KDK_COURANT_FACT. In addition:
 */
#define KDK_COURANT_FACT COURANT_FACT  // CFL number for kdk, the default of 0.4 refers to sml at the kernel zero
#define KDK_USE_FORCES_LIMIT 1      // local forces/acceleration, with dt ~ sqrt(sml/a), see FORCES_FACT
#define KDK_USE_DAMAGE_LIMIT 1      // rate of damage change (FRAGMENTATION only)
#define KDK_MAX_DAMAGE_CHANGE 0.15  // max allowed (absolute) change of the DIM-root of the tensile damage per step
#define KDK_MAX_ALPHA_CHANGE 1e-2   // max allowed (absolute) distention change per step (p-alpha model with the
                                    // time integration of alpha, i.e. PALPHA_RETURN_MAPPING 0)
#define KDK_SMALLEST_DT 1e-16       // simulation aborts if the time step falls below

#define NUM_THREADS_KDK_INTEGRATOR 256


/**
 * @brief Kick-drift-kick (leapfrog) integrator, 2nd order.
 *
 * @details One rhs per step:
 *     v^{n+1/2} = v^n + dt/2 a^n,  x^{n+1} = x^n + dt v^{n+1/2},
 *     a^{n+1} = a(x^{n+1}, v^n + dt a^n, Q^n + dt dQ/dt^n),
 *     v^{n+1} = v^{n+1/2} + dt/2 a^{n+1},
 * and the same kicks for all other integrated quantities Q (density, energy, smoothing length, deviatoric stress,
 * damage, distention, ...). Since the drift uses exactly the velocity of the kicks, the scheme conserves the total
 * angular momentum up to round-off for every time step size if the forces are torque-free, e.g. with
 * TENSORIAL_CORRECTION_CONSERVE_ANGULAR_MOMENTUM, and without XSPH, boundaries, tree gravity and decoupled gravity.
 */
void kdk(void);

/// prints the settings of the kdk integrator
void print_kdk_settings(void);

#endif
