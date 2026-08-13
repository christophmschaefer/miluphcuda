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

#ifndef _FAST_INTEGRATION_H
#define _FAST_INTEGRATION_H

#include "parameter.h"

#if FAST_INTEGRATION_SCHEME

/**
 * @brief Switches a material to the low-sound-speed medium of the fast integration scheme.
 * @details Follows Raducan & Jutzi (2022) and Jutzi et al. (2022). One thread per material.
 *          The kernel is idempotent: matFastSwitched guards against switching twice.
 */
__global__ void applyFastScheme(double time, int nMaterials);

/**
 * @brief Host wrapper. Call once per integrator substep, right before rightHandSide().
 */
void check_fast_scheme(void);

#endif

#endif