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

#ifndef _DENSITY_H
#define _DENSITY_H
/** 
 * @brief Calculates the density using the kernel sum. 
 * @details Executed for all particles if INTEGRATE_DENSITY is 0, otherwise only for materials with
 * density_via_kernel_sum = 1. With SML_CORRECTION and iterate_sml = 1, the smoothing length h = h(rho) is iterated
 * (Newton-Raphson); the smoothing lengths of the interaction partners are then read from p_rhs.h_snapshot, which
 * has to be filled by snapshotSml() before. With iterate_sml = 0 the smoothing lengths are not changed.
 */
__global__ void calculateDensity(int *interactions, int iterate_sml);

#if SML_CORRECTION
/**
 * @brief Copies the current smoothing lengths p.h to p_rhs.h_snapshot.
 * @details Used by calculateDensity() for the SML_CORRECTION iteration, so that all threads read the same
 * (unchanged) smoothing lengths of their interaction partners while the own ones are updated.
 */
__global__ void snapshotSml(void);
#endif




#endif
