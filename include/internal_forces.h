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


#ifndef _INTERNAL_FORCES_H
#define _INTERNAL_FORCES_H


__global__ void internalForces(int *interactions);

#if INTEGRATE_SML && !SML_CORRECTION
// dh/dt of all real particles, has to be called before internalForces(), see internal_forces.cu
__global__ void calculateSmlRate(int *interactions);
#endif

__global__ void calculatedeviatoricStress(int *interactions);

#if DENSITY_DIFFUSION
/**
 * @brief Renormalised density gradient (Antuono et al. 2010) over the neighbours of the same material, for the
 * delta-SPH density diffusion in internalForces.
 */
__global__ void densityGradientForDiffusion(int *interactions);
#endif

#if VIRTUAL_WALLS
/**
 * @brief Fixed walls (VIRTUAL_WALLS): velocity of the wall particles seen by their neighbours, extrapolated from the
 * non-wall neighbours. Called before the first neighbour loop that uses partner velocities.
 */
__global__ void wallVelocity(int *interactions);
/**
 * @brief Fixed walls (VIRTUAL_WALLS): pressure, stress, density, sound speed (and alpha, div v, curl v) of the wall
 * particles, extrapolated from the non-wall neighbours. Called after set_stress_tensor(), before internalForces().
 */
__global__ void wallState(int *interactions);
/**
 * @brief Fixed walls (VIRTUAL_WALLS): velocity and all rates of the wall particles zero, at the end of the rhs.
 */
__global__ void fixWallParticles(void);
#endif






#endif
