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


#ifndef _INVISCIDSPH_H
#define _INVISCIDSPH_H

#include "timeintegration.h"


#if INVISCID_SPH
/**
 * @brief Time-dependent artificial viscosity of Cullen & Dehnen (2010): shock indicator, local alpha and its decay
 * rate, at the end of every right-hand side (see viscosity.cu).
 */
__global__ void inviscidSPHSwitch(int *interactions);
/**
 * @brief After an accepted time step: alpha_i = max(alpha_i, alpha_loc).
 */
__global__ void inviscidSPHRaiseAlpha(void);
/**
 * @brief Start value of alpha_i, the alpha of the material.
 */
__global__ void initInviscidSPH(void);
#endif

__global__ void calculate_shear_stress_tensor(int *interactions);
__global__ void calculate_kinematic_viscosity(void);






#endif
