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

#ifndef _DISPH_yY_H
#define _DISPH_yY_H


#include "timeintegration.h"

/** 
 * @brief Calculates the DISPH_pressure using the kernel sum. This is only one part of the full algorithm for calculating the real pressure in DISPH
 */
__global__ void calculate_DISPH_y_DISPH_rho(int *interactions);
__global__ void calculate_DISPH_Y();
__global__ void determine_max_dp(double *maxDISPH_PressureErrorPerBlock);
__global__ void set_initial_DISPH_Y_if_its_zero(int *DISPH_initial_YPerBlock);
__global__ void DISPH_Y_to_zero();
__global__ void calculate_DISPH_f_grad(int *interactions);
__global__ void calc_DISPH_sml();
__global__ void SPH_rho_to_DISPH_rho();

#endif
