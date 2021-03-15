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


__global__ void betaviscosity(int *interactions);

__global__ void calculate_shear_stress_tensor(int *interactions);
__global__ void calculate_kinematic_viscosity(void);



__device__ int sign(double x);
__device__ void multiply(double mat1[][DIM], double mat2[][DIM], double res[][DIM]);



#endif
