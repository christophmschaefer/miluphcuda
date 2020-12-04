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


#ifndef _RK4_POINTMASS_H
#define _RK4_POINTMASS_H

#include "miluph.h"
#include "timeintegration.h"


void rk4_nbodies();

__global__ void rhs_pointmass();
__global__ void rk4_integrateFirstStep();
__global__ void rk4_integrateSecondStep();
__global__ void rk4_integrateThirdStep();
__global__ void rk4_integrateFourthStep();







#endif
