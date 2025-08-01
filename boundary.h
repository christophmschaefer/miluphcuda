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
#include "parameter.h"
#include "miluph.h"
#include "io.h"
#include "cuda_utils.h"

__global__ void BoundaryConditionsBeforeRHS(int *interactions);
__global__ void BoundaryConditionsAfterIntegratorStep(int *interactions);
__global__ void BoundaryConditionsBeforeIntegratorStep(int *interactions);
__global__ void BoundaryConditionsAfterRHS(int *interactions);
__global__ void BoundaryConditionsBrushesBefore(int *interactions);
__global__ void BoundaryConditionsBrushesAfter(int *interactions);
__global__ void BoundaryForce(int *interactions);
#if SOLID
__device__ void setBoundaryVelocity(int i, int *interactions);
#endif
#if GHOST_BOUNDARIES
__global__ void removeGhostParticles();
__global__ void insertGhostParticles();
__global__ void setQuantitiesGhostParticles();
#endif
__device__ void setQuantitiesFixedVirtualParticles(int i, int j, double *vxj, double *vyj, double *vzj, double *densityj, double *pressurej, double *Sj);
