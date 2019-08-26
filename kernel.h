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

#ifndef _KERNEL_H
#define _KERNEL_H

#include "parameter.h"

__device__ void fastKernelvalueAndDerivative(
        double &W, double &dWdx
#if DIM > 1
        , double &dWdy
#endif
#if DIM == 3
        , double &dWdz
#endif
        , double &dWdr,  int particle1, int particle2,
        double dx
#if DIM > 1
        , double dy
#endif
#if DIM == 3
        , double dz
#endif
);

// function pointer to generic SPH kernel function
// cuda allows function pointers since Fermi architecture.
// however, we need a typedef to set the function pointers
typedef void (*SPH_kernel) (double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
__device__ void cubic_spline(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
__device__ void spiky(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
__device__ void quartic_spline(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
__device__ void quintic_spline(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
__device__ void wendlandc2(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
__device__ void wendlandc4(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
__device__ void wendlandc6(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
__device__ void kernelvalue(double &W, int particle1, int particle2);
__device__ double fixTensileInstability( int particle1, int particle2);
__global__ void tensorialCorrection(int *interactions);



__global__ void CalcDivvandCurlv(int *interactions);




#endif
