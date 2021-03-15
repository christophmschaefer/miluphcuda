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
/**
 * @brief Calculates b-spline kernel and derivatives for an interaction
 * 
 */
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

/// function pointer to generic SPH kernel function
/// cuda allows function pointers since Fermi architecture.
/// however, we need a typedef to set the function pointers
typedef void (*SPH_kernel) (double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
/**
 * @brief *The* standard cubic b-spline.
 * 
 * @param W 
 * @param dWdx 
 * @param dWdr 
 * @param dx 
 * @param h 
 * @return __device__ 
 */
__device__ void cubic_spline(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
/**
 * @brief Spiky kernel. 
 * @details only implemented for 2D and 3D. 
 * @param W 
 * @param dWdx 
 * @param dWdr 
 * @param dx 
 * @param h 
 * @return __device__ 
 */
__device__ void spiky(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
/**
 * @brief Quartic spline kernel.
 * 
 * @param W 
 * @param dWdx 
 * @param dWdr 
 * @param dx 
 * @param h 
 * @return __device__ 
 */
__device__ void quartic_spline(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
/**
 * @brief Quintic spline kernel.
 * 
 * @param W 
 * @param dWdx 
 * @param dWdr 
 * @param dx 
 * @param h 
 * @return __device__ 
 */
__device__ void quintic_spline(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
/**
 * @brief Wendland C2 kernel.
 * 
 * @param W 
 * @param dWdx 
 * @param dWdr 
 * @param dx 
 * @param h 
 * @return __device__ 
 */
__device__ void wendlandc2(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
/**
 * @brief Wendland C4 kernel.
 * 
 * @param W 
 * @param dWdx 
 * @param dWdr 
 * @param dx 
 * @param h 
 * @return __device__ 
 */
__device__ void wendlandc4(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
/**
 * @brief Wendland C6 kernel.
 * 
 * @param W 
 * @param dWdx 
 * @param dWdr 
 * @param dx 
 * @param h 
 * @return __device__ 
 */
__device__ void wendlandc6(double *W, double dWdx[DIM], double *dWdr, double dx[DIM], double h);
/**
 * @brief Calculates the kernel for the tensile instability fix following Monaghan 2000.
 * 
 * @param particle1 
 * @param particle2 
 * @return __device__ 
 */
__device__ double fixTensileInstability( int particle1, int particle2);
/**
 * @brief Calculates the tensorial correction factors for linear consistency.
 * 
 * @param interactions 
 * @return __global__ 
 */
__global__ void tensorialCorrection(int *interactions);
/**
 * @brief Calculates the zeroth order corrections for the kernel sum.
 * 
 * @param interactions 
 * @return __global__ 
 */
__global__ void shepardCorrection(int *interactions);
/**
 * @brief Calculates \f$ \nabla \cdot \vec{v} \f$ and \f$ \nabla \times \vec{v} \f$
 * 
 * @param interactions 
 * @return __global__ 
 */
__global__ void CalcDivvandCurlv(int *interactions);

#endif
