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


#ifndef _LINALG_H
#define _LINALG_H

// returns the indices of the greatest non-diagonal element of M
__device__ int max_Matrix(double M[DIM][DIM], int *e, int *f, double *elmax);

/*
 * help function for the jacobi method
 * returns: M' = A^T M A, and A_ef = s = -A_ef, A_ee = A_ff = c
 */
__device__ void rotate_matrix(volatile double m[DIM][DIM], volatile double c, volatile double s, volatile int e,
volatile int f);
/*
 * computes the eigenvalues of the _symmetric_ matrix M
 * using the jacobi method
 * returns the greatest eigenvalue
 */
__device__ double calculateMaxEigenvalue(double M[DIM][DIM]);
__device__ int calculate_all_eigenvalues(double M[DIM][DIM], double eigenvals[DIM], double v[DIM][DIM]);
__device__ int invertMatrix(double *m, double *inverted);
__device__ void multiply_matrix(double a[DIM][DIM], double b[DIM][DIM], double c[DIM][DIM]);
__device__ void copy_matrix(double src[DIM][DIM], double dst[DIM][DIM]);
__device__ void identity_matrix(double a[DIM][DIM]);
__device__ void transpose_matrix(double a[DIM][DIM]);




#endif


