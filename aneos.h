/**
 * @author      Christoph Burger and Christoph Schaefer
 * @brief       Declarations for handling tabulated equations of state.
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

#ifndef _ANEOS_H
#define _ANEOS_H

#include "parameter.h"


extern int *g_eos_is_aneos;
extern const char **g_aneos_tab_file;
extern int *g_aneos_n_rho;
extern int *g_aneos_n_e;
extern double *g_aneos_rho_0;
extern double *g_aneos_bulk_cs;
extern double **g_aneos_rho;
extern double **g_aneos_e;
extern double ***g_aneos_p;
extern double ***g_aneos_cs;
#if MORE_ANEOS_OUTPUT
extern double ***g_aneos_T;
extern double ***g_aneos_entropy;
extern int ***g_aneos_phase_flag;
#endif


#if MORE_ANEOS_OUTPUT
/**
 * @brief Initializes EoS lookup table for one material.
 * @details Fully initializes tabulated EoS for one material by reading the whole lookup table from file.
 * Reads p, T, cs, entropy, phase-flag, as a function of rho and e.
 */
void initialize_aneos_eos_full(const char *aneos_tab_file, int n_rho, int n_e, double *rho, double *e, double **p, double **T, double **cs, double **entropy, int **phase_flag);


#else
/**
 * @brief Initializes EoS lookup table for one material.
 * @details Initializes tabulated EoS for one material by reading only basic quantities from lookup table file.
 * Reads p, cs, as a function of rho and e.
 */
void initialize_aneos_eos_basic(const char *aneos_tab_file, int n_rho, int n_e, double *rho, double *e, double **p, double **cs);
#endif


/** @brief Frees (global) ANEOS memory on the host. */
void free_aneos_memory();


/**
 * @brief Find index in ordered array.
 * @details Uses simple bisection to find index `i` in ordered array (length `n`) that satisfies `array[i] <= x < array[i+1]`.
 * Returns -1 if `x` lies outside the array-covered values.
 */
__device__ int array_index(double x, double* array, int n);


/**
 * @brief Bilinear interpolation of lookup table values.
 * @details Performs bilinear interpolation (2D linear interpolation) of values in `table`, which correspond
 * to x- and y-values in `xtab` and `ytab`.
 * If (x,y) lies outside the table then `ix<0 || iy<0` and the table values are (somewhat linearly) extrapolated.
 * 
 * @param table is linearized array, where rows (connected y-values for a single x-value) are saved successively.
 * @param ix holds the index that satisfies `xtab[ix] <= x < xtab[ix+1]` (same for `iy`).
 * @param n_x holds the length of a row of x-values for a single y-value (same for `n_y`).
 * @param pid is the index in the particle array.
 */
__device__ double bilinear_interpolation_from_linearized(double x, double y, double* table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y, int pid);


/**
 * @brief Bilinear interpolation of lookup table values + derivatives.
 * @details Performs bilinear interpolation (2D linear interpolation) of values in `table`, which correspond
 * to x- and y-values in `xtab` and `ytab`.
 * If (x,y) lies outside the table then `ix<0 || iy<0` and the table values are (somewhat linearly) extrapolated.
 * 
 * @param table is linearized array where rows (connected y-values for a single x-value) are saved successively.
 * @param ix holds the index that satisfies `xtab[ix] <= x < xtab[ix+1]` (same for `iy`).
 * @param n_x holds the length of a row of x-values for a single y-value (same for `n_y`).
 * @param z is the interpolated value.
 * @param dz_dx is the interpolated derivative in x-direction (same for `dz_dy`).
 * @param pid is the index in the particle array.
 */
__device__ void bilinear_interpolation_from_linearized_plus_derivatives(double x, double y, double* table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y, double* z, double* dz_dx, double* dz_dy, int pid);


#if MORE_ANEOS_OUTPUT
/**
 * @brief Find index in ordered array.
 * @details Uses simple bisection to find index `i` in ordered array (length `n`) that satisfies `array[i] <= x < array[i+1]`.
 * Returns -1 if `x` lies outside the array-covered values.
 */
int array_index_host(double x, double* array, int n);


/**
 * @brief Bilinear interpolation of lookup table values.
 * @details Performs bilinear interpolation (2D linear interpolation) of values in `table`, which correspond
 * to x- and y-values in `xtab` and `ytab`.
 * If (x,y) lies outside the table then `ix<0 || iy<0` and the table values are (somewhat linearly) extrapolated.
 * 
 * @param table is 2D array holding the lookup table.
 * @param ix holds the index that satisfies `xtab[ix] <= x < xtab[ix+1]` (same for `iy`).
 * @param n_x holds the length of a row of x-values for a single y-value (same for `n_y`).
 * @param pid is the index in the particle array.
 */
double bilinear_interpolation_from_matrix(double x, double y, double** table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y, int pid);


/**
 * @brief Find value in integer lookup table.
 * @details Discrete (int) values in `table` correspond to x- and y-values (doubles) in `xtab` and `ytab`.
 * This returns the closest corner (in the x-y-plane) of the respective cell of `table`.
 * If (x,y) lies outside the table then `ix<0 || iy<0` and the closest (in the x-y-plane) value of `table` is returned.
 * 
 * @param table is 2D integer array holding the lookup table
 * @param ix holds the index that satisfies `xtab[ix] <= x < xtab[ix+1]` (same for `iy`).
 * @param n_x holds the length of a row of x-values for a single y-value (same for `n_y`).
 * @param pid is the index in the particle array.
 */
int discrete_value_table_lookup_from_matrix(double x, double y, int** table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y, int pid);
#endif


#define ERRORTEXT(x) {fprintf(stderr,x); exit(1);}
#define ERRORVAR(x,y) {fprintf(stderr,x,y); exit(1);}
#define ERRORVAR2(x,y,z) {fprintf(stderr,x,y,z); exit(1);}
#define ERRORVAR3(x,y,z,a) {fprintf(stderr,x,y,z,a); exit(1);}
#define ERRORVAR4(x,y,z,a,b) {fprintf(stderr,x,y,z,a,b); exit(1);}

#endif
