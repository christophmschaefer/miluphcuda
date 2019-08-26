/**
 * @author      Christoph Burger and Christoph Schaefer
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
/* Christoph Burger 22/Jun/2018
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
#if MORE_ANEOS_OUTPUT
extern double ***g_aneos_T;
extern double ***g_aneos_cs;
extern double ***g_aneos_entropy;
extern int ***g_aneos_phase_flag;
#endif



#define ERRORTEXT(x) {fprintf(stderr,x); exit(1);}
#define ERRORVAR(x,y) {fprintf(stderr,x,y); exit(1);}
#define ERRORVAR2(x,y,z) {fprintf(stderr,x,y,z); exit(1);}


#if MORE_ANEOS_OUTPUT
void initialize_aneos_eos_full(const char *aneos_tab_file, int n_rho, int n_e, double *rho, double *e, double **p, double **T, double **cs, double **entropy, int **phase_flag);
#else
void initialize_aneos_eos_basic(const char *aneos_tab_file, int n_rho, int n_e, double *rho, double *e, double **p);
#endif
void free_aneos_memory();
__device__ int array_index(double x, double* array, int n);
__device__ double bilinear_interpolation_from_linearized(double x, double y, double* table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y);
#if MORE_ANEOS_OUTPUT
int array_index_host(double x, double* array, int n);
double bilinear_interpolation_from_matrix(double x, double y, double** table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y);
int discrete_value_table_lookup_from_matrix(double x, double y, int** table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y);
#endif


#endif

