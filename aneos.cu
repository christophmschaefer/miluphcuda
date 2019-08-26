/**
 * @author      Christoph Burger
 *
 * @section     LICENSE
 * Copyright (c) 2019 Christoph Burger, Christoph Schaefer
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

#include <stdio.h>
#include <stdlib.h>
#include "aneos.h"
#include "timeintegration.h"
#include "parameter.h"



// global variables (on the host)
int *g_eos_is_aneos;    // TRUE if eos of material is ANEOS
const char **g_aneos_tab_file;
int *g_aneos_n_rho;
int *g_aneos_n_e;
double *g_aneos_rho_0;
double *g_aneos_bulk_cs;
double **g_aneos_rho;
double **g_aneos_e;
double ***g_aneos_p;
#if MORE_ANEOS_OUTPUT
double ***g_aneos_T;
double ***g_aneos_cs;
double ***g_aneos_entropy;
int ***g_aneos_phase_flag;
#endif



#if MORE_ANEOS_OUTPUT
void initialize_aneos_eos_full(const char *aneos_tab_file, int n_rho, int n_e, double *rho, double *e, double **p, double **T, double **cs, double **entropy, int **phase_flag)
/* Fully initializes ANEOS EOS for one material by reading full lookup table from file.*/
{
    int i,j;
    FILE *f;
    
    // open file containing ANEOS lookup table
    if ( (f = fopen(aneos_tab_file,"r")) == NULL )
        ERRORVAR("FILE ERROR! Cannot open %s for reading!\n", aneos_tab_file)
    
    // read rho and e (vectors) and p, T, cs, entropy and phase-flag (matrices) from file
    for(i=0; i<3; i++)
        fscanf(f, "%*[^\n]\n");     // ignore first three lines
    if ( fscanf(f, "%le %le %le %le %le %le %d%*[^\n]\n", rho, e, &p[0][0], &T[0][0], &cs[0][0], &entropy[0][0], &phase_flag[0][0] ) != 7 )
        ERRORVAR("ERROR! Something's wrong with the ANEOS lookup table in %s\n", aneos_tab_file)
    for(j=1; j<n_e; j++)
        fscanf(f, "%*le %le %le %le %le %le %d%*[^\n]\n", &e[j], &p[0][j], &T[0][j], &cs[0][j], &entropy[0][j], &phase_flag[0][j] );
    for(i=1; i<n_rho; i++) {
        fscanf(f, "%le %*le %le %le %le %le %d%*[^\n]\n", &rho[i], &p[i][0], &T[i][0], &cs[i][0], &entropy[i][0], &phase_flag[i][0] );
        for(j=1; j<n_e; j++)
            fscanf(f, "%*le %*le %le %le %le %le %d%*[^\n]\n", &p[i][j], &T[i][j], &cs[i][j], &entropy[i][j], &phase_flag[i][j] );
    }
    fclose(f);
}



#else
void initialize_aneos_eos_basic(const char *aneos_tab_file, int n_rho, int n_e, double *rho, double *e, double **p)
/* Initializes basic quantities of the ANEOS EOS for one material by reading only these quantities from the lookup table file.*/
{
    int i,j;
    FILE *f;
    
    // open file containing ANEOS lookup table
    if ( (f = fopen(aneos_tab_file,"r")) == NULL )
        ERRORVAR("FILE ERROR! Cannot open %s for reading!\n", aneos_tab_file)
    
    // read rho and e (vectors) and p (matrix) from file
    for(i=0; i<3; i++)
        fscanf(f, "%*[^\n]\n");     // ignore first three lines
    if ( fscanf(f, "%le %le %le%*[^\n]\n", rho, e, &p[0][0] ) != 3 )
        ERRORVAR("ERROR! Something's wrong with the ANEOS lookup table in %s\n", aneos_tab_file)
    for(j=1; j<n_e; j++)
        fscanf(f, "%*le %le %le%*[^\n]\n", &e[j], &p[0][j] );
    for(i=1; i<n_rho; i++) {
        fscanf(f, "%le %*le %le%*[^\n]\n", &rho[i], &p[i][0] );
        for(j=1; j<n_e; j++)
            fscanf(f, "%*le %*le %le%*[^\n]\n", &p[i][j] );
    }
    fclose(f);
}
#endif



void free_aneos_memory()
/* Frees (global) ANEOS memory on the host */
{
    int i,j;
    
    for (i=0; i<numberOfMaterials; i++) {
        if (g_eos_is_aneos[i]) {
            free(g_aneos_rho[i]);
            free(g_aneos_e[i]);
            for(j = 0; j < g_aneos_n_rho[i]; j++) {
                free(g_aneos_p[i][j]);
#if MORE_ANEOS_OUTPUT
                free(g_aneos_T[i][j]);
                free(g_aneos_cs[i][j]);
                free(g_aneos_entropy[i][j]);
                free(g_aneos_phase_flag[i][j]);
#endif
            }
            free(g_aneos_p[i]);
#if MORE_ANEOS_OUTPUT
            free(g_aneos_T[i]);
            free(g_aneos_cs[i]);
            free(g_aneos_entropy[i]);
            free(g_aneos_phase_flag[i]);
#endif
        }
    }
    free(g_aneos_rho);
    free(g_aneos_e);
    free(g_aneos_p);
#if MORE_ANEOS_OUTPUT
    free(g_aneos_T);
    free(g_aneos_cs);
    free(g_aneos_entropy);
    free(g_aneos_phase_flag);
#endif
    free(g_eos_is_aneos);
    free(g_aneos_tab_file);
    free(g_aneos_n_rho);
    free(g_aneos_n_e);
    free(g_aneos_rho_0);
    free(g_aneos_bulk_cs);
}



__device__ int array_index(double x, double* array, int n)
/* Uses simple bisection to find the index i in an ordered array (length n)
 * that satisfies 'array[i] <= x < array[i+1]'. If x lies outside the array-covered values it returns -1.
 */
{
    int i,i1,i2;    // current index and its lower and upper bound
    
    // return -1 if x lies outside the array-covered values
    if( x < array[0] || x >= array[n-1])
        return(-1);
    
    i1 = 0;
    i2 = n-1;
    do {
        i = (int)( (double)(i1+i2)/2.0 );
        if( array[i] <= x )
            i1 = i;    // 'i' becomes new lower bound
        else
            i2 = i;    // 'i' becomes new upper bound
    }
    while( (i2-i1)>1 );
    
    return(i1);
}



#if MORE_ANEOS_OUTPUT
int array_index_host(double x, double* array, int n)
/* Uses simple bisection to find the index i in an ordered array (length n)
 * that satisfies 'array[i] <= x < array[i+1]'. If x lies outside the array-covered values it returns -1.
 */
{
    int i,i1,i2;    // current index and its lower and upper bound
    
    // return -1 if x lies outside the array-covered values
    if( x < array[0] || x >= array[n-1])
        return(-1);
    
    i1 = 0;
    i2 = n-1;
    do {
        i = (int)( (double)(i1+i2)/2.0 );
        if( array[i] <= x )
            i1 = i;    // 'i' becomes new lower bound
        else
            i2 = i;    // 'i' becomes new upper bound
    }
    while( (i2-i1)>1 );
    
    return(i1);
}
#endif



__device__ double bilinear_interpolation_from_linearized(double x, double y, double* table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y)
/* Performs bilinear interpolation (2d lin. interp.) of values in 'table' which correspond to x- and y-values in xtab and ytab.
 * table is a linearized array where rows (connected y-values for a single x-value) are saved successively.
 * The target values are x and y. ix holds the index that satisfies 'xtab[ix] <= x < xtab[ix+1]' (similar for iy).
 * n_x holds the length of a row of x-values for a single y-value (similar for n_y).
 * If (x,y) lies outside the table then ix<0 || iy<0 and the table values are (somewhat linearly) extrapolated.
 */
{
    double normx, normy, a, b, p;
    
    // if (x,y) lies outside table then extrapolate (somewhat linearly) and print a warning
    if( ix < 0 || iy < 0 )
    {
        if( ix < 0 && iy < 0 )  // (x,y) lies in one of the 4 "corners"
        {
            if( x < xtab[0] && y < ytab[0] )
            {
                normx = (xtab[0]-x) / (xtab[1]-xtab[0]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                normy = (ytab[0]-y) / (ytab[1]-ytab[0]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = table[0] + normx*(table[0]-table[n_y]) + normy*(table[0]-table[1]);
            }
            else if( x < xtab[0] && y >= ytab[n_y-1] )
            {
                normx = (xtab[0]-x) / (xtab[1]-xtab[0]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                normy = (y-ytab[n_y-1]) / (ytab[n_y-1]-ytab[n_y-2]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = table[n_y-1] + normx*(table[n_y-1]-table[2*n_y-1]) + normy*(table[n_y-1]-table[n_y-2]);
            }
            else if( x >= xtab[n_x-1] && y < ytab[0] )
            {
                normx = (x-xtab[n_x-1]) / (xtab[n_x-1]-xtab[n_x-2]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                normy = (ytab[0]-y) / (ytab[1]-ytab[0]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = table[(n_x-1)*n_y] + normx*(table[(n_x-1)*n_y]-table[(n_x-2)*n_y]) + normy*(table[(n_x-1)*n_y]-table[(n_x-1)*n_y+1]);
            }
            else if( x >= xtab[n_x-1] && y >= ytab[n_y-1] )
            {
                normx = (x-xtab[n_x-1]) / (xtab[n_x-1]-xtab[n_x-2]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                normy = (y-ytab[n_y-1]) / (ytab[n_y-1]-ytab[n_y-2]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = table[n_x*n_y-1] + normx*(table[n_x*n_y-1]-table[(n_x-1)*n_y-1]) + normy*(table[n_x*n_y-1]-table[n_x*n_y-2]);
            }
            else
                printf("WARNING: Some odd behavior during extrapolation from ANEOS table encountered for rho = %e and e = %e !\n", x, y);
        }
        else if( ix < 0 )
        {
            normy = (y-ytab[iy]) / (ytab[iy+1]-ytab[iy]);
            if( x < xtab[0] )
            {
                // linear interpolation in y-direction at xtab[0] and xtab[1]
                a = table[iy] + normy*(table[iy+1]-table[iy]);
                b = table[n_y+iy] + normy*(table[n_y+iy+1]-table[n_y+iy]);
                // linear extrapolation in x-direction from a and b
                normx = (x-xtab[0]) / (xtab[1]-xtab[0]);    // (always negative) distance from table end, normalized to x-spacing between 2 outermost table values
                p = a + normx*(b-a);
            }
            else if( x >= xtab[n_x-1] )
            {
                // linear interpolation in y-direction at xtab[n_x-1] and xtab[n_x-2]
                a = table[(n_x-1)*n_y+iy] + normy*(table[(n_x-1)*n_y+iy+1]-table[(n_x-1)*n_y+iy]);
                b = table[(n_x-2)*n_y+iy] + normy*(table[(n_x-2)*n_y+iy+1]-table[(n_x-2)*n_y+iy]);
                // linear extrapolation in x-direction from a and b
                normx = (x-xtab[n_x-1]) / (xtab[n_x-1]-xtab[n_x-2]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                p = a + normx*(a-b);
            }
            else
                printf("WARNING: Some odd behavior during extrapolation from ANEOS table encountered for rho = %e and e = %e !\n", x, y);
        }
        else if( iy < 0 )
        {
            normx = (x-xtab[ix]) / (xtab[ix+1]-xtab[ix]);
            if( y < ytab[0] )
            {
                // linear interpolation in x-direction at ytab[0] and ytab[1]
                a = table[ix*n_y] + normx*(table[(ix+1)*n_y]-table[ix*n_y]);
                b = table[ix*n_y+1] + normx*(table[(ix+1)*n_y+1]-table[ix*n_y+1]);
                // linear extrapolation in y-direction from a and b
                normy = (y-ytab[0]) / (ytab[1]-ytab[0]);    // (always negative) distance from table end, normalized to y-spacing between 2 outermost table values
                p = a + normy*(b-a);
            }
            else if( y >= ytab[n_y-1] )
            {
                // linear interpolation in x-direction at ytab[n_y-1] and ytab[n_y-2]
                a = table[(ix+1)*n_y-1] + normx*(table[(ix+2)*n_y-1]-table[(ix+1)*n_y-1]);
                b = table[(ix+1)*n_y-2] + normx*(table[(ix+2)*n_y-2]-table[(ix+1)*n_y-2]);
                // linear extrapolation in y-direction from a and b
                normy = (y-ytab[n_y-1]) / (ytab[n_y-1]-ytab[n_y-2]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = a + normy*(a-b);
            }
            else
                printf("WARNING: Some odd behavior during extrapolation from ANEOS table encountered for rho = %e and e = %e !\n", x, y);
        }
        else
            printf("WARNING: Some odd behavior during extrapolation from ANEOS table encountered for rho = %e and e = %e !\n", x, y);
        
        printf("WARNING: At least one of rho = %e and e = %e is out of ANEOS lookup table range! Use extrapolated f(rho,e) = %e\n", x, y, p);
        return(p);
    }
    
    // calculate normalized distances of x and y from (lower) table values
    normx = (x-xtab[ix]) / (xtab[ix+1]-xtab[ix]);
    normy = (y-ytab[iy]) / (ytab[iy+1]-ytab[iy]);

    // linear interpolation in x-direction at ytab[iy] and ytab[iy+1]
    a = table[ix*n_y+iy] + normx*(table[(ix+1)*n_y+iy]-table[ix*n_y+iy]);
    b = table[ix*n_y+iy+1] + normx*(table[(ix+1)*n_y+iy+1]-table[ix*n_y+iy+1]);
    
    // linear interpolation in y-direction between a and b
    return( a + normy*(b-a) );
}



#if MORE_ANEOS_OUTPUT
double bilinear_interpolation_from_matrix(double x, double y, double** table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y)
// Performs bilinear interpolation (2d lin. interp.) of values in 'table' which correspond to x- and y-values in 'xtab' and 'ytab'.
// The target values are 'x' and 'y'. 'ix' holds the index that satisfies 'xtab[ix] <= x < xtab[ix+1]' (similar for iy).
// 'n_x' holds the length of a row of x-values for a single y-value (similar for n_y).
// If (x,y) lies outside the table then ix<0 || iy<0 and the table values are (somewhat linearly) extrapolated.
{
    double normx = -1.0, normy = -1.0;
    double a, b, p = -1.0;
//    FILE *f;
    
    
    // if (x,y) lies outside table then extrapolate (somewhat linearly) and print a warning
    if( ix < 0 || iy < 0 )
    {
        if( ix < 0 && iy < 0 )  // (x,y) lies in one of the 4 "corners"
        {
            if( x < xtab[0] && y < ytab[0] )
            {
                normx = (xtab[0]-x) / (xtab[1]-xtab[0]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                normy = (ytab[0]-y) / (ytab[1]-ytab[0]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = table[0][0] + normx*(table[0][0]-table[1][0]) + normy*(table[0][0]-table[0][1]);
            }
            else if( x < xtab[0] && y >= ytab[n_y-1] )
            {
                normx = (xtab[0]-x) / (xtab[1]-xtab[0]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                normy = (y-ytab[n_y-1]) / (ytab[n_y-1]-ytab[n_y-2]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = table[0][n_y-1] + normx*(table[0][n_y-1]-table[1][n_y-1]) + normy*(table[0][n_y-1]-table[0][n_y-2]);
            }
            else if( x >= xtab[n_x-1] && y < ytab[0] )
            {
                normx = (x-xtab[n_x-1]) / (xtab[n_x-1]-xtab[n_x-2]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                normy = (ytab[0]-y) / (ytab[1]-ytab[0]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = table[n_x-1][0] + normx*(table[n_x-1][0]-table[n_x-2][0]) + normy*(table[n_x-1][0]-table[n_x-1][1]);
            }
            else if( x >= xtab[n_x-1] && y >= ytab[n_y-1] )
            {
                normx = (x-xtab[n_x-1]) / (xtab[n_x-1]-xtab[n_x-2]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                normy = (y-ytab[n_y-1]) / (ytab[n_y-1]-ytab[n_y-2]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = table[n_x-1][n_y-1] + normx*(table[n_x-1][n_y-1]-table[n_x-2][n_y-1]) + normy*(table[n_x-1][n_y-1]-table[n_x-1][n_y-2]);
            }
            else
                ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table encountered for rho = %e and e = %e !\n", x, y)
        }
        else if( ix < 0 )
        {
            normy = (y-ytab[iy]) / (ytab[iy+1]-ytab[iy]);
            if( x < xtab[0] )
            {
                // linear interpolation in y-direction at xtab[0] and xtab[1]
                a = table[0][iy] + normy*(table[0][iy+1]-table[0][iy]);
                b = table[1][iy] + normy*(table[1][iy+1]-table[1][iy]);
                // linear extrapolation in x-direction from a and b
                normx = (x-xtab[0]) / (xtab[1]-xtab[0]);    // (always negative) distance from table end, normalized to x-spacing between 2 outermost table values
                p = a + normx*(b-a);
            }
            else if( x >= xtab[n_x-1] )
            {
                // linear interpolation in y-direction at xtab[n_x-1] and xtab[n_x-2]
                a = table[n_x-1][iy] + normy*(table[n_x-1][iy+1]-table[n_x-1][iy]);
                b = table[n_x-2][iy] + normy*(table[n_x-2][iy+1]-table[n_x-2][iy]);
                // linear extrapolation in x-direction from a and b
                normx = (x-xtab[n_x-1]) / (xtab[n_x-1]-xtab[n_x-2]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                p = a + normx*(a-b);
            }
            else
                ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table encountered for rho = %e and e = %e !\n", x, y)
        }
        else if( iy < 0 )
        {
            normx = (x-xtab[ix]) / (xtab[ix+1]-xtab[ix]);
            if( y < ytab[0] )
            {
                // linear interpolation in x-direction at ytab[0] and ytab[1]
                a = table[ix][0] + normx*(table[ix+1][0]-table[ix][0]);
                b = table[ix][1] + normx*(table[ix+1][1]-table[ix][1]);
                // linear extrapolation in y-direction from a and b
                normy = (y-ytab[0]) / (ytab[1]-ytab[0]);    // (always negative) distance from table end, normalized to y-spacing between 2 outermost table values
                p = a + normy*(b-a);
            }
            else if( y >= ytab[n_y-1] )
            {
                // linear interpolation in x-direction at ytab[n_y-1] and ytab[n_y-2]
                a = table[ix][n_y-1] + normx*(table[ix+1][n_y-1]-table[ix][n_y-1]);
                b = table[ix][n_y-2] + normx*(table[ix+1][n_y-2]-table[ix][n_y-2]);
                // linear extrapolation in y-direction from a and b
                normy = (y-ytab[n_y-1]) / (ytab[n_y-1]-ytab[n_y-2]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = a + normy*(a-b);
            }
            else
                ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table encountered for rho = %e and e = %e !\n", x, y)
        }
        else
            ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table encountered for rho = %e and e = %e !\n", x, y)
        
        // write a warning to warnings file
//        if ( (f = fopen("miluphcuda.warnings", "a")) == NULL )
//            ERRORTEXT("FILE ERROR! Cannot open 'miluphcuda.warnings' for appending!\n")
//        fprintf(f, "WARNING: At least one of rho = %e and e = %e is out of ANEOS lookup table range! Use extrapolated p(rho,e) = %e\n", x, y, p);
//        fclose(f);
        
        return(p);
    }
    
    
    // calculate normalized distances of x and y from (lower) table values
    normx = (x-xtab[ix]) / (xtab[ix+1]-xtab[ix]);
    normy = (y-ytab[iy]) / (ytab[iy+1]-ytab[iy]);
    
    // linear interpolation in x-direction at ytab[iy] and ytab[iy+1]
    a = table[ix][iy] + normx*(table[ix+1][iy]-table[ix][iy]);
    b = table[ix][iy+1] + normx*(table[ix+1][iy+1]-table[ix][iy+1]);
    
    // linear interpolation in y-direction between a and b
    return( a + normy*(b-a) );
    
}   // end function 'bilinear_interpolation()'
#endif



#if MORE_ANEOS_OUTPUT
int discrete_value_table_lookup_from_matrix(double x, double y, int** table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y)
// Discrete (int) values in 'table' correspond to x- and y-values (doubles) in 'xtab' and 'ytab'.
// This function finds the closest "corner" (in the x-y-plane) of the respective cell and returns the value of 'table' in that corner.
// The target values are 'x' and 'y'. 'ix' holds the index that satisfies 'xtab[ix] <= x < xtab[ix+1]' (similar for iy).
// 'n_x' holds the length of a row of x-values for a single y-value (similar for n_y).
// If (x,y) lies outside the table then ix<0 || iy<0 and the closest (in the x-y-plane) value of 'table' is returned.
{
    int phase_flag = -1;
    double normx = -1.0, normy = -1.0;
//    FILE *f;
    
    
    // if (x,y) lies outside table then find the closest value (in the x-y-plane) of 'table' and print a warning
    if( ix < 0 || iy < 0 )
    {
        if( ix < 0 && iy < 0 )  // (x,y) lies in one of the 4 "corners"
        {
            if( x < xtab[0] && y < ytab[0] )    // "lower left" corner
            {
                phase_flag = table[0][0];
            }
            else if( x < xtab[0] && y >= ytab[n_y-1] )  // "upper left" corner
            {
                phase_flag = table[0][n_y-1];
            }
            else if( x >= xtab[n_x-1] && y < ytab[0] )  // "lower right" corner
            {
                phase_flag = table[n_x-1][0];
            }
            else if( x >= xtab[n_x-1] && y >= ytab[n_y-1] ) // "upper right" corner
            {
                phase_flag = table[n_x-1][n_y-1];
            }
            else
                ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %e and e = %e !\n", x, y)
        }
        else if( ix < 0 )
        {
            normy = (y-ytab[iy]) / (ytab[iy+1]-ytab[iy]);
            if( normy >= 0.5 && normy <= 1.0 )
            {
                if( x < xtab[0] )
                {
                    phase_flag = table[0][iy+1];
                }
                else if( x >= xtab[n_x-1] )
                {
                    phase_flag = table[n_x-1][iy+1];
                }
                else
                    ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %e and e = %e !\n", x, y)
            }
            else if( normy < 0.5 && normy >= 0.0 )
            {
                if( x < xtab[0] )
                {
                    phase_flag = table[0][iy];
                }
                else if( x >= xtab[n_x-1] )
                {
                    phase_flag = table[n_x-1][iy];
                }
                else
                    ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %e and e = %e !\n", x, y)
            }
            else
                ERRORVAR("ERROR! 'normy' = %e (is not in [0,1]) in 'discrete_value_table_lookup()' ...\n", normy)
        }
        else if( iy < 0 )
        {
            normx = (x-xtab[ix]) / (xtab[ix+1]-xtab[ix]);
            if( normx >= 0.5 && normx <= 1.0 )
            {
                if( y < ytab[0] )
                {
                    phase_flag = table[ix+1][0];
                }
                else if( y >= ytab[n_y-1] )
                {
                    phase_flag = table[ix+1][n_y-1];
                }
                else
                    ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %e and e = %e !\n", x, y)
            }
            else if( normx < 0.5 && normx >= 0.0 )
            {
                if( y < ytab[0] )
                {
                    phase_flag = table[ix][0];
                }
                else if( y >= ytab[n_y-1] )
                {
                    phase_flag = table[ix][n_y-1];
                }
                else
                    ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %e and e = %e !\n", x, y)
            }
            else
                ERRORVAR("ERROR! 'normx' = %e (is not in [0,1]) in 'discrete_value_table_lookup()' ...\n", normx)
        }
        else
            ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %e and e = %e !\n", x, y)
        
        // write a warning to warnings file
//        if ( (f = fopen("miluphcuda.warnings", "a")) == NULL )
//            ERRORTEXT("FILE ERROR! Cannot open 'miluphcuda.warnings' for appending!\n")
//        fprintf(f, "WARNING: At least one of rho = %e and e = %e is out of ANEOS lookup table range! Use extrapolated phase-flag = %d\n", x, y, phase_flag);
//        fclose(f);
        
        return(phase_flag);
    }
    
    
    // calculate normalized distances of x and y from (lower) table values
    normx = (x-xtab[ix]) / (xtab[ix+1]-xtab[ix]);
    normy = (y-ytab[iy]) / (ytab[iy+1]-ytab[iy]);
    
    // find the closest "corner" (in the x-y-plane) and return respective value of 'table'
    if( normx >= 0.5 && normx <= 1.0 && normy >= 0.5 && normy <= 1.0 )  // "upper right" quadrant of cell
    {
        phase_flag = table[ix+1][iy+1];
    }
    else if( normx >= 0.5 && normx <= 1.0 && normy < 0.5 && normy >= 0.0 )  // "lower right" quadrant of cell
    {
        phase_flag = table[ix+1][iy];
    }
    else if( normx < 0.5 && normx >= 0.0 && normy >= 0.5 && normy <= 1.0 )  // "upper left" quadrant of cell
    {
        phase_flag = table[ix][iy+1];
    }
    else if( normx < 0.5 && normx >= 0.0 && normy < 0.5 && normy >= 0.0 )   // "lower left" quadrant of cell
    {
        phase_flag = table[ix][iy];
    }
    else
        ERRORVAR2("ERROR: Some odd behavior during \"discrete interpolation\" from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %e and e = %e !\n", x, y)
    
    return( phase_flag );
    
}   // end function 'discrete_value_table_lookup()'
#endif

