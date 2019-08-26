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


#include "timeintegration.h"
#include "miluph.h"
#include "parameter.h"
#include "linalg.h"



__device__ void copy_matrix(double src[DIM][DIM], double dst[DIM][DIM])
{
    int i, j;

    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            dst[i][j] = src[i][j];
        }
    }

}

__device__ void transpose_matrix(double m[DIM][DIM])
{
    int i, j;
    double mt[DIM][DIM];
    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            mt[j][i] = m[i][j];
        }
    }
    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            m[i][j] = mt[i][j];
        }
    }
}

// calculates C = A B and stores in C
__device__  void multiply_matrix(double A[DIM][DIM], double B[DIM][DIM], double C[DIM][DIM])
{
    int i, j, k;

    double vprime[DIM][DIM];

    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            vprime[i][j] = 0.0;
        }
    }

    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            for (k = 0; k < DIM; k++) {
                vprime[i][j] += A[i][k]*B[k][j];
            }
        }
    }
    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            C[i][j] = vprime[i][j];
        }
    }

}

__device__ void identity_matrix(double A[DIM][DIM])
{
    int i, j;
    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            A[i][j] = 0.0;
        }
        A[i][i] = 1.0;
    }
}





// returns the indices of the greatest non-diagonal element of M
__device__ int max_Matrix(double M[DIM][DIM], int *e, int *f, double *elmax)
{
    int i, j;
    double max = 0.0;
    int ierror = 1;

    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            if (i == j)
                continue;
            if (fabs(M[i][j]) >= max) {
                max = fabs(M[i][j]);
                *e = i;
                *f = j;
                ierror = 0;
            }
        }
    }
    *elmax = max;
    return ierror;
}


/*
 * help function for the jacobi method
 * returns: M' = A^T M A, and A_ef = s = -A_ef, A_ee = A_ff = c
 */
__device__ void rotate_matrix(volatile double m[DIM][DIM], volatile double c, volatile double s, volatile int e,
volatile int f)
{
    int i, j;
    volatile double mprime[DIM][DIM];

    /* first copy the matrix */
    for (i = 0; i < DIM; i++)
        for (j = 0; j < DIM; j++)
            mprime[i][j] = m[i][j];

    /* now the elements that change */
    mprime[e][e] = c*c*m[e][e] + s*s*m[f][f] - 2*s*c*m[e][f];
    mprime[f][f] = c*c*m[f][f] + s*s*m[e][e] + 2*s*c*m[e][f];
    mprime[e][f] = (c*c-s*s)*m[e][f] + s*c*(m[e][e]-m[f][f]);
    mprime[f][e] = mprime[e][f];

    /* the other elements in columns and rows e, f*/
    /* actually, this is only one in 3D and 0 in 2D */
    for (i = 0; i < DIM; i++) {
        if (i == f || i == e)
            continue;
        mprime[e][i] = c*m[i][e] - s*m[i][f];
        mprime[i][e] = mprime[e][i];
        mprime[f][i] = c*m[i][f] + s*m[i][e];
        mprime[i][f] = mprime[f][i];
    }

    /* set the matrix to the rotated one */
    for (i = 0; i < DIM; i++)
        for (j = 0; j < DIM; j++)
            m[i][j] = mprime[i][j];
}



/*
 * computes all eigenvalues and eigenvectors of the _symmetric_ matrix M
 * using the jacobi method and stores them in eigenvals and the eigenvecs as columns
 * in the transformation matrix v
 *
 * returns the number of iterations
 */
__device__ int calculate_all_eigenvalues(double M[DIM][DIM], double eigenvalues[DIM], double v[DIM][DIM]) {
    int i, j;
    double diagM[DIM][DIM] = {0.0, };
    double c, s, t, thta;
    double A[DIM][DIM];
    double vtmp[DIM][DIM];
    int e, f;
    int error;
    double max = -1e300;
    int nit = 0;
    i = j = e = f = 0;
    c = s = t = thta = 0.0;
    error = 0;

#define EPS_JACOBI 1e-10

    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            diagM[i][j] = M[i][j];
            v[i][j] = 0.0;
        }
        v[i][i] = 1.0;
    }

    do {
        nit++;
        error = max_Matrix(diagM, &e, &f, &max);
        if (error) {
            printf("No maximum element found.\n");
        }
        if (max > 0) {
            // rotate matrix
            thta = (diagM[f][f] - diagM[e][e])/(2*diagM[e][f]);
            if (thta < 0)
                t = -1./(fabs(thta) + sqrt(thta*thta+1));
            else
                t = 1./(fabs(thta) + sqrt(thta*thta+1));
            // the elements of the rotation matrix
            c = 1./(sqrt(t*t+1));
            s = t*c;
            // do diagM' = A^T diagM A
            rotate_matrix(diagM, c, s, e, f);
            identity_matrix(A);
            A[e][e] = c;
            A[f][f] = c;
            A[e][f] = -s;
            A[f][e] = s;
            // calculate the eigenvectors
            multiply_matrix(v, A, vtmp);
            copy_matrix(vtmp, v);
        }
    } while (max > EPS_JACOBI);

    for (i = 0; i < DIM; i++) {
        eigenvalues[i] = diagM[i][i];
    }
    return nit;
}





/*
 * computes the eigenvalues of the _symmetric_ matrix M
 * using the jacobi method
 * returns the greatest eigenvalue
 */
__device__ double calculateMaxEigenvalue(double M[DIM][DIM]) {
    int i, j;
    double diagM[DIM][DIM] = {0.0, };
    double c, s, t, thta;
    int e, f;
    int error;
    double max;
    double max_ev;
    int nit = 0;
    i = j = e = f = 0;
    c = s = t = thta = 0.0;
    max = max_ev = 0;
    error = 0;


#define EPS_JACOBI 1e-10

    for (i = 0; i < DIM; i++)
        for (j = 0; j < DIM; j++)
            diagM[i][j] = M[i][j];

    do {
        nit++;
        error = max_Matrix(diagM, &e, &f, &max);
        if (error) {
            printf("No maximum element found.\n");
        }
        if (max > 0) {
            // rotate matrix
            thta = (diagM[f][f] - diagM[e][e])/(2*diagM[e][f]);
            if (thta < 0)
                t = -1./(fabs(thta) + sqrt(thta*thta+1));
            else
                t = 1./(fabs(thta) + sqrt(thta*thta+1));
            // the elements of the rotation matrix
            c = 1./(sqrt(t*t+1));
            s = t*c;
            // do diagM' = A^T diagM A
            rotate_matrix(diagM, c, s, e, f);
        }
    } while (max > EPS_JACOBI || nit < 5);

    max_ev = diagM[0][0];
    for (i = 1; i < DIM; i++) {
        if (diagM[i][i] > max_ev) {
            max_ev = diagM[i][i];
        }
    }
    return max_ev;
}

__device__ double det2x2(double a, double b, double c, double d) {
    return a*d-c*b;
}

__device__ int invertMatrix(double *m, double *inverted) {
    double det;
#if (DIM == 2)
    double a, b, c, d;
    a = m[0*DIM+0];
    b = m[0*DIM+1];
    c = m[1*DIM+0];
    d = m[1*DIM+1];

    det = det2x2(a,b,c,d);
  //  if (det < 1e-8) return -1;
   // if (det < 1e-10) det = 1e-10;
    det = 1./det;

    inverted[0*DIM+0] = det*d;
    inverted[0*DIM+1] = -det*b;
    inverted[1*DIM+0] = -det*c;
    inverted[1*DIM+1] = det*a;
#elif (DIM == 3)
    det = m[0 * DIM + 0] * (m[1 * DIM + 1] * m[2 * DIM + 2] - m[2 * DIM + 1] * m[1 * DIM + 2])
        - m[0 * DIM + 1] * (m[1 * DIM + 0] * m[2 * DIM + 2] - m[1 * DIM + 2] * m[2 * DIM + 0])
        + m[0 * DIM + 2] * (m[1 * DIM + 0] * m[2 * DIM + 1] - m[1 * DIM + 1] * m[2 * DIM + 0]);

    // inverse determinante

    if (det < 1e-8) return -1;
    det = 1.0 / det;

    inverted[0*DIM+0] = (m[1*DIM+ 1] * m[2*DIM+ 2] - m[2*DIM+ 1] * m[1*DIM+ 2]) * det;
    inverted[0*DIM+1] = (m[0*DIM+ 2] * m[2*DIM+ 1] - m[0*DIM+ 1] * m[2*DIM+ 2]) * det;
    inverted[0*DIM+2] = (m[0*DIM+ 1] * m[1*DIM+ 2] - m[0*DIM+ 2] * m[1*DIM+ 1]) * det;
    inverted[1*DIM+0] = (m[1*DIM+ 2] * m[2*DIM+ 0] - m[1*DIM+ 0] * m[2*DIM+ 2]) * det;
    inverted[1*DIM+1] = (m[0*DIM+ 0] * m[2*DIM+ 2] - m[0*DIM+ 2] * m[2*DIM+ 0]) * det;
    inverted[1*DIM+2] = (m[1*DIM+ 0] * m[0*DIM+ 2] - m[0*DIM+ 0] * m[1*DIM+ 2]) * det;
    inverted[2*DIM+0] = (m[1*DIM+ 0] * m[2*DIM+ 1] - m[2*DIM+ 0] * m[1*DIM+ 1]) * det;
    inverted[2*DIM+1] = (m[2*DIM+ 0] * m[0*DIM+ 1] - m[0*DIM+ 0] * m[2*DIM+ 1]) * det;
    inverted[2*DIM+2] = (m[0*DIM+ 0] * m[1*DIM+ 1] - m[1*DIM+ 0] * m[0*DIM+ 1]) * det;
#endif

    return 1;
}
