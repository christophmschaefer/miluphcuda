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
            A[e][f] = s;
            A[f][e] = -s;
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

__device__ void symmetrizeMatrix(double A[DIM][DIM]) {
    for (int i = 0; i < DIM; ++i)
        for (int j = i + 1; j < DIM; ++j) {
            double s = 0.5 * (A[i][j] + A[j][i]);
            A[i][j] = s;
            A[j][i] = s;
        }
}

__device__ int invert_svd(double *m, double *inverted, double threshold_svd) {
    // SVD based matrix inversion for symmetric matrices by Sascha Eckstein
    int i, j, k;
    double A[DIM][DIM];
    double V[DIM][DIM];
    double eigenvalues[DIM];
    double P[DIM][DIM];

    // Load matrix into local memory
    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            A[i][j] = m[i * DIM + j];
        }
    }

    // Since m (and thus A) is symmetric for SPH tensor corrections,
    // we can compute Eigenvalues and Eigenvectors directly on A.
    // This avoids limiting precision by computing A^T * A.
    calculate_all_eigenvalues(A, eigenvalues, V);

    // Compute Pseudo-Inverse: M^+ = V * Sigma^-1 * V^T
    // For a symmetric matrix, SVD singular values are abs(eigenvalues).
    // The pseudo-inverse eigenvalues are 1/eigenvalue.

    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            P[i][j] = 0.0;
        }
    }

    int used_eigenvalues = 0;

    for (k = 0; k < DIM; k++) {
        double ev = eigenvalues[k];
        // Threshold check on absolute value of eigenvalue (singular value)
        if (fabs(ev) > threshold_svd) {
            used_eigenvalues++;
            double inv_ev = 1.0 / ev;
            for (i = 0; i < DIM; i++) {
                for (j = 0; j < DIM; j++) {
                    P[i][j] += inv_ev * V[i][k] * V[j][k];
                }
            }
        }
    }

    // Store result
    // For symmetric matrices, the result P is already the inverse.
    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            inverted[i * DIM + j] = P[i][j];
        }
    }
    return used_eigenvalues;
}

// cms invert matrix using SVD, tests by Sascha show better stability
// 2025-12-19
#define EPSILON_SVD 1e-10
#define MAX_ITER 100

// Matrix operations
__device__ void mat_multiply(double A[DIM][DIM], double B[DIM][DIM], double C[DIM][DIM]) {
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++) {
            C[i][j] = 0;
            for (int k = 0; k < DIM; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

__device__ void mat_transpose(double A[DIM][DIM], double AT[DIM][DIM]) {
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++)
            AT[j][i] = A[i][j];
}

__device__ void mat_copy(double src[DIM][DIM], double dst[DIM][DIM]) {
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++)
            dst[i][j] = src[i][j];
}

__device__ void identity(double I[DIM][DIM]) {
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++)
            I[i][j] = (i == j) ? 1.0 : 0.0;
}

// Jacobi eigenvalue decomposition
__device__ void jacobi(double A[DIM][DIM], double eigenval[DIM], double eigenvec[DIM][DIM]) {
    double S[DIM][DIM], temp[DIM][DIM];
    mat_copy(A, S);
    identity(eigenvec);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Find largest off-diagonal element
        int p = 0, q = 1;
        double max_val = fabs(S[0][1]);

        for (int i = 0; i < DIM; i++)
            for (int j = i + 1; j < DIM; j++)
                if (fabs(S[i][j]) > max_val) {
                    max_val = fabs(S[i][j]);
                    p = i; q = j;
                }

        if (max_val < EPSILON_SVD) break;

        // Compute rotation angle
        double theta = (fabs(S[p][p] - S[q][q]) < EPSILON_SVD) ?
            M_PI / 4.0 : 0.5 * atan2(2.0 * S[p][q], S[q][q] - S[p][p]);

        double c = cos(theta), s = sin(theta);

        // Build rotation matrix
        double R[DIM][DIM];
        identity(R);
        R[p][p] = c; R[q][q] = c;
        R[p][q] = s; R[q][p] = -s;

        // S = R^T * S * R
        double RT[DIM][DIM];
        mat_transpose(R, RT);
        mat_multiply(RT, S, temp);
        mat_multiply(temp, R, S);

        // Accumulate eigenvectors
        mat_multiply(eigenvec, R, temp);
        mat_copy(temp, eigenvec);
    }

    for (int i = 0; i < DIM; i++)
        eigenval[i] = S[i][i];
}

// Singular Value Decomposition: A = U * Sigma * V^T
__device__ void svd_3x3(double A[DIM][DIM], double U[DIM][DIM], double S[DIM], double V[DIM][DIM]) {
    double AT[DIM][DIM], ATA[DIM][DIM];
    double eigenval[DIM];

    // Compute A^T * A
    mat_transpose(A, AT);
    mat_multiply(AT, A, ATA);

    // Eigendecomposition of A^T * A gives V and sigma^2
    jacobi(ATA, eigenval, V);

    // Sort singular values (descending)
    for (int i = 0; i < DIM; i++)
        for (int j = i + 1; j < DIM; j++)
            if (eigenval[i] < eigenval[j]) {
                double temp = eigenval[i];
                eigenval[i] = eigenval[j];
                eigenval[j] = temp;

                for (int k = 0; k < DIM; k++) {
                    temp = V[k][i];
                    V[k][i] = V[k][j];
                    V[k][j] = temp;
                }
            }

    // Singular values: sigma = sqrt(λ)
    for (int i = 0; i < DIM; i++)
        S[i] = (eigenval[i] > 0) ? sqrt(eigenval[i]) : 0.0;

    // Compute U: u_i = A * v_i / sigma_i
    for (int i = 0; i < DIM; i++) {
        if (S[i] > EPSILON_SVD) {
            for (int j = 0; j < DIM; j++) {
                U[j][i] = 0;
                for (int k = 0; k < DIM; k++)
                    U[j][i] += A[j][k] * V[k][i];
                U[j][i] /= S[i];
            }
        } else {
            for (int j = 0; j < DIM; j++)
                U[j][i] = (i == j) ? 1.0 : 0.0;
        }
    }
}

// Matrix inversion via SVD: A^(-1) = V * Sigma^(-1) * U^T
// Returns rank of matrix
// threshold: singular values < threshold are treated as zero (pseudo-inverse)
__device__ int invert_svd_schaefer(double *Atmp, double *A_tmpinv, double threshold)
{
    double U[DIM][DIM], V[DIM][DIM], S[DIM];
    double A[DIM][DIM], A_inv[DIM][DIM];


    // map Atmp to A
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++)
            A[i][j] = Atmp[i * DIM + j];

    svd_3x3(A, U, S, V);

    // Count rank and build Σ^(-1)
    int rank = 0;
    double S_inv[DIM][DIM] = {0};
    for (int i = 0; i < DIM; i++) {
        if (S[i] > threshold) {
            S_inv[i][i] = 1.0 / S[i];
            rank++;
        }
    }

    // A^(-1) = V * Σ^(-1) * U^T
    double UT[DIM][DIM], temp[DIM][DIM];
    mat_transpose(U, UT);
    mat_multiply(S_inv, UT, temp);
    mat_multiply(V, temp, A_inv);

    // map A_inv to A_tmpinv 
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++)
            A_tmpinv[i * DIM + j] = A_inv[i][j];

    return rank;
}

// Print functions for testing
__device__ void print_mat(const char* name, double M[DIM][DIM])
{
    printf("%s:\n", name);
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++)
            printf("%10.6f ", M[i][j]);
        printf("\n");
    }
    printf("\n");
}

/*
    double L_inv[DIM][DIM];
    double threshold = 1e-10;  // adjust based on condition number

    int rank = invert_svd(L, L_inv, threshold);
    printf("Matrix rank: %d\n\n", rank);

    print_mat("L^(-1)", L_inv);

    // Verify: L * L^(-1) = I
    double check[DIM][DIM];
    mat_multiply(L, L_inv, check);
    print_mat("L * L^(-1) (should be identity)", check);

*/