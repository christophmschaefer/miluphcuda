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

#include "artificial_stress.h"
#include "parameter.h"
#include "miluph.h"
#include "linalg.h"
#include "config_parameter.h"

// transform to principal axes before applying artificial stress
#define PRINCIPAL_AXES_ARTIFICIAL_STRESS 1

#if ARTIFICIAL_STRESS
__global__ void compute_artificial_stress(int *interactions) 
{
    int i, inc;
    int d, e;
    int niters = 0;
    int matId;
    double max_ev = -1e300;

    // the diagonalized tensors
    double main_stresses[DIM];
    double rotation_matrix[DIM][DIM];
    double R[DIM][DIM];
    double sigma[DIM][DIM];
    double Rtmp[DIM][DIM];


    inc = blockDim.x * gridDim.x;

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];
        // build stress tensor from deviatoric stress and pressure
        for (d = 0; d < DIM; d++) { 
            for (e = 0; e < DIM; e++) {
                sigma[d][e] = p_rhs.sigma[stressIndex(i,d,e)];
            }
        }
#if PRINCIPAL_AXES_ARTIFICIAL_STRESS
        niters = calculate_all_eigenvalues(sigma, main_stresses, rotation_matrix);
       // determine the maximum stress
        max_ev = main_stresses[0];
	    for (e = 1; e < DIM; e++) {
		    if (main_stresses[e] > max_ev) {
			    max_ev = main_stresses[e];
            }
		}
        // now calculate the artificial stress from the main stresses
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                R[d][e] = 0.0;
            }
            if (main_stresses[d] > 0) {
                R[d][d] = -matepsilon_stress[matId]*main_stresses[d];
            }
        }    
        // convert back in the original coordinate system with the rotation matrix
        multiply_matrix(R, rotation_matrix, Rtmp);
        transpose_matrix(rotation_matrix);
        multiply_matrix(rotation_matrix, Rtmp, R);
        // now save R for the particle
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                p_rhs.R[stressIndex(i,d,e)]= R[d][e];
            }
        }
#else
        // no transformation, just reduce the standard stress
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                if (sigma[d][e] > 0) {
                    p_rhs.R[stressIndex(i,d,e)] = -matepsilon_stress[matId] * sigma[d][e];
                }
            }
        }
#endif // PRINCIPAL_AXES_ARTIFICIAL_STRESS

    }
}
#endif









