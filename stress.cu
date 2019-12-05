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



#include "stress.h"
#include "parameter.h"
#include "miluph.h"
#include "timeintegration.h"
#include "linalg.h"


#if FRAGMENTATION
#warning DAMAGE_ACTS_ON_PRINCIPAL_STRESSES is defined in stress.cu
// if 1, then damage reduces the principal stresses
// if 0, then p<0 -> (1-d) p and S -> (1-d) S
# define DAMAGE_ACTS_ON_PRINCIPAL_STRESSES 0
#else
# define DAMAGE_ACTS_ON_PRINCIPAL_STRESSES 0
#endif


// principal axes damage does not work for pressure dependent yield strengths
#if DAMAGE_ACTS_ON_PRINCIPAL_STRESSES && COLLINS_PRESSURE_DEPENDENT_YIELD_STRENGTH
#error Do not combine DAMAGE_ACTS_ON_PRINCIPAL_STRESSES and COLLINS_PRESSURE_DEPENDENT_YIELD_STRENGTH
#endif

#if SOLID

__global__ void set_stress_tensor(void)
{
    register int i, inc, matId;
    int d, e;
    int niters;
    double sigma[DIM][DIM];
    double sigmatmp[DIM][DIM];
    double rotation_matrix[DIM][DIM];
    double main_stresses[DIM];
    double max_ev;
    double damage = 0.0;
    double stress_damage = 0.0;

    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];
        niters = 0;

        // here we set the stress tensor sigma from pressure and deviatoric stress S
        // note, that S was already lowered in plasticity
#if FRAGMENTATION
        damage = pow(p.damage_total[i], DIM);
        if (damage > 1.0) damage = 1.0;
        stress_damage = damage;

        // special handling of granular media with pressure dependent yield strengths
# if COLLINS_PRESSURE_DEPENDENT_YIELD_STRENGTH
        if (!(damage < 1)) {
            stress_damage = 0.0;
        }
# endif
#else
        damage = 0.0;
        stress_damage = 0.0;
#endif


#if DAMAGE_ACTS_ON_PRINCIPAL_STRESSES
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                sigmatmp[d][e] = 0.0;
                sigma[d][e] = p.S[stressIndex(i, d, e)];
                if (d == e) {
                    sigma[d][e] += -p.p[i];
                }
            }
        }
        // calculate main stresses
        niters = calculate_all_eigenvalues(sigma, main_stresses, rotation_matrix);
        for (d = 0; d < DIM; d++) {
            sigmatmp[d][d] = main_stresses[d];
            if (sigmatmp[d][d] > 0) {
                sigmatmp[d][d] *= (1.0 - damage);
            }
        }
        // rotate back the lowered principal stresses
        multiply_matrix(sigmatmp, rotation_matrix, sigma);
        transpose_matrix(rotation_matrix);
        multiply_matrix(rotation_matrix, sigma, sigmatmp);

        // sigmatmp now holds the stress tensor for particle i with damaged reduced stresses
        copy_matrix(sigmatmp, sigma);
#else
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                sigma[d][e] = (1.0 - stress_damage) * p.S[stressIndex(i, d, e)];
                if (d == e) { // the trace
                    if (p.p[i] < 0) {
                        sigma[d][e] += - (1.0 - damage) * p.p[i];
                    } else {
                        sigma[d][e] += -p.p[i];
                    }
                }
            }
        }

#endif

        // remember sigma
        for (d = 0; d < DIM; d++) {
            for (e = 0; e < DIM; e++) {
                p_rhs.sigma[stressIndex(i,d,e)] = sigma[d][e];
            }
        }

    }
}
#endif
