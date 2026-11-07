/**
 * @author      Oliver Wandel and Christoph Schaefer
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


#ifndef _POROSITY_H
#define _POROSITY_H

#include "parameter.h"


__global__ void calculateDistensionChange();
#if PALPHA_POROSITY && PALPHA_RETURN_MAPPING
/// TRUE if the next call of rightHandSide() evaluates an accepted state and commits its distention as history
extern int palpha_commit_pending;
/**
 * @brief Return mapping of the p-alpha model: the distention of the current state from the crush curve.
 *
 * @details For the particles in p, alpha = alpha_cc(P_s(alpha*rho, e)/alpha) if the particle is compressed beyond
 * the crush curve, otherwise the distention after the last accepted step. S and damage_porjutzi are changed
 * accordingly. With commit, alpha is stored as history (p_rhs.alpha_jutzi_hist). numCompacted (may be NULL) counts
 * the particles that lie above the crush curve by more than PALPHA_ELASTIC_RELOADING_TOL.
 */
__global__ void palphaReturnMapping(int commit, int *numCompacted);
/// host wrapper for palphaReturnMapping(), warns once if the initial state lies above the crush curve
void palpha_return_mapping(int commit);
#endif

__global__ void calculateCompressiveStrength(void);
__global__ void calculateTensileStrength(void);

#endif
