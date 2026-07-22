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

#ifndef _PLASTICITY_H
#define _PLASTICITY_H

#include "timeintegration.h"


#if PURE_REGOLITH
/**
 * @brief Scales the deviatoric stresses for material model `PURE_REGOLITH`.
 */
__global__ void plasticity(void);
#endif

#if PLASTICITY
/**
 * @brief Limits the deviatoric stresses for various material models.
 * @details Here the deviatoric stress tensor is reduced once the yield limit is exceeded, which is either
 * a simple constant for `VON_MISES_PLASTICITY`, or a more complicated function of pressure, etc. for
 * `MOHR_COULOMB_PLASTICITY`, `DRUCKER_PRAGER_PLASTICITY`, `COLLINS_PLASTICITY`, and `COLLINS_PLASTICITY_SIMPLE`.
 */
__global__ void plasticityModel(void);
#endif

#if JC_PLASTICITY
/**
 * @brief This is the Johnson-Cook plasticity model.
 */
__global__ void JohnsonCookPlasticity(void);
#endif


#endif
