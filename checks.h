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
#ifndef _CHECKS_H
#define _CHECKS_H
#if MOHR_COULOMB_PLASTICITY && DRUCKER_PRAGER_PLASTICITY
#error choose only one of the three available plastic flow rules
#endif
#if MOHR_COULOMB_PLASTICITY && COLLINS_PRESSURE_DEPENDENT_YIELD_STRENGTH
#error choose only one of the three available plastic flow rules
#endif
#if DRUCKER_PRAGER_PLASTICITY && COLLINS_PRESSURE_DEPENDENT_YIELD_STRENGTH
#error choose only one of the three available plastic flow rules
#endif

#if USE_BSPLINE_KERNEL && USE_WENDLAND_KERNEL
#error specifiy only one kernel
#endif

#if VISCOUS_REGOLITH && !SOLID
#error turn SOLID on when using VISCOUS_REGOLITH
#endif

#if VON_MISES_PLASTICITY && !SOLID
#error turn SOLID on when using VON_MISES_PLASTICITY
#endif

#if JC_PLASTICITY && !SOLID
#error turn SOLID on when using JC_PLASTICITY
#endif

#if FRAGMENTATION && !SOLID
#error turn SOLID on when using FRAGMENTATION
#endif


#if SYMMETRIC_STRESSTENSOR && !SOLID
#error turn SOLID on when using SYMMETRIC_STRESSTENSOR
#endif

#if COHESION_FOR_DAMAGED_MATERIAL && !FRAGMENTATION
# error turn on FRAGMENTATION when using COHESION_FOR_DAMAGED_MATERIAL
#endif


#if ARTIFICIAL_STRESS && !SOLID
# error turn off ARTIFICIAL_STRESS when running pure hydro
#endif

#if VARIABLE_SML
# if FIXED_NOI && INTEGRATE_SML
#  error use VARIABLE_SML only with FIXED_NOI or INTEGRATE_SML
# endif
#endif


#endif
