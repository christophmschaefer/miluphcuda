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


#if MOHR_COLOUMB_PLASTICITY
# define PLASTICITY
#endif
#if DRUCKER_PRAGER_PLASTICITY
# define PLASTICITY
#endif
#if COLLINS_PLASTICITY
# define PLASTICITY
#endif
#if VON_MISES_PLASTICITY
# define PLASTICITY
#endif
#if COLLINS_PLASTICITY_SIMPLE
# define PLASTICITY
#endif


#if MOHR_COULOMB_PLASTICITY && DRUCKER_PRAGER_PLASTICITY
#error ERROR. Choose only one of the three available plastic flow rules in parameter.h.
#endif
#if MOHR_COULOMB_PLASTICITY && COLLINS_PLASTICITY
#error ERROR. Choose only one of the three available plastic flow rules in parameter.h.
#endif
#if DRUCKER_PRAGER_PLASTICITY && COLLINS_PLASTICITY
#error ERROR. Choose only one of the three available plastic flow rules in parameter.h.
#endif
#if COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY && !COLLINS_PLASTICITY
#error ERROR. You have chosen COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY but not also COLLINS_PLASTICITY in parameter.h. That is not what you want.
#endif
#if COLLINS_PLASTICITY && COLLINS_PLASTICITY_SIMPLE
#error ERROR. You have chosen COLLINS_PLASTICITY and also COLLINS_PLASTICITY_SIMPLE in parameter.h. Choose either one, not both.
#endif
#if COLLINS_PLASTICITY_SIMPLE && COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY
#error ERROR. You have chosen COLLINS_PLASTICITY_SIMPLE and also COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY in parameter.h. This combination is not implemented yet...
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

#if VON_MISES_PLASTICITY && JC_PLASTICITY
#error Error: Cannot use both Von Mises and Johnson-Cook Plasticity Models at the same time. Decide for one and recompile.
#endif


#if SYMMETRIC_STRESSTENSOR && !SOLID
#error turn SOLID on when using SYMMETRIC_STRESSTENSOR
#endif

#if COHESION_FOR_DAMAGED_MATERIAL && !FRAGMENTATION
# error turn on FRAGMENTATION when using COHESION_FOR_DAMAGED_MATERIAL
#endif

#if SHAKURA_SUNYAEV_ALPHA && CONSTANT_KINEMATIC_VISCOSITY 
# error choose only one viscosity model
#endif

#if NAVIER_STOKES
# if !SHAKURA_SUNYAEV_ALPHA && !CONSTANT_KINEMATIC_VISCOSITY
#error set either SHAKURA_SUNYAEV_ALPHA or CONSTANT_KINEMATIC_VISCOSITY
#endif
#endif

#if DIM == 1 && PARTICLE_ACCRETION
#error Particle accretion only if DIM > 1
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