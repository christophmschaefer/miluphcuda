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


#if VON_MISES_PLASTICITY || MOHR_COULOMB_PLASTICITY || DRUCKER_PRAGER_PLASTICITY || COLLINS_PLASTICITY || COLLINS_PLASTICITY_SIMPLE
# define PLASTICITY 1
#endif


// basic checks
#if (!SOLID && !HYDRO) || (SOLID && HYDRO)
# error Choose either SOLID or HYDRO in parameter.h.
#endif


// checks for plasticity models
#if VON_MISES_PLASTICITY && COLLINS_PLASTICITY
# error You cannot choose VON_MISES_PLASTICITY and COLLINS_PLASTICITY at the same time.
#endif

#if VON_MISES_PLASTICITY && COLLINS_PLASTICITY_SIMPLE
# error You cannot choose VON_MISES_PLASTICITY and COLLINS_PLASTICITY_SIMPLE at the same time.
#endif

#if MOHR_COULOMB_PLASTICITY && DRUCKER_PRAGER_PLASTICITY
# error You cannot choose MOHR_COULOMB_PLASTICITY and DRUCKER_PRAGER_PLASTICITY at the same time.
#endif

#if MOHR_COULOMB_PLASTICITY && COLLINS_PLASTICITY
# error You cannot choose MOHR_COULOMB_PLASTICITY and COLLINS_PLASTICITY at the same time.
#endif

#if DRUCKER_PRAGER_PLASTICITY && COLLINS_PLASTICITY
# error You cannot choose DRUCKER_PRAGER_PLASTICITY and COLLINS_PLASTICITY at the same time.
#endif

#if COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY && !COLLINS_PLASTICITY
# error You have chosen COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY but not also COLLINS_PLASTICITY in parameter.h. That is not what you want.
#endif

#if COLLINS_PLASTICITY && COLLINS_PLASTICITY_SIMPLE
# error You have chosen COLLINS_PLASTICITY and also COLLINS_PLASTICITY_SIMPLE in parameter.h. Choose either one, not both.
#endif

#if COLLINS_PLASTICITY_SIMPLE && COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY
# error You have chosen COLLINS_PLASTICITY_SIMPLE and also COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY in parameter.h. This combination is not implemented yet...
#endif

#if COLLINS_PLASTICITY && DAMAGE_ACTS_ON_S
# error You chose COLLINS_PLASTICITY and also DAMAGE_ACTS_ON_S in parameter.h. Not a good idea.
#endif

#if VISCOUS_REGOLITH && !SOLID
# error turn SOLID on when using VISCOUS_REGOLITH
#endif

#if KLEY_VISCOSITY && !NAVIER_STOKES
# error turn on NAVIER_STOKES when using KLEY_VISCOSITY
#endif

#if PURE_REGOLITH && !SOLID
# error turn SOLID on when using PURE_REGOLITH
#endif

#if PLASTICITY && !SOLID
# error Using a PLASTICITY model is only possible in combination with SOLID...
#endif

#if JC_PLASTICITY && !SOLID
# error turn SOLID on when using JC_PLASTICITY
#endif

#if PLASTICITY && JC_PLASTICITY
# error Cannot use another PLASTICITY model along with JC_PLASTICITY at the same time. Decide for one and recompile.
#endif

#if LOW_DENSITY_WEAKENING && !(VON_MISES_PLASTICITY || MOHR_COULOMB_PLASTICITY || DRUCKER_PRAGER_PLASTICITY || COLLINS_PLASTICITY || COLLINS_PLASTICITY_SIMPLE)
# error LOW_DENSITY_WEAKENING only works in combination with either VON_MISES_PLASTICITY, MOHR_COULOMB_PLASTICITY, DRUCKER_PRAGER_PLASTICITY, COLLINS_PLASTICITY, or COLLINS_PLASTICITY_SIMPLE.
#endif

#if COLLINS_PLASTICITY && LOW_DENSITY_WEAKENING && !FRAGMENTATION
# error You want to use COLLINS_PLASTICITY together with LOW_DENSITY_WEAKENING, but without FRAGMENTATION. Makes no sense since LOW_DENSITY_WEAKENING only affects the damaged yield curve (its cohesion).
#endif


// checks for fragmentation model
#if FRAGMENTATION && !SOLID
# error turn SOLID on when using FRAGMENTATION
#endif

#if DAMAGE_ACTS_ON_S && !FRAGMENTATION
# error You set DAMAGE_ACTS_ON_S but not FRAGMENTATION in parameter.h. Not working...
#endif

#if MOHR_COULOMB_PLASTICITY && FRAGMENTATION
# error MOHR_COULOMB_PLASTICITY is intended for granular-like materials and does not make sense together with FRAGMENTATION, does it? (Its negative-pressure cap may inhibit flaw activation for example, and additional stress reduction by damage would be ambiguous as well...)
#endif

#if COLLINS_PLASTICITY_SIMPLE && FRAGMENTATION
# error COLLINS_PLASTICITY_SIMPLE does not make sense together with FRAGMENTATION, does it? (Its negative-pressure cap may inhibit flaw activation for example, and additional stress reduction by damage would be ambiguous as well...) Use the regular COLLINS_PLASTICITY instead?!
#endif


// checks for variable sml
#if VARIABLE_SML
# if FIXED_NOI && INTEGRATE_SML
#  error use VARIABLE_SML only with FIXED_NOI or INTEGRATE_SML, not both
# endif
# if !(FIXED_NOI || INTEGRATE_SML)
#  error choose either one of FIXED_NOI or INTEGRATE_SML in combination with VARIABLE_SML
# endif
#endif


// misc checks
#if SYMMETRIC_STRESSTENSOR && !SOLID
# error turn SOLID on when using SYMMETRIC_STRESSTENSOR
#endif

#if SHAKURA_SUNYAEV_ALPHA && CONSTANT_KINEMATIC_VISCOSITY
# error choose only one viscosity model
#endif

#if NAVIER_STOKES
# if !SHAKURA_SUNYAEV_ALPHA && !CONSTANT_KINEMATIC_VISCOSITY
#  error set either SHAKURA_SUNYAEV_ALPHA or CONSTANT_KINEMATIC_VISCOSITY
# endif
#endif

#if DIM == 1 && PARTICLE_ACCRETION
# error Particle accretion only if DIM > 1.
#endif

#if USE_BSPLINE_KERNEL && USE_WENDLAND_KERNEL
# error specifiy only one kernel
#endif

#if ARTIFICIAL_STRESS && !SOLID
# error turn off ARTIFICIAL_STRESS when running hydro sims
#endif

#if STRESS_PALPHA_POROSITY && !PALPHA_POROSITY
# error You set STRESS_PALPHA_POROSITY but not PALPHA_POROSITY in parameter.h.
#endif

#if (MORE_OUTPUT || MORE_ANEOS_OUTPUT) && !HDF5IO
# error You need to set HDF5IO if you want MORE_OUTPUT or MORE_ANEOS_OUTPUT in parameter.h.
#endif


#endif
