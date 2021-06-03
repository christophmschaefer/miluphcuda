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


#ifndef _PRESSURE_H
#define _PRESSURE_H

#include "timeintegration.h"

/// implemented equation of states
enum EquationOfStates {
    EOS_TYPE_ACCRETED = -2, /// special flag for particles that got accreted by a gravitating point mass
    EOS_TYPE_IGNORE = -1, /// particle is ignored 
    EOS_TYPE_POLYTROPIC_GAS = 0, /// polytropic EOS for gas, needs polytropic_K and polytropic_gamma in material.cfg file
    EOS_TYPE_MURNAGHAN = 1, /// Murnaghan EOS for solid bodies, see Melosh "Impact Cratering", needs in material.cfg: rho_0, bulk_modulus, n
    EOS_TYPE_TILLOTSON = 2, /// Tillotson EOS for solid bodies, see Melosh "Impact Cratering", needs in material.cfg: till_rho_0, till_A, till_B, till_E_0, till_E_iv, till_E_cv, till_a, till_b, till_alpha, till_beta; bulk_modulus and shear_modulus are needed to calculate the sound speed and crack growth speed for FRAGMENTATION
    EOS_TYPE_ISOTHERMAL_GAS = 3, /// this is pure molecular hydrogen at 10 K
    EOS_TYPE_REGOLITH = 4, /// The Bui et al. 2008 soil model
    EOS_TYPE_JUTZI = 5, /// Tillotson EOS with p-alpha model by Jutzi et al.
    EOS_TYPE_JUTZI_MURNAGHAN = 6, /// Murnaghan EOS with p-alpha model by Jutzi et al.
    EOS_TYPE_ANEOS = 7, /// ANEOS (or tabulated EOS in ANEOS format)
    EOS_TYPE_VISCOUS_REGOLITH = 8, /// describe regolith as a viscous material -> EXPERIMENTAL DO NOT USE
    EOS_TYPE_IDEAL_GAS = 9, /// ideal gas equation, set polytropic_gamma in material.cfg
    EOS_TYPE_SIRONO = 10, /// Sirono EOS modifed by Geretshauser in 2009/10
    EOS_TYPE_EPSILON = 11, /// Tillotson EOS with epsilon-alpha model by Wuennemann, Collins et al.
    EOS_TYPE_LOCALLY_ISOTHERMAL_GAS = 12, /// locally isothermal gas: \f$ p = c_s^2 \times \varrho \f$
    EOS_TYPE_JUTZI_ANEOS = 13/// ANEOS EOS with p-alpha model by Jutzi et al.
};

/**
 * @brief Calculates the pressure for all particles
 * @details Depending on the EOS_TYPE which is set in material.cfg for the material, the pressure is calculated for all particles.
 * 
 * @return __global__ 
 */
__global__ void calculatePressure(void);

#endif
