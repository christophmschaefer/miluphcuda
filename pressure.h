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

enum EquationOfStates {
    EOS_TYPE_ACCRETED = -2,
    EOS_TYPE_IGNORE = -1,
// polytropic EOS for gas, needs polytropic_K and polytropic_gamma in material.cfg file
    EOS_TYPE_POLYTROPIC_GAS = 0,
// Murnaghan EOS for solid bodies, see Melosh "Impact Cratering" for reference
// needs bulk_modulus, rho_0 and n in material.cfg
    EOS_TYPE_MURNAGHAN = 1,
// Tillotson EOS for solid bodies, see Melosh "Impact Cratering" for reference
// needs alot of parameters in material.cfg:
// yield_stress till_rho_0 till_A till_B till_E_0 till_E_iv till_E_cv till_a till_b till_alpha till_beta
// bulk modulus and shear modulus are needed to calculate the sound speed and the crack growth speed
    EOS_TYPE_TILLOTSON = 2,
// this is pure molecular hydrogen at 10 K
    EOS_TYPE_ISOTHERMAL_GAS = 3,
// The Bui et al. 2008 soil model
    EOS_TYPE_REGOLITH = 4,
// Tillotson EOS with p-alpha model by Jutzi et al.
    EOS_TYPE_JUTZI = 5,
// Murnaghan EOS with p-alpha model by Jutzi et al.
    EOS_TYPE_JUTZI_MURNAGHAN = 6,
// ANEOS
    EOS_TYPE_ANEOS = 7,
// describe regolith as a viscous material -> EXPERIMENTAL DO NOT USE
    EOS_TYPE_VISCOUS_REGOLITH = 8,
// ideal gas equation, set polytropic_gamma in material.cfg
    EOS_TYPE_IDEAL_GAS = 9,
// Sirono EOS modifed by Geretshauser in 2009/10
    EOS_TYPE_SIRONO = 10,
// Tillotson EOS with espilon-alpha model by Wuennemann Collins ..
    EOS_TYPE_EPSILON = 11,
// locally isothermal gas: p = c_s**2 \times rho
    EOS_TYPE_LOCALLY_ISOTHERMAL_GAS = 12,
// ANEOS EOS with p-alpha model by Jutzi et al.
	EOS_TYPE_JUTZI_ANEOS = 13
};


__global__ void calculatePressure(void);

#endif
