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

#ifndef _HEUN_RK4_H
#define _HEUN_RK4_H

#include "miluph.h"
#include "timeintegration.h"


void heun_rk4();

__global__ void setTimestep_heun(double *forcesPerBlock, double *courantPerBlock, double *dtSPerBlock, double *dtePerBlock, double *dtrhoPerBlock, double *dtdamagePerBlock, double *dtalphaPerBlock, double *dtartviscPerBlock, double *dtbetaPerBlock, double *dtalpha_epsporPerBlock, double *dtepsilon_vPerBlock);
__global__ void pressureChangeCheck_heun(double *maxpressureDiffPerBlock);

#endif
