/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com and Thomas I. Maindl
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

#ifndef _GRAVITY_H
#define _GRAVITY_H
#include "parameter.h"

__global__ void direct_selfgravity();
__global__ void selfgravity();
__global__ void addoldselfgravity();
__global__ void gravitation_from_point_masses(int calculate_nbody);
__global__ void particles_gravitational_feedback(int n, double *, double *, double *);
void backreaction_from_disk_to_point_masses(int calculate_nbody);
#endif
