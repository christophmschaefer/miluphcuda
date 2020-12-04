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

#include "timeintegration.h"
#include "parameter.h"
#include "miluph.h"

__global__ void get_extrema()
{

	register int i, inc, matId;
    	inc = blockDim.x * gridDim.x;
#if MORE_OUTPUT
    	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
		//looking for maximum pressure
		if (p.p[i] > p.p_max[i]) {
			p.p_max[i] = p.p[i];
		}
		//looking for minimal pressure
		if (p.p[i] < p.p_min[i]) {
			p.p_min[i] = p.p[i];
		}
		//looking for maximum density
		if (p.rho[i] > p.rho_max[i]) {
			p.rho_max[i] = p.rho[i];
		}
		//looking for minimal density
		if (p.rho[i] < p.rho_min[i]) {
			p.rho_min[i] = p.rho[i];
		}
		//looking for maximum energy
		if (p.e[i] > p.e_max[i]) {
			p.e_max[i] = p.e[i];
		}
		//looking for minimal energy
		if (p.e[i] < p.e_min[i]) {
			p.e_min[i] = p.e[i];
		}
		//looking for maximum soundspeed
		if (p.cs[i] > p.cs_max[i]) {
			p.cs_max[i] = p.cs[i];
		}
		//looking for minimal soundspeed
		if (p.cs[i] < p.cs_min[i]) {
			p.cs_min[i] = p.cs[i];
		}
	}
#endif
}


