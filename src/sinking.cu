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
 * You should have received a copy of the GNU General Public License;
 * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "timeintegration.h"
#include "parameter.h"
#include "miluph.h"
#include "pressure.h"
#include "config_parameter.h"


#if PARTICLE_ACCRETION
#if UPDATE_SINK_VALUES
//function adds to sink: mass and velocity (angular momentum, linear momentum) of the accreted particle and calculate the new position of sink (COM)
__device__ void UpdateSinkValues(int sink_num, int particle_id) 
{
    //COM - position - velocity
    pointmass.x[sink_num] = (pointmass.m[sink_num]*pointmass.x[sink_num] + p.m[particle_id]*p.x[particle_id]) / (pointmass.m[sink_num] + p.m[particle_id]);
    pointmass.y[sink_num] = (pointmass.m[sink_num]*pointmass.y[sink_num] + p.m[particle_id]*p.y[particle_id]) / (pointmass.m[sink_num] + p.m[particle_id]);

    pointmass.vx[sink_num] = (p.m[particle_id]*p.vx[particle_id] + pointmass.m[sink_num]*pointmass.vx[sink_num]) / (p.m[particle_id] + pointmass.m[sink_num]);
    pointmass.vy[sink_num] = (p.m[particle_id]*p.vy[particle_id] + pointmass.m[sink_num]*pointmass.vy[sink_num]) / (p.m[particle_id] + pointmass.m[sink_num]);

#if DIM == 3
    pointmass.z[sink_num] = (pointmass.m[sink_num]*pointmass.z[sink_num] + p.m[particle_id]*p.z[particle_id]) / (pointmass.m[sink_num] + p.m[particle_id]);
    
    pointmass.vz[sink_num] = (p.m[particle_id]*p.vz[particle_id] + pointmass.m[sink_num]*pointmass.vz[sink_num]) / (p.m[particle_id] + pointmass.m[sink_num]);

#endif //DIM == 3

    pointmass.m[sink_num] += p.m[particle_id];
}
#endif //UPDATE_SINK_VALUES



//function checks if particle is to be accreted on sink particle
__global__ void ParticleSinking()
{
	register int i, inc, n;
	double vel_esc, vel, dist_0, dist_1, distance, h, h_circ, r_x, r_y, r_z, v_x, v_y, v_z, h_x, h_y, h_z;
	inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {

        //look for a particle with material type = -2 and reset it to material type = -1 if it is bound to sink particle
        if (p_rhs.materialId[i] == EOS_TYPE_ACCRETED) {

            //distance between particle and each sink and particle velocity
#if DIM == 2
            dist_0 = sqrt( (pointmass.x[0] - p.x[i])*(pointmass.x[0] - p.x[i]) + (pointmass.y[0] - p.y[i])*(pointmass.y[0] - p.y[i]) );
	        dist_1 = sqrt( (pointmass.x[1] - p.x[i])*(pointmass.x[1] - p.x[i]) + (pointmass.y[1] - p.y[i])*(pointmass.y[1] - p.y[i]) );

            vel = sqrt(p.vx[i]*p.vx[i] + p.vy[i]*p.vy[i]);
#endif //end DIM == 2

#if DIM == 3
            dist_0 = sqrt( (pointmass.x[0] - p.x[i])*(pointmass.x[0] - p.x[i]) + (pointmass.y[0] - p.y[i])*(pointmass.y[0] - p.y[i]) + (pointmass.z[0] - p.z[i])*(pointmass.z[0] - p.z[i]) );
	        dist_1 = sqrt( (pointmass.x[1] - p.x[i])*(pointmass.x[1] - p.x[i]) + (pointmass.y[1] - p.y[i])*(pointmass.y[1] - p.y[i]) + (pointmass.z[1] - p.z[i])*(pointmass.z[1] - p.z[i]) );

            vel = sqrt(p.vx[i]*p.vx[i] + p.vy[i]*p.vy[i] + p.vz[i]*p.vz[i]);
#endif //end DIM == 3

	    	if (dist_0 < dist_1) {
		        n = 0;
                distance = dist_0;
			}
		    else {
			    n = 1;
                distance = dist_1;
		    }

    	    //escape velocity at r_acc(rmin)
	        vel_esc = sqrt(2. * gravConst * pointmass.m[n] / distance);
                  
            //specific angular momentum of each particle about the sink particle
	    	r_x = pointmass.x[n] - p.x[i];
            r_y = pointmass.y[n] - p.y[i];

            v_x = pointmass.vx[n] - p.vx[i];
            v_y = pointmass.vy[n] - p.vy[i];

            h_z = r_x*v_y - r_y*v_x;

#if DIM == 2
            h = sqrt(h_z*h_z);
#endif //DIM ==2

#if DIM == 3
            r_z = pointmass.z[n] - p.z[i];

            v_z = pointmass.vz[n] - p.vz[i];

            h_x = r_y*v_z - r_z*v_y;
            h_y = r_z*v_x - r_x*v_z;
            h = sqrt(h_x*h_x + h_y*h_y + h_z*h_z);
#endif //DIM == 3

            //specific angular momentum to form circular orbit at semi-major axis
            h_circ = sqrt(gravConst * pointmass.m[n] * distance);

		    //check if particle is to be accreted (bound) to sink: particle velocity < escape velocity && specific angular momentum of particle < angular momentum to form circular orbit
		    if (vel < vel_esc && h < h_circ) {
#if UPDATE_SINK_VALUES
			    UpdateSinkValues(n, i);
#endif //UPDATE_SINK_VALUES
			    p_rhs.materialId[i] = EOS_TYPE_IGNORE;
			} 
            else {
            //particle is not accreted, particle material set to inital one
                p_rhs.materialId[i] = p_rhs.materialId0[i];                       
            }
		}
	}
}
#endif //PARTICLE_ACCRETION
