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

#include "gravity.h"
#include "tree.h"
#include "timeintegration.h"
#include "parameter.h"
#include "miluph.h"
#include "pressure.h"

extern __device__ volatile double radius;


// add acceleration due to gravity to particle acceleration
__global__ void addoldselfgravity() 
{
	int i;

	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += blockDim.x * gridDim.x) {
        p.ax[i] += p.g_ax[i];
#if DIM > 1
        p.ay[i] += p.g_ay[i];
#if DIM > 2
        p.az[i] += p.g_az[i];
#endif
#endif
    }
}


// adds the acceleration due to the point masses 
__global__ void gravitation_from_point_masses()
{
    int i, inc;
    int j;
    int d;
    double r;
    double rrr;
    double dr[DIM];
    inc = blockDim.x * gridDim.x;
    // loop for point masses
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i += inc) {
        for (j = 0; j < numPointmasses; j++) {
            if (i == j) continue;
            r = 0.0;
            dr[0] = pointmass.x[j] - pointmass.x[i];
#if DIM > 1
            dr[1] = pointmass.y[j] - pointmass.y[i];
#if DIM > 2
            dr[2] = pointmass.z[j] - pointmass.z[i];
#endif
#endif
            for (d = 0; d < DIM; d++) {
                r += dr[d]*dr[d];
            }
            r = sqrt(r);
            rrr = r*r*r;
            pointmass.ax[i] += C_GRAVITY_SI * pointmass.m[j] * dr[0]/(rrr);
#if DIM > 1
            pointmass.ay[i] += C_GRAVITY_SI * pointmass.m[j] * dr[1]/(rrr);
#if DIM > 2
            pointmass.az[i] += C_GRAVITY_SI * pointmass.m[j] * dr[2]/(rrr);
#endif
#endif
        }
    }


    // loop over all particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {
        if (p_rhs.materialId[i] == EOS_TYPE_IGNORE || matEOS[p_rhs.materialId[i]] == EOS_TYPE_IGNORE) {
            continue;
        }

        for (j = 0; j < numPointmasses; j++) {
            r = 0.0;
            dr[0] = pointmass.x[j] - p.x[i];
#if DIM > 1
            dr[1] = pointmass.y[j] - p.y[i];
#if DIM > 2
            dr[2] = pointmass.z[j] - p.z[i];
#endif
#endif
            for (d = 0; d < DIM; d++) {
                r += dr[d]*dr[d];
            }
            r = sqrt(r);
            rrr = r*r*r;
            if (r < pointmass.rmax[j] && r > pointmass.rmin[j]) {
                p.ax[i] += C_GRAVITY_SI * pointmass.m[j] * dr[0]/(rrr);
#if DIM > 1
                p.ay[i] += C_GRAVITY_SI * pointmass.m[j] * dr[1]/(rrr);
#if DIM > 2
                p.az[i] += C_GRAVITY_SI * pointmass.m[j] * dr[2]/(rrr);
#endif
#endif
            } else {
                p_rhs.materialId[i] = EOS_TYPE_IGNORE;
            }
        }
    }
}

// compute self gravity using N**2 algorithm
__global__ void direct_selfgravity() 
{
    int i, inc;
    int j;
    int d;
    double a_grav[DIM];
    double dist;
    double f;
    double dx[DIM];
    double sml;

    inc = blockDim.x * gridDim.x;
    // loop over all particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {
        for (d = 0; d < DIM; d++) {
            a_grav[d] = 0.0;
        }
        sml = p.h[i];
        if (p_rhs.materialId[i] == EOS_TYPE_IGNORE || matEOS[p_rhs.materialId[i]] == EOS_TYPE_IGNORE) {
            continue;
        }

        // loop over all other particles
        for (j = 0; j < numRealParticles; j++) {
            if (i == j)
                continue;

            dist = 0.0;
            dx[0] = p.x[i] - p.x[j];
#if DIM > 1
            dx[1] = p.y[i] - p.y[j];
#if DIM > 2
            dx[2] = p.z[i] - p.z[j];
#endif
#endif
            for (d = 0; d < DIM; d++) {
                dist += dx[d]*dx[d];
            }
            dist = sqrt(dist);
		    f = C_GRAVITY_SI * p.m[j]; // / (distance*distance*distance);
		    f /= dist > sml ? (dist*dist*dist) : (sml*sml*sml);
            for (d = 0; d < DIM; d++) {
                a_grav[d] -= f*dx[d];
            }
        }

		p.ax[i] += a_grav[0];
		p.g_ax[i] = a_grav[0];
#if DIM > 1
		p.ay[i] += a_grav[1];
		p.g_ay[i] = a_grav[1];
#if DIM == 3
		p.az[i] += a_grav[2];
		p.g_az[i] = a_grav[2];
#endif
#endif

    }
}

// compute self gravity using the tree
__global__ void selfgravity() 
{
	int i, child, nodeIndex, childNumber, depth;
	double px, ax, dx, f, distance;
#if DIM > 1
    double py, ay, dy;
#endif
	int currentNodeIndex[MAXDEPTH];
	int currentChildNumber[MAXDEPTH];
#if DIM == 3
	double pz, az, dz;
#endif
	double sml;
    double thetasq = theta*theta;

	__shared__ volatile double cellsize[MAXDEPTH];
	if (0 == threadIdx.x) {
		cellsize[0] = 4.0 * radius * radius;
		for (i = 1; i < MAXDEPTH; i++) {
			cellsize[i] = cellsize[i - 1] * 0.25;
		}
	}

	__syncthreads();

	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += blockDim.x * gridDim.x) {
		px = p.x[i];
#if DIM > 1
		py = p.y[i];
#if DIM == 3
		pz = p.z[i];
#endif
#endif
        p.g_ax[i] = 0.0;
#if DIM > 1
        p.g_ay[i] = 0.0;
#endif
        sml = p.h[i];
		ax = 0.0;
#if DIM > 1
		ay = 0.0;
#if DIM == 3
		az = 0.0;
        p.g_az[i] = 0.0;
#endif
#endif

		// start at root
		depth = 1;
		currentNodeIndex[depth] = numNodes - 1;
		currentChildNumber[depth] = 0;

		do {
			childNumber = currentChildNumber[depth];
			nodeIndex = currentNodeIndex[depth];

			while(childNumber < numChildren) {
				do {
					child = childList[childListIndex(nodeIndex, childNumber)];
					childNumber++;
				} while(child == EMPTY && childNumber < numChildren);
				if (child != EMPTY && child != i) { // dont do selfgravity with yourself!
					dx = p.x[child] - px;
					distance = dx*dx;
#if DIM > 1
					dy = p.y[child] - py;
					distance += dy*dy;
#endif
#if DIM == 3
					dz = p.z[child] - pz;
					distance += dz*dz;
#endif
					// if child is leaf or far away
					//if (child < numParticles || distance * theta > cellsize[depth]) {
					if (child < numParticles || distance * thetasq > cellsize[depth]) {
						distance = sqrt(distance);
                        //distance += 1e10;
						f = C_GRAVITY_SI * p.m[child]; // / (distance*distance*distance);
						f /= distance > sml ? (distance*distance*distance) : (sml*sml*sml);
           //             f = 0.0;
						ax += f*dx;
#if DIM > 1
						ay += f*dy;
#if DIM == 3
						az += f*dz;
#endif
#endif
					} else {
						// put child on stack
						currentChildNumber[depth] = childNumber;
						currentNodeIndex[depth] = nodeIndex;
						depth++;
                        if (depth == MAXDEPTH) {
                            printf("\n\nMAXDEPTH reached in selfgravity... this is not good.\n\n");
                            assert(depth < MAXDEPTH);
                        }
						childNumber = 0;
						nodeIndex = child;
					}
				}
			}
			depth--;
		} while(depth > 0);



		p.ax[i] += ax;
		p.g_ax[i] = ax;
#if DIM > 1
		p.ay[i] += ay;
		p.g_ay[i] = ay;
#if DIM == 3
		p.az[i] += az;
		p.g_az[i] = az;
#endif
#endif
	}
}






