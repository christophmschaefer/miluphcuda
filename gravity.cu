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
#include "config_parameter.h"
#include "timeintegration.h"
#include "parameter.h"
#include "miluph.h"
#include "pressure.h"

extern __device__ volatile double radius;
extern __device__ int blockCount;

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



#if GRAVITATING_POINT_MASSES
// launches the kernel to calculate the gravitational force from the particles on the pointmasses if required
void backreaction_from_disk_to_point_masses(int calculate_nbody)
{
    int n = 0;
    // value for the time being
    const int h_blocksize = 256;
    double *g_x, *g_y, *g_z;

    // check flag if we use the coupled heun_rk4 integrator
    // if so, return
    if (!calculate_nbody) {
        return;
    }

    cudaMalloc((void **) &g_x, h_blocksize*sizeof(double));
    cudaMalloc((void **) &g_y, h_blocksize*sizeof(double));
    cudaMalloc((void **) &g_z, h_blocksize*sizeof(double));


    for (n = 0; n < numberOfPointmasses; n++) {

        if (!pointmass_host.feels_particles[n]) {
            continue;
        }
#if DEBUG_GRAVITY
        fprintf(stdout, "Calculating force from particles on star/planet no. %d\n", n);
#endif
        cudaVerifyKernel((particles_gravitational_feedback<<<h_blocksize, NUM_THREADS_REDUCTION>>>(n, g_x, g_y, g_z)));
        cudaVerify(cudaDeviceSynchronize());

    }


    cudaFree(g_x);
    cudaFree(g_y);
    cudaFree(g_z);
}


__device__ void get_acceleration_by_particle(int n, double *ax, double *ay, double *az, int i)
{
    // do some more magic here
    int d;
    double r = 0.0;
    double rrr;
    double smlcubed;
    double dr[DIM];

    dr[0] = pointmass.x[n] - p.x[i];
#if DIM > 1
    dr[1] = pointmass.y[n] - p.y[i];
#if DIM > 2
    dr[2] = pointmass.z[n] - p.z[i];
#endif
#endif
    for (d = 0; d < DIM; d++) {
        r += dr[d]*dr[d];
    }
    r = sqrt(r);
    rrr = r*r*r;
    smlcubed = p.h[i]*p.h[i]*p.h[i];
    if (rrr < smlcubed) {
        rrr = smlcubed;
    }
    if (r < pointmass.rmax[n]) {
        *ax = -gravConst * p.m[i] * dr[0]/(rrr);
#if DIM > 1
        *ay = -gravConst * p.m[i] * dr[1]/(rrr);
#if DIM > 2
        *az = -gravConst * p.m[i] * dr[2]/(rrr);
#endif
#endif
    } else {
        *ax = 0.0;
        *ay = 0.0;
        *az = 0.0;
    }
}



__global__ void particles_gravitational_feedback(int n, double *g_ax, double *g_ay, double *g_az)
{
    int idx, inc;
    int i;
    int tid;
    int j, k, m;
    volatile double local_ax, local_ay, local_az;
    double ax, ay, az;
    __shared__ double sh_ax[NUM_THREADS_REDUCTION];
    __shared__ double sh_ay[NUM_THREADS_REDUCTION];
    __shared__ double sh_az[NUM_THREADS_REDUCTION];

    tid = threadIdx.x;
    idx = threadIdx.x + blockIdx.x * blockDim.x;
    inc = blockDim.x * gridDim.x;
    local_ax = local_ay = local_az = 0.0;
    for (i = idx; i < numParticles; i += inc) {
        if (p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
            continue;
        }
        ax = ay = az = 0.0;
        get_acceleration_by_particle(n, &ax, &ay, &az, i);
        local_ax += ax;
#if DIM > 1
        local_ay += ay;
#if DIM > 2
        local_az += az;
#endif
#endif
    }
    sh_ax[tid] = local_ax;
#if DIM > 1
    sh_ay[tid] = local_ay;
#if DIM > 2
    sh_az[tid] = local_az;
#endif
#endif
    for (j = NUM_THREADS_REDUCTION/2; j > 0; j>>=1) {
        __syncthreads();
        if (tid < j) {
            sh_ax[tid] = local_ax = sh_ax[tid+j] + local_ax;
#if DIM > 1
            sh_ay[tid] = local_ay = sh_ay[tid+j] + local_ay;
#if DIM > 2
            sh_az[tid] = local_az = sh_az[tid+j] + local_az;
#endif
#endif
        }
    }
    // write block result to global memory
    ax = 0;
    ay = 0;
    az = 0;
    // only first thread in each block
    if (tid == 0) {
        k = blockIdx.x;
        g_ax[k] = local_ax;
#if DIM > 1
        g_ay[k] = local_ay;
#if DIM > 2
        g_az[k] = local_az;
#endif
#endif
        m = gridDim.x-1;
        if (m == atomicInc((unsigned int *) &blockCount, m)) {
            // last block, combine all results
            for (j = 0; j <= m; j++) {
                ax += g_ax[j];
#if DIM > 1
                ay += g_ay[j];
#if DIM > 2
                az += g_az[j];
#endif
#endif
            }
            // set the accels and remember the feedback values in extra variables
            pointmass.ax[n] += ax;
            pointmass.feedback_ax[n] = ax;
#if DIM > 1
            pointmass.ay[n] += ay;
            pointmass.feedback_ay[n] = ay;
#if DIM > 2
            pointmass.az[n] += az;
            pointmass.feedback_az[n] = az;
#endif
#endif    
#if DEBUG_GRAVITY
            printf("id:%d ax=%e ay=%e az=%e\n", n, ax, ay, az);
#endif
            blockCount = 0;
        }
    }

}


// adds the acceleration due to the point masses
__global__ void gravitation_from_point_masses(int calculate_nbody)
{
    int i, inc;
    int j;
    int d;
    double r;
    double rrr;
    double dr[DIM];
    inc = blockDim.x * gridDim.x;
    // loop for point masses
    if (calculate_nbody) {
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
                pointmass.ax[i] += gravConst * pointmass.m[j] * dr[0]/(rrr);
    #if DIM > 1
                pointmass.ay[i] += gravConst * pointmass.m[j] * dr[1]/(rrr);
    #if DIM > 2
                pointmass.az[i] += gravConst * pointmass.m[j] * dr[2]/(rrr);
    #endif
    #endif
            }
        }
    } // if calculate_nbody


    // loop over all particles
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numRealParticles; i += inc) {
        if (p_rhs.materialId[i] == EOS_TYPE_IGNORE || matEOS[p_rhs.materialId[i]] == EOS_TYPE_IGNORE) {
            continue;
        }

        double smlcubed;
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
	        smlcubed = p.h[i]*p.h[i]*p.h[i];
	        if (rrr < smlcubed) {
	    	    rrr = smlcubed;
	        }
            if (r < pointmass.rmax[j]) {
                p.ax[i] += gravConst * pointmass.m[j] * dr[0]/(rrr);
#if DIM > 1
                p.ay[i] += gravConst * pointmass.m[j] * dr[1]/(rrr);
#if DIM > 2
                p.az[i] += gravConst * pointmass.m[j] * dr[2]/(rrr);
#endif
#endif
            } else {
                p_rhs.materialId[i] = EOS_TYPE_IGNORE;
            }
            if (r < pointmass.rmin[j]) {
#if PARTICLE_ACCRETION
                p_rhs.materialId[i] = EOS_TYPE_ACCRETED;
#else
                p_rhs.materialId[i] = EOS_TYPE_IGNORE;
#endif  // PARTICLE_ACCRETION
            }
        }
    }
}
#endif //GRAVITATING_POINT_MASSES

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
		    f = gravConst * p.m[j]; // / (distance*distance*distance);
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
						f = gravConst * p.m[child]; // / (distance*distance*distance);
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
