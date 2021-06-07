 /* @author      Christoph Schaefer cm.schaefer@gmail.com and Thomas I. Maindl
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

// flag for FIXED_BINARY
int feedback_cnt = 0;

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
    int do_calculate = 0;
    // value for the time being
    const int h_blocksize = 256;
    double *g_x, *g_y, *g_z;
    double *torque_x, *torque_y, *torque_z;
    double *Power_x, *Power_y, *Power_z;

    // check flag if we use the coupled heun_rk4 integrator
    // if so, return
    if (!calculate_nbody) {
        return;
    }

    cudaMalloc((void **) &g_x, h_blocksize*sizeof(double));
    cudaMalloc((void **) &Power_x, h_blocksize*sizeof(double));
#if DIM > 1
    cudaMalloc((void **) &g_y, h_blocksize*sizeof(double));
    cudaMalloc((void **) &Power_y, h_blocksize*sizeof(double));
    cudaMalloc((void **) &torque_z, h_blocksize*sizeof(double));
#if DIM > 2
    cudaMalloc((void **) &g_z, h_blocksize*sizeof(double));
    cudaMalloc((void **) &torque_x, h_blocksize*sizeof(double));
    cudaMalloc((void **) &torque_y, h_blocksize*sizeof(double));
    cudaMalloc((void **) &Power_z, h_blocksize*sizeof(double));
#endif
#endif

    for (n = 0; n < numberOfPointmasses; n++) {
// fixed binary always ignores the backreaction of the disc but needs to calculate it for the torque measurement
#if !(FIXED_BINARY)
        if (!pointmass_host.feels_particles[n]) {
            continue;
        }
#endif
#if FIXED_BINARY
        if (n == 0 && feedback_cnt <= 9) {
            fprintf(stdout, "SKIPPING FEEDBACK CALC SINCE COUNTER IS %d\n", feedback_cnt);
            feedback_cnt += 1;
        }
        if (feedback_cnt == 10) {   // calculate torques only every 10th rhs call since it's so expensive
            do_calculate = 1;
        }
#else
        do_calculate = 1;
#endif
        if (do_calculate) {
            fprintf(stdout, "Calculating force from particles on star/planet no. %d\n", n);
            cudaVerifyKernel((particles_gravitational_feedback<<<h_blocksize, NUM_THREADS_REDUCTION>>>(n, g_x, g_y, g_z, torque_x, torque_y, torque_z, Power_x, Power_y, Power_z)));
            cudaVerify(cudaDeviceSynchronize());
            if (n == numberOfPointmasses-1) {
                feedback_cnt = 0;
            }
        }
    }


    cudaFree(g_x);
    cudaFree(Power_x);
#if DIM > 1
    cudaFree(g_y);
    cudaFree(Power_y);
    cudaFree(torque_z);
#if DIM > 2
    cudaFree(g_z);
    cudaFree(torque_x);
    cudaFree(torque_y);
    cudaFree(Power_z);
#endif
#endif
}


__device__ void get_acceleration_by_particle(int n, double *ax, double *ay, double *az, double *tx, double *ty, double *tz, double *Px, double *Py, double *Pz,  int i)
{
    // do some more magic here
    int d;
    double r = 0.0;
    double rrr;
    double smlcubed;
    double dr[DIM];
    double v_star[DIM];
    double r_star[DIM];

    dr[0] = pointmass.x[n] - p.x[i];
    v_star[0] = pointmass.vx[n];
    r_star[0] = pointmass.x[n];
    m_star = pointmass.m[n];
#if DIM > 1
    dr[1] = pointmass.y[n] - p.y[i];
    v_star[1] = pointmass.vy[n];
    r_star[1] = pointmass.y[n];
#if DIM > 2
    dr[2] = pointmass.z[n] - p.z[i];
    v_star[2] = pointmass.vz[n];
    r_star[2] = pointmass.z[n];
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
        *ax = -C_GRAVITY * p.m[i] * dr[0]/(rrr);
        *Px = *ax * v_star[0];
#if DIM > 1
        *ay = -C_GRAVITY * p.m[i] * dr[1]/(rrr);
        *tz = m_star * (r_star[0] * (*ay) - r_star[1] * (*ax));
        *Py = *ay * v_star[1];
#if DIM > 2
        *az = -C_GRAVITY * p.m[i] * dr[2]/(rrr);
        *tx = m_star * (r_star[1] * (*az) - r_star[2] * (*ay));
        *ty = m_star * (r_star[2] * (*ax) - r_star[0] * (*az));
        *Pz = *az * v_star[2];
    
    //printf("Device function check: torque_z from single particule is %.9f \n", *tz);
#endif
#endif
    } else {
        *ax = 0.0;
        *ay = 0.0;
        *az = 0.0;
        *tx = 0.0;
        *ty = 0.0;
        *tz = 0.0;
        *Px = 0.0;
        *Py = 0.0;
        *Pz = 0.0;
    }
}



__global__ void particles_gravitational_feedback(int n, double *g_ax, double *g_ay, double *g_az, double *g_tx, double *g_ty, double *g_tz, double *g_Px, double *g_Py, double *g_Pz)
{
    int idx, inc;
    int i;
    int tid;
    int j, k, m;
    volatile double local_ax, local_ay, local_az;
    volatile double local_tx, local_ty, local_tz;
    volatile double local_Px, local_Py, local_Pz;
    double ax, ay, az;
    double tx, ty, tz;
    double Px, Py, Pz;
    __shared__ double sh_ax[NUM_THREADS_REDUCTION];
    __shared__ double sh_ay[NUM_THREADS_REDUCTION];
    __shared__ double sh_az[NUM_THREADS_REDUCTION];
    __shared__ double sh_tx[NUM_THREADS_REDUCTION];
    __shared__ double sh_ty[NUM_THREADS_REDUCTION];
    __shared__ double sh_tz[NUM_THREADS_REDUCTION];
    __shared__ double sh_Px[NUM_THREADS_REDUCTION];
    __shared__ double sh_Py[NUM_THREADS_REDUCTION];
    __shared__ double sh_Pz[NUM_THREADS_REDUCTION];

    tid = threadIdx.x;
    idx = threadIdx.x + blockIdx.x * blockDim.x;
    inc = blockDim.x * gridDim.x;
    local_ax = local_ay = local_az = 0.0;
    local_tx = local_ty = local_tz = 0.0;
    local_Px = local_Py = local_Pz = 0.0;
   
            //Reset torques and power
    pointmass_rhs.Power_x[n] = 0.0;
#if DIM > 1
    pointmass_rhs.torque_z[n] = 0.0;
    pointmass_rhs.Power_y[n] = 0.0;
#if DIM > 2
    pointmass_rhs.Power_z[n] = 0.0;
    pointmass_rhs.torque_x[n] = 0.0;
    pointmass_rhs.torque_y[n] = 0.0;
#endif
#endif
 

    for (i = idx; i < numParticles; i += inc) {
        if (p_rhs.materialId[i] == EOS_TYPE_IGNORE) {
            continue;
        }
        ax = ay = az = 0.0;
        tx = ty = tz = 0.0;
        Px = Py = Pz = 0.0;
        get_acceleration_by_particle(n, &ax, &ay, &az, &tx, &ty, &tz, &Px, &Py, &Pz, i);
        //printf("Device function check: torque_z from single particule is %.9f \n", tz);
        local_ax += ax;
        local_Px += Px;
#if DIM > 1
        local_ay += ay;
        local_tz += tz;
        local_Py += Py; 
#if DIM > 2
        local_az += az;
        local_tx += tx;
        local_ty += ty;
        local_Pz += Pz;
#endif
#endif
    }
    sh_ax[tid] = local_ax;
    sh_Px[tid] = local_Px;
#if DIM > 1
    sh_ay[tid] = local_ay;
    sh_tz[tid] = local_tz;
    sh_Py[tid] = local_Py;
#if DIM > 2
    sh_az[tid] = local_az;
    sh_tx[tid] = local_tx;
    sh_ty[tid] = local_ty;
    sh_Pz[tid] = local_Pz;
#endif
#endif
    for (j = NUM_THREADS_REDUCTION/2; j > 0; j>>=1) {
        __syncthreads();
        if (tid < j) {
            sh_ax[tid] = local_ax = sh_ax[tid+j] + local_ax;
            sh_Px[tid] = local_Px = sh_Px[tid+j] + local_Px;
#if DIM > 1
            sh_ay[tid] = local_ay = sh_ay[tid+j] + local_ay;
            sh_tz[tid] = local_tz = sh_tz[tid+j] + local_tz;
            sh_Py[tid] = local_Py = sh_Py[tid+j] + local_Py;
#if DIM > 2
            sh_az[tid] = local_az = sh_az[tid+j] + local_az;
            sh_tx[tid] = local_tx = sh_tx[tid+j] + local_tx;
            sh_ty[tid] = local_ty = sh_ty[tid+j] + local_ty;
            sh_Pz[tid] = local_Pz = sh_Pz[tid+j] + local_Pz;
#endif
#endif
        }
    }
    // write block result to global memory
    ax = 0;
    ay = 0;
    az = 0;
    tx = 0;
    ty = 0;
    tz = 0;
    Px = 0;
    Py = 0;
    Pz = 0;
    // only first thread in each block
    if (tid == 0) {
        k = blockIdx.x;
        g_ax[k] = local_ax;
        g_Px[k] = local_Px;
#if DIM > 1
        g_ay[k] = local_ay;
        g_Py[k] = local_Py;
        g_tz[k] = local_tz;
#if DIM > 2
        g_az[k] = local_az;
        g_Pz[k] = local_Pz;
        g_tx[k] = local_tx;
        g_ty[k] = local_ty;
#endif
#endif
        m = gridDim.x-1;
        if (m == atomicInc((unsigned int *) &blockCount, m)) {
            // last block, combine all results
            for (j = 0; j <= m; j++) {
                ax += g_ax[j];
                Px += g_Px[j];
#if DIM > 1
                ay += g_ay[j];
                Py += g_Py[j];
                tz += g_tz[j];
#if DIM > 2
                az += g_az[j];
                Pz += g_Pz[j];
                tx += g_tx[j];
                ty += g_ty[j];
#endif
#endif
            }
            // set the accels and remember the feedback values in extra variables
#if !(FIXED_BINARY)
            pointmass.ax[n] += ax;               
#endif
            pointmass.feedback_ax[n] += ax;
            pointmass_rhs.Power_x[n] += Px;
#if DIM > 1
#if !(FIXED_BINARY)
            pointmass.ay[n] += ay; 
#endif
            pointmass.feedback_ay[n] += ay;
            pointmass_rhs.Power_y[n] += Py;  // minus here because we consider the star PoV
            pointmass_rhs.torque_z[n] += tz;
            if (idx == 0) {
                printf("The gravitational torque on the binary is %.17le \n", tz);
                printf("The power exerted on the binary is %.17le and %.17le \n", Px, Py);
            }
#if DIM > 2
#if !(FIXED_BINARY)
            pointmass.az[n] += az;
#endif 
            pointmass.feedback_az[n] += az;
            pointmass_rhs.Power_z[n] += Pz;   // minus here because star PoV
            pointmass_rhs.torque_x[n] += tx;
            pointmass_rhs.torque_y[n] += ty;
#endif
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
                pointmass.ax[i] += C_GRAVITY * pointmass.m[j] * dr[0]/(rrr);
    #if DIM > 1
                pointmass.ay[i] += C_GRAVITY * pointmass.m[j] * dr[1]/(rrr);
    #if DIM > 2
                pointmass.az[i] += C_GRAVITY * pointmass.m[j] * dr[2]/(rrr);
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
            if (r < pointmass.rmax[j] & r > pointmass.rmin[j]) {
                p.ax[i] += C_GRAVITY * pointmass.m[j] * dr[0]/(rrr);
#if DIM > 1
                p.ay[i] += C_GRAVITY * pointmass.m[j] * dr[1]/(rrr);
#if DIM > 2
                p.az[i] += C_GRAVITY * pointmass.m[j] * dr[2]/(rrr);
#endif
#endif
            }
#if PARTICLE_ACCRETION                                    // Need to add r.max criteria for deactivation but later because
                                                          // now current work on torks
            if (r < pointmass.rmin[j]) {
                p_rhs.materialId[i] = EOS_TYPE_ACCRETED;
            }
#endif
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
		    f = C_GRAVITY * p.m[j]; // / (distance*distance*distance);
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
						f = C_GRAVITY * p.m[child]; // / (distance*distance*distance);
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
