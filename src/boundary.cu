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
#include "timeintegration.h"
#include "boundary.h"
#include "miluph.h"
#include "pressure.h"
#include "config_parameter.h"


extern __device__ double substep_currentTimeD;
extern __device__ double currentTimeD;
extern __device__ double dt;

#if GHOST_BOUNDARIES
/* these are the locations and the properties of the boundary walls */
const __device__ int numWalls = 1;
__device__ double d[numWalls] = {-0.007};
__device__ double nx[numWalls] = {0};
__device__ double ny[numWalls] = {0};
#if DIM == 3
__device__ double nz[numWalls] = {1};
#endif
    //boundary type: 0 = no slip, 1 = free slip
#define NO_SLIP_BOUNDARY_TYPE 0
#define FREE_SLIP_BOUNDARY_TYPE 1
__device__ int boundaryType[numWalls] = {NO_SLIP_BOUNDARY_TYPE};
#endif



/* set quantities for Fixed Virtual Particles with matId == BOUNDARY_PARTICLE_ID */
__device__ void setQuantitiesFixedVirtualParticles(int i, int j, double *vxj, double *vyj, double *vzj, double *densityj, double *pressurej, double *Sj)
{
    /* j is the virtual particle, i is the real particle */
    int e;
    /* distance to plane */
    double dI, dJ;
    double beta;
    double oneminusbeta = 0;
#define BETA_MAX 1.5

#if DIM > 2
    /* test values only for plane at z=0 */
    dI = p.z[i];
    dJ = p.z[j];

    beta = min(BETA_MAX, 1.0 + dJ/dI);
    oneminusbeta = 1-beta;
#endif

    *vxj = oneminusbeta*p.vx[i];
#if DIM > 1
    *vyj = oneminusbeta*p.vy[i];
#if DIM > 2
    *vzj = oneminusbeta*p.vz[i];
#endif
#endif
#if SOLID
    for (e = 0; e < DIM*DIM; e++) {
        Sj[e] = p.S[i*DIM*DIM+e];
    }
#endif

    *pressurej = p.p[i];
    *densityj = p.rho[i];
}



// declare some boundary conditions here: this is called at the beginning of each RHS step
__global__ void BoundaryConditionsBeforeRHS(int *interactions) 
{
#if 1
    register int i, inc;
    int matId, d, e;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];
        
        if (matId == EOS_TYPE_IGNORE) {
            p.ax[i] = 0;
#if DIM > 1
            p.ay[i] = 0;
            p.dydt[i] = 0;
            p.vy[i] = 0;
#endif
            p.dxdt[i] = 0;
            p.vx[i] = 0;
#if DIM > 2
            p.az[i] = 0;
            p.dzdt[i] = 0;
            p.vz[i] = 0;
#endif
#if SOLID
            for (d = 0; d < DIM*DIM; d++) {
                p.dSdt[i*DIM*DIM + d] = 0;
            }
#endif
#if INTEGRATE_DENSITY
            p.drhodt[i] = 0;
#endif
        }

        if (matId == BOUNDARY_PARTICLE_ID) {
            p.ax[i] = 0;
            p.vx[i] = 0;
            p.dxdt[i] = 0;
#if DIM > 1
            p.dydt[i] = 0;
            p.ay[i] = 0;
            p.vy[i] = 0;
#endif
#if DIM > 2
            p.az[i] = 0;
            p.dzdt[i] = 0;
            p.vz[i] = 0;
#endif
#if SOLID
            for (d = 0; d < DIM*DIM; d++) {
                p.dSdt[i*DIM*DIM + d] = 0;
            }
#endif
#if INTEGRATE_DENSITY
            p.drhodt[i] = 0;
#endif
        } else {
            if( p.rho[i] < matDensityFloor[matId] ) {
                p.rho[i] = matDensityFloor[matId];
#if INTEGRATE_DENSITY
                p.drhodt[i] = 0.0;
#endif
            }
            if( p.e[i] < matEnergyFloor[matId] ) {
                p.e[i] = matEnergyFloor[matId];
#if INTEGRATE_ENERGY
                p.dedt[i] = 0.0;
#endif
            }
        }
    }
#endif
}


// boundary conditions called after the integration step of rk2adaptive only
__global__ void BoundaryConditionsBeforeIntegratorStep(int *interactions) 
{
    register int i, inc;
    int matId, d, e;
    double distance;
    double ddistance;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];
        // unset all deactivation flags
        p_rhs.deactivate_me_flag[i] = 0;
#if 0 // deactivated, usually not wanted
        // deactivate particles that have no interaction partners
        if (p.noi[i] < 1 && matId != EOS_TYPE_IGNORE) {
            p_rhs.materialId[i] = EOS_TYPE_IGNORE;
#if DIM > 2
            printf("DEBUG: Deactivating particle %d at position %e %e %e with speed %e %e %e and density %e and energy %e\n", i, p.x[i], p.y[i], p.z[i], p.vx[i], p.vy[i], p.vz[i], p.rho[i], p.e[i]);
#endif
        }
#endif
    }
}


// boundary conditions called after the integration step of rk2adaptive only
__global__ void BoundaryConditionsAfterIntegratorStep(int *interactions) 
{
    register int i, inc;
    int matId, d, e;
    double distance;
    double ddistance;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];
#if 0 // deactivated, usually not wanted
        // deactivate particles that have no interaction partners
        if (p.noi[i] < 1 && matId != EOS_TYPE_IGNORE) {
            p_rhs.materialId[i] = EOS_TYPE_IGNORE;
#if DIM > 2
            printf("DEBUG: Deactivating particle %d at position %e %e %e with speed %e %e %e and density %e and energy %e\n", i, p.x[i], p.y[i], p.z[i], p.vx[i], p.vy[i], p.vz[i], p.rho[i], p.e[i]);
#endif
        }
#endif
    }
}



// declare some boundary conditions here: this is called at the end of each RHS step
__global__ void BoundaryConditionsAfterRHS(int *interactions) 
{
#if 1
    register int i, inc;
    int matId, d, e;
    double distance;
    double ddistance;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];

        if (matId == EOS_TYPE_IGNORE) {
            p.ax[i] = 0;
            p.dxdt[i] = 0;
            p.vx[i] = 0;
#if DIM > 1
            p.dydt[i] = 0;
            p.ay[i] = 0;
            p.vy[i] = 0;
#endif
#if DIM > 2
            p.az[i] = 0;
            p.dzdt[i] = 0;
            p.vz[i] = 0;
#endif
#if SOLID
            for (d = 0; d < DIM*DIM; d++) {
                p.dSdt[i*DIM*DIM + d] = 0;
            }
#endif
#if INTEGRATE_DENSITY
            p.drhodt[i] = 0;
#endif
        }

// adding central star with one solar mass
// at (0,0)            

#if 0
        distance = 0.0;
        ddistance = p.x[i]*p.x[i] + p.y[i]*p.y[i];
        distance = sqrt(ddistance);
        distance *= ddistance;
        p.ax[i] -= 1.327474512e+20 * p.x[i] / distance;
        p.ay[i] -= 1.327474512e+20 * p.y[i] / distance;
#endif

        // feel the Earth!
#if 0
        if (p.y[i] < 0.0) {
            p.vy[i] = 0.0;
            p.ay[i] = 0.0;
        } else {
            p.ay[i] -= 9.81;
        }
#endif

        // do not fall below y = 0.0

        /* let's stick to the ground */
#if 0
        if (p.z[i] <= 1e-3) {
            p.ax[i] = 0;
            p.ay[i] = 0;
            p.dxdt[i] = 0;
            p.dydt[i] = 0;
            p.vx[i] = 0;
            p.vy[i] = 0;
#if DIM == 3
            p.az[i] = 0;
            p.dzdt[i] = 0;
            p.vz[i] = 0;
#endif
        }
#endif

        if (matId == BOUNDARY_PARTICLE_ID) {
            p.ax[i] = 0;
            p.vx[i] = 0;
            p.dxdt[i] = 0;
#if DIM > 1
            p.dydt[i] = 0;
            p.ay[i] = 0;
            p.vy[i] = 0;
#endif
#if DIM > 2
            p.az[i] = 0;
            p.dzdt[i] = 0;
            p.vz[i] = 0;
#endif
#if SOLID
            for (d = 0; d < DIM*DIM; d++) {
                p.dSdt[i*DIM*DIM + d] = 0;
            }
#endif
#if INTEGRATE_DENSITY
            p.drhodt[i] = 0;
#endif
        } else {
            if( p.rho[i] < matDensityFloor[matId] ) {
                p.rho[i] = matDensityFloor[matId];
#if INTEGRATE_DENSITY
                p.drhodt[i] = 0.0;
#endif
            }
            if( p.e[i] < matEnergyFloor[matId] ) {
                p.e[i] = matEnergyFloor[matId];
#if INTEGRATE_ENERGY
                p.dedt[i] = 0.0;
#endif
            }
        }
    }
#endif
}



#if GHOST_BOUNDARIES
__global__ void removeGhostParticles()
{
    //call with only one thread and one block
    numParticles = numRealParticles;
}



/* set the density, pressure and other quantities for the ghost particles */
__global__ void setQuantitiesGhostParticles() 
{
    register int i, inc, k, idx, currentNumParticles;
    register int pidx;
#if SOLID
    int a, b;
#endif
    double normalVel;
    double x, y;
#if DIM == 3
    double z;
#endif


/* for NO_SLIP_BOUNDARY_TYPE, we stored i
   for FREE_SLIP_BOUNDARY_TYPE, we stored -i see function insertGhostParticles() below */
    inc = blockDim.x * gridDim.x;
    /* loop over all ghost particles */
    for (i = numRealParticles + threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i +=inc) {
        /* the index of the corresponding real particle shifted by 1 (since we need the sign) */
        idx = p.real_partner[i];
        if (idx < 0) {
            pidx = -idx;
        } else { 
            pidx = idx;
        }
        pidx -= 1;

        //mirror particle
        p.cs[i] = p.cs[pidx];
        p.p[i] = p.p[pidx];
        p.e[i] = p.e[pidx];
        p.rho[i] = p.rho[pidx];
#if SOLID
        /* set deviatoric stress tensor depending on boundary type */
        if (idx > 0) { /* NO_SLIP_BOUNDARY */ 
            for (a = 0; a < DIM; a++) {
                for (b = 0; b < DIM; b++) {
                    p.S[i*DIM*DIM+a*DIM+b] = p.S[pidx*DIM*DIM+a*DIM+b];
                }
            }
        } else if (idx < 0) { /* FREE_SLIP_BOUNDARY */
            for (a = 0; a < DIM; a++) {
                for (b = 0; b < DIM; b++) {
                    p.S[i*DIM*DIM+a*DIM+b] = -p.S[pidx*DIM*DIM+a*DIM+b];
                }
                if (matEOS[p_rhs.materialId[i]] == EOS_TYPE_REGOLITH) {
                    p.S[i*DIM*DIM+a*DIM+a] *= -1;
                }
            }
        } else {
            printf("Error, cannot happen. Go away!\n");
            assert(false);
        }
#endif


    }
}



/* sets the location, mass, sml for the ghost particles */
__global__ void insertGhostParticles()
{
    //call with only one block
    int i, inc, k;
    volatile int idx;
#if SOLID
    int a, b;
#endif
    double sml;
    double distance;
    double normalVel;
    double x, y;

#if DIM == 3
    double z;
#endif
    //boundary type: 0 = no slip, 1 = free slip
    inc = blockDim.x * gridDim.x;
    for (k = 0; k < numWalls; k++) {
        __syncthreads();
        int currentNumParticles = numParticles;
        for (i = threadIdx.x + blockIdx.x * blockDim.x; i < currentNumParticles; i += inc) {
            double sml;
            sml = p.h[i];

            x = p.x[i];
            y = p.y[i];
#if DIM == 3 
            z = p.z[i];
#endif


            //get distance to wall
            distance = x*nx[k] + y*ny[k]-d[k];
#if DIM == 3
            distance += z*nz[k];
#endif


            //if distance small enough
            if (fabs(distance) <= sml/2.0) {

                //atomic read and increment of numParticles
                idx = atomicAdd(&numParticles, 1);
                assert(idx < maxNumParticles);

#if 1 // moved to extra function!
                //mirror particle
#if (VARIABLE_SML || INTEGRATE_SML || DEAL_WITH_TOO_MANY_INTERACTIONS)
                p.h[idx] = sml;
#endif
                p.noi[idx] = p.noi[i];
                p.cs[idx] = p.cs[i];
                p.depth[idx] = p.depth[i];
                p.p[idx] = p.p[i];
              //  p.e[idx] = p.e[i];
                p_rhs.materialId[idx] = p_rhs.materialId[i];

                p.m[idx] = p.m[i];
                p.rho[idx] = p.rho[i];
#endif

                /* set location of ghost particle */
                p.x[idx] = x - 2*distance*nx[k];
                p.y[idx] = y - 2*distance*ny[k];
#if DIM == 3
                p.z[idx] = z - 2*distance*nz[k];
#endif
                /* remember the real particle where the ghost particle
                   originates from */
                /* for NO_SLIP_BOUNDARY_TYPE, we store i
                   for FREE_SLIP_BOUNDARY_TYPE, we store -i */
#if 1
                if (boundaryType[k] == NO_SLIP_BOUNDARY_TYPE) {
                    p.real_partner[idx] = i+1;
                } else if (boundaryType[k] == FREE_SLIP_BOUNDARY_TYPE) {
                    p.real_partner[idx] = -i-1;
                } else {
                    printf("Error: no such boundary type for particle.\n");
                    assert(false);
                }
#endif
                /* set mass and material type and sml */
                p.h[idx] = sml;
                p_rhs.materialId[idx] = p_rhs.materialId[i];

                /* all other quantities are set in function setQuantitiesGhostParticles() */
                if (boundaryType[k] == NO_SLIP_BOUNDARY_TYPE) {
                    //free slip boundary
                    p.vx[idx] = -p.vx[i];
#if DIM > 1
                    p.vy[idx] = -p.vy[i];
#if DIM == 3
                    p.vz[idx] = -p.vz[i];
#endif
#endif
#if 0
#if SOLID
                    for (a = 0; a < DIM; a++) {
                        for (b = 0; b < DIM; b++) {
                            p.S[idx*DIM*DIM+a*DIM+b] = p.S[i*DIM*DIM+a*DIM+b];
                        }
                    }
#endif
#endif

                } else if (boundaryType[k] == FREE_SLIP_BOUNDARY_TYPE) {
                    //free slip boundary

                    normalVel = nx[k]*p.vx[i];
#if DIM > 1
                    normalVel += ny[k]*p.vy[i];
#endif
#if DIM == 3
                    normalVel += nz[k]*p.vz[i];
#endif

                    p.vx[idx] = p.vx[i] - 2*normalVel*nx[k];
#if DIM > 1
                    p.vy[idx] = p.vy[i] - 2*normalVel*ny[k];
#if DIM == 3
                    p.vz[idx] = p.vz[i] - 2*normalVel*nz[k];
#endif
#endif
#if 0
#if SOLID
                    for (a = 0; a < DIM; a++) {
                        for (b = 0; b < DIM; b++) {
                            p.S[idx*DIM*DIM+a*DIM+b] = -p.S[i*DIM*DIM+a*DIM+b];
                        }
                        p.S[idx*DIM*DIM+a*DIM+a] *= -1;
                    }
#endif
#endif
                }

            } //end distance if
        } //end particle loop
        __syncthreads();
    } //end wall loop
    if (threadIdx.x + blockIdx.x*blockDim.x == 0) {
        printf("number of particles after inserting: %d\t\t", numParticles);
        printf("added %d particles\n", numParticles - numRealParticles);
    }
}
#endif



/* this function places the brushes according to their rotation speed */
__global__ void BoundaryConditionsBrushesBefore(int *interactions) 
{
#if 0
#warning: brushes on
    register int i, inc;
    int matId, d, e;
    inc = blockDim.x * gridDim.x;

    // revolutions per minute
    const double rpm = 100;
    const double omega = rpm * 2 * M_PI / 60;

    // the offset, this is printed out by brush3D.py
    const double yoff = 0.0523512;

    double phi0 = 0;
    double phi = 0;
    double phit = 0; // no, it's really a p
    double r = 0;
    double y = 0;
    double vz = -1e-2; // speed of the brushes in z direction 
    double zoff = 0;
    double zoffangle = 0.170125; // the z-offset, this is printed out by brush3D.py
    double zmax = -0.02; ; // = 1/6 brushdiameter from brush3D.py
    double myz = 0.0;

    zoff = substep_currentTimeD * vz;
    if (zoff < zmax) 
        zoff = zmax;

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
            matId = p_rhs.materialId[i];
            if (matId > 1) {
            // new rotating angle
                phi = omega*substep_currentTimeD;
                // original angle
                // new angle
// brush left (y<0) is matId = 1 and rotates counterclockwise
// brush right (y>0) is matId = 2 and rotates clockwise
// rotation is around x axis
                myz = p.z0[i] - zoffangle;
                if (matId == 2) {
                    y = p.y0[i] + yoff;
                    phi0 = atan2(myz,y);
                    phit = phi + phi0;
                    r = myz * myz + y*y; 
                    r = sqrt(r);
                    p.y[i] = r * cos(phit) - yoff;
                    // coordinates 
                    p.z[i] = r * sin(phit) + zoff + zoffangle;
                    p.x[i] = p.x0[i];

                    // velocity
                    p.vx[i] = 0.0;
                    p.vy[i] = -omega * r * sin(phit);
                    p.vz[i] = omega * r * cos(phit);

                } else if (matId == 3) {
                    y = p.y0[i] - yoff;
                    phi0 = atan2(myz,y);
                    phit = phi0 - phi;
                    r = myz * myz + y*y; 
                    r = sqrt(r);
                    p.y[i] = r * cos(phit) + yoff;
                    // coordinates 
                    p.z[i] = r * sin(phit) + zoff + zoffangle;
                    p.x[i] = p.x0[i];

                    // velocity
                    p.vx[i] = 0.0;
                    p.vy[i] = omega * r * sin(phit);
                    p.vz[i] = -omega * r * cos(phit);
                } 
            }
    }
#endif
}



__global__ void BoundaryConditionsBrushesAfter(int *interactions) 
{
#if 0
#warning: brushes on
    register int i, inc;
    int matId, d, e;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
            matId = p_rhs.materialId[i];
            if (matId > 0) {
                p.ax[i] = 0;
                p.ay[i] = 0;
                p.dxdt[i] = 0;
                p.dydt[i] = 0;
                p.vx[i] = 0;
                p.vy[i] = 0;
                for (d = 0; d < DIM*DIM; d++) {
                    p.dSdt[i*DIM*DIM + d] = 0;
                }
                p.drhodt[i] = 0;
            } else {
            }
    }
#endif
}



__global__ void BoundaryForce(int *interactions) 
{
#if 0
#warning: brushes on
	register int64_t interactions_index;
    register int i, inc;
    int matId, d, e, matIdj;
    int k, j, numInteractions;
    double distance;
    double ljf = 0;
    // D is somehow related to the largest velocity
    double D = 10.0;
    const double tiny = 1e-6;
    const double r0 = 0.022574999999999998;
    double dx, dy, dz;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
            matId = p_rhs.materialId[i];
            // only for regolith with matId == 0
            if (matId > 0)
                continue;
            numInteractions = p.noi[i];
            for (k = 0; k < numInteractions; k++) {
            // the interaction partner
                interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + k;
                j = interactions[interactions_index];

                // check if interaction partner is boundary_particle and if not, continue
                matIdj = p_rhs.materialId[j];
                if (matIdj == BOUNDARY_PARTICLE_ID) {
                // calculate lennard jones force
                    dx = p.x[i] - p.x[j];
                    dy = p.y[i] - p.y[j];
                    dz = 0.0;
#if DIM > 2
                    dz = p.z[i] - p.z[j];
#endif
                    distance = dx*dx + dy*dy + dz*dz;
                    distance += tiny;
                    distance = sqrt(distance);
                    if (r0/distance < 1) {
                        ljf =  D * (pow(r0/distance, 12) - pow(r0/distance, 6)) * pow(distance, -2);
                        p.ax[i] -= ljf*dx;
                        p.ay[i] -= ljf*dy;
#if DIM > 2
                        p.az[i] -= ljf*dz;
#endif
                    } 
                } 

#if 0            
                // check if interaction partner is brush and if not, continue
                matIdj = p_rhs.materialId[j];
                if (matIdj == 1 || matIdj == 2) {
                // calculate lennard jones force
                    dx = p.x[i] - p.x[j];
                    dy = p.y[i] - p.y[j];
                    dz = p.z[i] - p.z[j];
                    distance = dx*dx + dy*dy + dz*dz;
                    distance += tiny;
                    distance = sqrt(distance);
                    if (r0/distance < 1) {
                        ljf = p.m[i] *  D * (pow(r0/distance, 12) - pow(r0/distance, 4)) * pow(distance, -2);
                        p.ax[i] += ljf*dx;
                        p.ay[i] += ljf*dy;
                        p.az[i] += ljf*dz;
                    } 
                } 
#endif
            }
    }
#endif
}
