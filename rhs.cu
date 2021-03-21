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
#include "rhs.h"
#include "miluph.h"
#include "parameter.h"
#include "tree.h"
#include "boundary.h"
#include "density.h"
#include "plasticity.h"
#include "porosity.h"
#include "pressure.h"
#include "soundspeed.h"
#include "gravity.h"
#include "xsph.h"
#include "internal_forces.h"
#include "velocity.h"
#include "little_helpers.h"
#include "viscosity.h"
#include "artificial_stress.h"
#include "stress.h"
#include "damage.h"

extern int flag_force_gravity_calc;
extern int gravity_index;
extern __device__ int movingparticles;
extern __device__ int reset_movingparticles;


extern __device__ volatile int maxNodeIndex;
extern __device__ int treeMaxDepth;

// tree computational domain
extern double *minxPerBlock, *maxxPerBlock;
extern __device__ double minx, maxx;
#if DIM > 1
extern double *minyPerBlock, *maxyPerBlock;
extern __device__ double miny, maxy;
#endif
#if DIM == 3
extern double *minzPerBlock, *maxzPerBlock;
extern __device__ double minz, maxz;
#endif

extern volatile int terminate_flag;

// zero all derivatives
__global__ void zero_all_derivatives(int *interactions)
{
    register int i, inc, dd;
#if SOLID
    register int ddd;
#endif
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        p.ax[i] = 0.0;
#if DIM > 1
        p.ay[i] = 0.0;
#if DIM > 2
        p.az[i] = 0.0;
#endif
#endif
#if INTEGRATE_SML
        p.dhdt[i] = 0.0;
#endif
        p.drhodt[i] = 0.0;
#if INTEGRATE_ENERGY
        p.dedt[i] = 0.0;
#endif
#if SHEPARD_CORRECTION
        p_rhs.shepard_correction[i] = 1.0;
#endif
#if SML_CORRECTION
        p.sml_omega[i] = 1.0;
#endif
#if SOLID
        for (dd = 0; dd < DIM*DIM; dd++) {
            p.dSdt[i*DIM*DIM+dd] = 0.0;
            p_rhs.sigma[i*DIM*DIM+dd] = 0.0;
        }
#if TENSORIAL_CORRECTION
        for (dd = 0; dd < DIM; dd++) {
            for (ddd = 0; ddd < DIM; ddd++) {
                p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+dd*DIM+ddd] = 0.0;
                if (dd == ddd) {
                    p_rhs.tensorialCorrectionMatrix[i*DIM*DIM+dd*DIM+ddd] = 1.0;
                }
            }
        }
#endif
#endif
        // reset all interactions
        for (dd = 0; dd < MAX_NUM_INTERACTIONS; dd++) {
            interactions[i*MAX_NUM_INTERACTIONS + dd] = -1;
        }

#if FRAGMENTATION
        p.dddt[i] = 0.0;
#endif

    }
#if GRAVITATING_POINT_MASSES
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numPointmasses; i += inc) {
        pointmass.ax[i] = 0.0;
        pointmass.feedback_ax[i] = 0.0;
#if DIM > 1
        pointmass.ay[i] = 0.0;
        pointmass.feedback_ay[i] = 0.0;
#if DIM > 2
        pointmass.az[i] = 0.0;
        pointmass.feedback_az[i] = 0.0;
#endif
#endif
    }
#endif // GRAVITATING_POINT_MASSES
}


/* determine all derivatives */
void rightHandSide()
{
    cudaEvent_t start, stop;
    float time[MAX_NUMBER_PROFILED_KERNELS];
    float totalTime = 0;
    double radiusmax, radiusmin;
    int timerCounter = 0;
    int *treeDepthPerBlock;
    int *movingparticlesPerBlock;
    int maxtreedepth_host = 0;
    int movingparticles_host = 0;
    int calculate_nbody = 0;


#if GRAVITATING_POINT_MASSES
    if (param.integrator_type == HEUN_RK4) {
        calculate_nbody = 0;
    } else {
        calculate_nbody = 1;
    }
#endif


#if USE_SIGNAL_HANDLER
    if (terminate_flag) {
        copyToHostAndWriteToFile(-2, -2);
    }
#endif

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaVerify(cudaMemset(childListd, EMPTY, memorySizeForChildren));
    cudaVerify(cudaDeviceSynchronize());


    if (param.verbose) fprintf(stdout, "rhs call\n");

    // zero all accelerations
    cudaEventRecord(start, 0);
    cudaVerifyKernel((zero_all_derivatives<<<numberOfMultiprocessors, NUM_THREADS_256>>>(interactions)));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration zeroing all: %.7f ms\n", time[timerCounter]);

    // check if boundary conditions are violated
    cudaVerifyKernel((BoundaryConditionsBeforeRHS<<<16 * numberOfMultiprocessors, NUM_THREADS_BOUNDARY_CONDITIONS>>>(interactions)));

    cudaVerify(cudaDeviceSynchronize());
#if GHOST_BOUNDARIES
    /*
       the location of the ghost boundary particles are set. The quantities for the ghost particles will
       be set later on as soon as we know the quantities for the real particles (density, pressure...)
     */
    cudaEventRecord(start, 0);
    cudaVerifyKernel((insertGhostParticles<<<4 * numberOfMultiprocessors, NUM_THREADS_BOUNDARY_CONDITIONS>>>()));
    //cudaVerifyKernel((insertGhostParticles<<<1, 1>>>()));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration inserting ghost particles: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

    cudaVerify(cudaDeviceSynchronize());

    cudaEventRecord(start, 0);
    cudaVerifyKernel((computationalDomain<<<numberOfMultiprocessors, NUM_THREADS_COMPUTATIONAL_DOMAIN>>>(
                    minxPerBlock, maxxPerBlock
#if DIM > 1
                    , minyPerBlock, maxyPerBlock
#endif
#if DIM == 3
                    , minzPerBlock, maxzPerBlock
#endif
                    )));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration comp domain: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
    if (param.verbose || param.decouplegravity) {
        double xmin, xmax;
#if DIM > 1
        double ymin, ymax;
#endif
#if DIM == 3
        double zmin, zmax;
#endif
        cudaMemcpyFromSymbol(&xmin, minx, sizeof(double));
        cudaMemcpyFromSymbol(&xmax, maxx, sizeof(double));
#if DIM > 1
        cudaMemcpyFromSymbol(&ymin, miny, sizeof(double));
        cudaMemcpyFromSymbol(&ymax, maxy, sizeof(double));
#endif
#if DIM == 3
        cudaMemcpyFromSymbol(&zmin, minz, sizeof(double));
        cudaMemcpyFromSymbol(&zmax, maxz, sizeof(double));
#endif
        radiusmax = xmax - xmin;
#if DIM > 1
        radiusmax = max(radiusmax, ymax-ymin);
#endif
        if (param.verbose) {
            printf("computational domain: x [%e, %e]", xmin, xmax);
#if DIM > 1
            printf(", y [%e, %e]", ymin, ymax);
#endif
#if DIM == 3
            printf(", z [%e, %e]", zmin, zmax);
            radiusmax = max(radiusmax, zmax-zmin);
#endif
            printf("\n");
        }
    }

    cudaVerify(cudaDeviceSynchronize());

    cudaEventRecord(start, 0);
    cudaVerifyKernel((buildTree<<<numberOfMultiprocessors, NUM_THREADS_BUILD_TREE>>>()));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    cudaMemcpyFromSymbol(&maxNodeIndex_host, maxNodeIndex, sizeof(int));
    if (param.verbose) fprintf(stdout, "build tree duration: %.7f ms\n", time[timerCounter]);
    if (param.verbose) fprintf(stdout, "number of inner nodes: %d\n", (numberOfNodes - maxNodeIndex_host));
    if (param.verbose) fprintf(stdout, "number of used inner nodes / number of allocated nodes: %.7f %%\n", 100.0 * (float)(numberOfNodes - maxNodeIndex_host) / (float)(numberOfNodes - numberOfParticles));


    // get maximum depth of tree
    if (param.decouplegravity || param.treeinformation) {
        cudaVerify(cudaMalloc((void**)&treeDepthPerBlock, sizeof(int)*numberOfMultiprocessors));
        if (param.verbose) fprintf(stdout, "Determing depth of tree\n");
        cudaVerifyKernel((getTreeDepth<<<numberOfMultiprocessors, NUM_THREADS_TREEDEPTH>>>(treeDepthPerBlock)));
        cudaMemcpyFromSymbol(&maxtreedepth_host, treeMaxDepth, sizeof(int));
        if (param.verbose) fprintf(stdout, "Maximum depth of tree is: %d\n", maxtreedepth_host);
        radiusmin = radiusmax * pow(0.5, maxtreedepth_host-1);
        if (param.verbose) fprintf(stdout, "Largest node length: %g \t smallest node length: %g\n", radiusmax, radiusmin);
        cudaVerify(cudaFree(treeDepthPerBlock));
    }


    totalTime += time[timerCounter++];
    cudaVerify(cudaDeviceSynchronize());

    cudaEventRecord(start, 0);


#if VARIABLE_SML
    // boundary conditions for the smoothing lengths
    if (param.verbose) printf("calling check_sml_boundary\n");
    cudaVerifyKernel((check_sml_boundary<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>()));
    cudaVerify(cudaDeviceSynchronize());
#endif

#if VARIABLE_SML && FIXED_NOI
    // call only for the fixed number of interactions case
    // if INTEGRATE_SML is set, the sml is integrated and we only need to symmetrize the interactions
    // later on
    if (param.verbose) printf("calling knnNeighbourSearch\n");
    cudaVerifyKernel((knnNeighbourSearch<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>(
                    interactions)));
    cudaVerify(cudaDeviceSynchronize());
#endif

#if DEAL_WITH_TOO_MANY_INTERACTIONS // make sure that a particle does not get more than MAX_NUM_INTERACTIONS
    if (param.verbose) printf("calling nearNeighbourSearch_modify_sml\n");
    cudaVerifyKernel((nearNeighbourSearch_modify_sml<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>(
                    interactions)));
#else // risk a termination if MAX_NUM_INTERACTIONS is reached for one particle
    if (param.verbose) printf("calling nearNeighbourSearch\n");
    cudaVerifyKernel((nearNeighbourSearch<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>(
                    interactions)));
#endif


    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration neighboursearch: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
    cudaVerifyKernel((setEmptyMassForInnerNodes<<<numberOfMultiprocessors * 4, NUM_THREADS_512>>>()));
    cudaVerify(cudaDeviceSynchronize());
    // TODO: only if debug
#if 0
    cudaMemcpy(p_host.noi, p_device.noi, memorySizeForInteractions, cudaMemcpyDeviceToHost);
    cudaVerify(cudaDeviceSynchronize());
    int i;
    int maxNumInteractions = 0;
    for (i = 0; i < numberOfParticles; i++) {
        maxNumInteractions = max(maxNumInteractions, p_host.noi[i]);
        if (maxNumInteractions > MAX_NUM_INTERACTIONS) {
            fprintf(stderr, "max num interactions exceeded by particle %d\n", i);
            exit(1);
        }
    }
    printf("maximum number of interactions: %d\n", maxNumInteractions);
#endif

    time[timerCounter] = 0;

#if !INTEGRATE_DENSITY
    cudaEventRecord(start, 0);
    cudaVerifyKernel((calculateDensity<<<numberOfMultiprocessors * 4, NUM_THREADS_DENSITY>>>( interactions)));
//    cudaVerifyKernel((calculateDensity<<<1,1>>>( interactions)));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration density: %.7f ms\n", time[timerCounter]);
#endif
    totalTime += time[timerCounter++];


#if SHEPARD_CORRECTION
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((shepardCorrection<<<numberOfMultiprocessors*4, NUM_THREADS_256>>>( interactions)));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration shepard correction: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
    //cudaVerifyKernel((printTensorialCorrectionMatrix<<<1,1>>>( interactions)));
#endif
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((calculateSoundSpeed<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration soundspeed: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];

#if (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH)
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((CalcDivvandCurlv<<<numberOfMultiprocessors * 4, NUM_THREADS_128>>>(
                    interactions)));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration div v and curl v: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

#if SIRONO_POROSITY
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((calculateCompressiveStrength<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
    cudaVerify(cudaDeviceSynchronize());
    cudaVerifyKernel((calculateTensileStrength<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration compressive, tensile and shear strength: %.2f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
    time[timerCounter] = 0;
#endif

#if PURE_REGOLITH
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((plasticity<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration plasticity: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((calculatePressure<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration pressure: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
/*  function is not in porosity.cu anymore but in timeintecration.cu internal forces
#if PALPHA_POROSITY
    cudaVerifyKernel((calculateDistensionChange<<<numberOfMultiprocessors * 4, NUM_THREADS_PALPHA_POROSITY>>>()));
    cudaVerify(cudaDeviceSynchronize());
#endif
*/

    time[timerCounter] = 0;
    if (param.selfgravity) {
        cudaEventRecord(start, 0);
        cudaVerifyKernel((calculateCentersOfMass<<<1, NUM_THREADS_CALC_CENTER_OF_MASS>>>()));
        cudaVerify(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[timerCounter], start, stop);
        if (param.verbose) {
            printf("duration calc center of mass: %.7f ms\n", time[timerCounter]);
        }
    }
    totalTime += time[timerCounter++];

#if INVISCID_SPH
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((betaviscosity<<<numberOfMultiprocessors * 4, NUM_THREADS_128>>>(
		    interactions)));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration betaviscosity: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif


#if (SYMMETRIC_STRESSTENSOR || FRAGMENTATION || PLASTICITY)
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((symmetrizeStress<<<4 * numberOfMultiprocessors, NUM_THREADS_512>>>()));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration symmetrize stress tensor: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

#if PLASTICITY
    cudaEventRecord(start, 0);
    time[timerCounter] = 0;
    cudaVerifyKernel((plasticityModel<<<numberOfMultiprocessors * 4, NUM_THREADS_512>>>()));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration plasticity: %.7f ms\n", time[timerCounter]);
    cudaEventRecord(start, 0);
    totalTime += time[timerCounter++];
#endif

#if JC_PLASTICITY
    cudaEventRecord(start, 0);
    time[timerCounter] = 0;
    cudaVerifyKernel((JohnsonCookPlasticity<<<numberOfMultiprocessors * 4, NUM_THREADS_512>>>()));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration johnson-cook: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

#if FRAGMENTATION
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((damageLimit<<<numberOfMultiprocessors*4, NUM_THREADS_512>>>()));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration damage limit: %.7f ms\n", time[timerCounter]);
    fflush(stdout);
    totalTime += time[timerCounter++];
#endif

#if TENSORIAL_CORRECTION
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((tensorialCorrection<<<numberOfMultiprocessors*4, NUM_THREADS_256>>>( interactions)));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration tensorial correction: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
//    cudaVerifyKernel((printTensorialCorrectionMatrix<<<1,1>>>( interactions)));
#endif

#if VISCOUS_REGOLITH
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((calculatedeviatoricStress<<<numberOfMultiprocessors*4, NUM_THREADS_256>>>( interactions)));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration viscous regolith : %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

#if XSPH
    cudaVerify(cudaDeviceSynchronize());
    cudaVerifyKernel((calculateXSPHchanges<<<4 * numberOfMultiprocessors, NUM_THREADS_512>>>(interactions)));
#endif /*XSPH */

#if GHOST_BOUNDARIES
    /*
       the location of the ghost boundary particles are set. The quantities for the ghost particles will
       be set later on as soon as we know the quantities for the real particles (density, pressure...)
     */
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((setQuantitiesGhostParticles<<<numberOfMultiprocessors, NUM_THREADS_BOUNDARY_CONDITIONS>>>()));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration quantities ghost particles: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

#if DEBUG
    if (param.verbose) fprintf(stdout, "checking correlation matrix\n");
    fflush(stdout);
    cudaVerifyKernel((checkNaNs<<<numberOfMultiprocessors, NUM_THREADS_128>>>(interactions)));
    cudaVerify(cudaDeviceSynchronize());
    if (param.verbose) fprintf(stdout, "starting internalForces\n");
    fflush(stdout);
#endif

#if SOLID
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((set_stress_tensor<<<numberOfMultiprocessors, NUM_THREADS_256>>>()));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration set stress tensor: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

#if NAVIER_STOKES
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((calculate_kinematic_viscosity<<<numberOfMultiprocessors, NUM_THREADS_256>>>()));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration calculation kinematic viscosity: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

#if NAVIER_STOKES
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((calculate_shear_stress_tensor<<<numberOfMultiprocessors, NUM_THREADS_256>>>(interactions)));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration calculation shear stress tensor: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif


#if ARTIFICIAL_STRESS
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((compute_artificial_stress<<<numberOfMultiprocessors, NUM_THREADS_256>>>(interactions)));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration artificial_stress: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

    // the main loop, where all accelerations are calculated
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((internalForces<<<numberOfMultiprocessors, NUM_THREADS_128>>>(interactions)));
    //cudaVerifyKernel((internalForces<<<1, 1 >>>(interactions)));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration internal forces: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];

#if GRAVITATING_POINT_MASSES
    // interaction with the point masses
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    cudaVerifyKernel((gravitation_from_point_masses<<<numberOfMultiprocessors, NUM_THREADS_128>>>(calculate_nbody)));
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration gravitation from point masses: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif
#if GRAVITATING_POINT_MASSES
    // back reaction from the disk
    time[timerCounter] = 0;
    cudaEventRecord(start, 0);
    backreaction_from_disk_to_point_masses(calculate_nbody);
    cudaVerify(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    if (param.verbose) printf("duration backreaction from the particles on pointmasses: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

#if DEBUG
    if (param.verbose) fprintf(stdout, "checking for nans after internal_forces\n");
    fflush(stdout);
    cudaVerifyKernel((checkNaNs<<<numberOfMultiprocessors, NUM_THREADS_128>>>(interactions)));
    cudaVerify(cudaDeviceSynchronize());
    if (param.verbose) fprintf(stdout, "starting internalForces\n");
    fflush(stdout);
#endif

#if GHOST_BOUNDARIES
    cudaVerifyKernel((removeGhostParticles<<<1,1>>>()));
    cudaVerify(cudaDeviceSynchronize());
#endif


    /* check if we need the nbody-tree stuff has to be re-organised or
       if we could use the node masses and positions of last time step */

    if (param.selfgravity && param.decouplegravity) {
        time[timerCounter] = 0;
        cudaEventRecord(start, 0);
        if (gravity_index%10 == 0) {
            flag_force_gravity_calc = 1;
        }
        /* alloc mem */
        cudaVerify(cudaMalloc((void**)&movingparticlesPerBlock, sizeof(int)*numberOfMultiprocessors));
        /* determine how many particles will change their node */
        cudaVerifyKernel(((measureTreeChange<<<numberOfMultiprocessors, NUM_THREADS_TREECHANGE>>>(movingparticlesPerBlock))));
        /* get number of changing particles */
        cudaMemcpyFromSymbol(&movingparticles_host, movingparticles, sizeof(int));
        double changefraction = movingparticles_host*1.0/numberOfParticles;
        if (param.verbose) {
            fprintf(stdout, "%d particles change their nodes, this is a fraction of %g %% \n", movingparticles_host, changefraction*1e2);
            fprintf(stdout, "currently allowed maximum fraction is 0.1 %%.\n");
        }
        if (changefraction > 1e-3) {
            flag_force_gravity_calc = 1;
            cudaMemcpyToSymbol(reset_movingparticles, &flag_force_gravity_calc, sizeof(int));
        }
        /* free mem */
        cudaVerify(cudaFree(movingparticlesPerBlock));
        if (param.verbose) printf("duration tree changes: %.7f ms\n", time[timerCounter]);
        totalTime += time[timerCounter++];
    }

    /* self-gravitation using TREE */
    time[timerCounter] = 0;
    if (param.selfgravity) {
        cudaEventRecord(start, 0);
        if (!param.decouplegravity)
            flag_force_gravity_calc = 1;
        if (flag_force_gravity_calc) {
            if (param.verbose) fprintf(stdout, "Calculating accelerations using new tree.\n");
            cudaVerifyKernel((selfgravity<<<16*numberOfMultiprocessors, NUM_THREADS_SELFGRAVITY>>>()));
            flag_force_gravity_calc = 0;
            cudaMemcpyToSymbol(reset_movingparticles, &flag_force_gravity_calc, sizeof(int));
        } else {
            if (param.verbose) printf("Skipping calculation of self_gravity, using values from last timestep.\n");
            cudaVerifyKernel((addoldselfgravity<<<16*numberOfMultiprocessors, NUM_THREADS_SELFGRAVITY>>>()));
        }
        cudaVerify(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        gravity_index++;
        cudaEventElapsedTime(&time[timerCounter], start, stop);
        if (param.verbose)
            printf("duration selfgravity: %.7f ms\n", time[timerCounter]);
    }

    /* self gravitation using particle-particle forces */
    if (param.directselfgravity) {
        if (param.verbose) fprintf(stdout, "Calculating accelerations using n**2 algorithm.\n");
        cudaVerifyKernel((direct_selfgravity<<<numberOfMultiprocessors, NUM_THREADS_SELFGRAVITY>>>()));
        cudaVerify(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[timerCounter], start, stop);
        if (param.verbose)
            printf("duration selfgravity: %.7f ms\n", time[timerCounter]);
    }


    totalTime += time[timerCounter];

    /* set any special particle values */
    cudaVerifyKernel((BoundaryConditionsAfterRHS<<<16 * numberOfMultiprocessors, NUM_THREADS_BOUNDARY_CONDITIONS>>>(interactions)));

    // set dx/dt = v or dx/dt = v + dxsph/dt
    cudaVerifyKernel((setlocationchanges<<<4 * numberOfMultiprocessors, NUM_THREADS_512>>>(interactions)));


#if 0 // disabled, cms 2019-12-03: should be sufficient to do this at start of rhs
#if VARIABLE_SML && !READ_INITIAL_SML_FROM_PARTICLE_FILE
    // boundary conditions for the smoothing lengths
    if (param.verbose) printf("calling check_sml_boundary\n");
    cudaVerifyKernel((check_sml_boundary<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>()));
    cudaVerify(cudaDeviceSynchronize());
#endif
#endif // 0

    if (param.verbose) fprintf(stdout, "total duration right hand side: %.7f ms\n", totalTime);

    if (param.performanceTest) {
        write_performance(time);
    }

}
