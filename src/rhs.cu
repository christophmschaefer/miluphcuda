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
    register int64_t interactions_index;
    register size_t i, inc, dd;
#if SOLID
    register int ddd;
#endif
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        //printf("DEBUG: zeroing derivatives for particle %zu\n", i);
        // printf("DEBUG: zeroing derivatives for particle %llu\n", i);
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
            interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + dd;
            interactions[interactions_index] = -1;
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
#if DEBUG_RHS_RUNTIMES
    cudaEvent_t start, stop;
    float time[MAX_NUMBER_PROFILED_KERNELS];
    float totalTime = 0.0;
    int timerCounter = 0;
#endif
#if DEBUG_TREE
    double xmin, xmax, ymin, ymax, zmin, zmax;
    double radiusmax, radiusmin;
    int *treeDepthPerBlock;
    int maxtreedepth_host = 0;
    int maxNodeIndex_host;
#endif
    int *movingparticlesPerBlock;
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

#if DEBUG_RHS_RUNTIMES
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    cudaVerify(cudaMemset(childListd, EMPTY, memorySizeForChildren));
    cudaVerify(cudaDeviceSynchronize());

#if DEBUG_RHS || DEBUG_TIMESTEP
    fprintf(stdout, "rhs call\n");
#endif

    // zero all accelerations
#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
#endif
    cudaVerifyKernel((zero_all_derivatives<<<numberOfMultiprocessors, NUM_THREADS_256>>>(interactions)));
#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration zeroing all: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

    // check if boundary conditions are violated
    cudaVerifyKernel((BoundaryConditionsBeforeRHS<<<16 * numberOfMultiprocessors, NUM_THREADS_BOUNDARY_CONDITIONS>>>(interactions)));

    cudaVerify(cudaDeviceSynchronize());

#if GHOST_BOUNDARIES
    /*
       the location of the ghost boundary particles are set. The quantities for the ghost particles will
       be set later on as soon as we know the quantities for the real particles (density, pressure...)
     */
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((insertGhostParticles<<<4 * numberOfMultiprocessors, NUM_THREADS_BOUNDARY_CONDITIONS>>>()));
    //cudaVerifyKernel((insertGhostParticles<<<1, 1>>>()));
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration inserting ghost particles: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif

    cudaVerify(cudaDeviceSynchronize());

#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
#endif
    cudaVerifyKernel((computationalDomain<<<numberOfMultiprocessors, NUM_THREADS_COMPUTATIONAL_DOMAIN>>>(
                    minxPerBlock, maxxPerBlock
#if DIM > 1
                    , minyPerBlock, maxyPerBlock
#endif
#if DIM == 3
                    , minzPerBlock, maxzPerBlock
#endif
                    )));
#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration comp domain: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

    cudaVerify(cudaDeviceSynchronize());

#if DEBUG_TREE
    cudaMemcpyFromSymbol(&xmin, minx, sizeof(double));
    cudaMemcpyFromSymbol(&xmax, maxx, sizeof(double));
    radiusmax = xmax - xmin;
# if DIM > 1
    cudaMemcpyFromSymbol(&ymin, miny, sizeof(double));
    cudaMemcpyFromSymbol(&ymax, maxy, sizeof(double));
    radiusmax = max(radiusmax, ymax-ymin);
# endif
# if DIM == 3
    cudaMemcpyFromSymbol(&zmin, minz, sizeof(double));
    cudaMemcpyFromSymbol(&zmax, maxz, sizeof(double));
    radiusmax = max(radiusmax, zmax-zmin);
# endif
    printf("computational domain: x [%e, %e]", xmin, xmax);
# if DIM > 1
    printf(", y [%e, %e]", ymin, ymax);
# endif
# if DIM == 3
    printf(", z [%e, %e]", zmin, zmax);
# endif
    printf("\n");
#endif  // DEBUG_TREE

#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
#endif
    cudaVerifyKernel((buildTree<<<numberOfMultiprocessors, NUM_THREADS_BUILD_TREE>>>()));
    cudaVerify(cudaDeviceSynchronize());
#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration build tree: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

#if DEBUG_TREE
    cudaMemcpyFromSymbol(&maxNodeIndex_host, maxNodeIndex, sizeof(int));
    fprintf(stdout, "number of inner nodes: %d\n", (numberOfNodes - maxNodeIndex_host));
    fprintf(stdout, "highest index number in tree: %d\n", maxNodeIndex_host);
    fprintf(stdout, "number of used inner nodes / number of allocated nodes: %.7f %%\n",
            100.0 * (float)(numberOfNodes - maxNodeIndex_host) / (float)(numberOfNodes - numberOfParticles));
    // get maximum depth of tree
    cudaVerify(cudaMalloc((void**)&treeDepthPerBlock, sizeof(int)*numberOfMultiprocessors));
    cudaVerifyKernel((getTreeDepth<<<numberOfMultiprocessors, NUM_THREADS_TREEDEPTH>>>(treeDepthPerBlock)));
    cudaMemcpyFromSymbol(&maxtreedepth_host, treeMaxDepth, sizeof(int));
    fprintf(stdout, "max depth of tree: %d\n", maxtreedepth_host);
    radiusmin = radiusmax * pow(0.5, maxtreedepth_host-1);
    fprintf(stdout, "largest node length: %g \t smallest node length: %g\n", radiusmax, radiusmin);
    cudaVerify(cudaFree(treeDepthPerBlock));
#endif

    cudaVerify(cudaDeviceSynchronize());

#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
#endif
#if VARIABLE_SML
    // boundary conditions for sml
# if DEBUG_RHS
    printf("calling check_sml_boundary\n");
# endif
    cudaVerifyKernel((check_sml_boundary<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>()));
    cudaVerify(cudaDeviceSynchronize());
#endif
#if VARIABLE_SML && FIXED_NOI
    // call only for the fixed number of interactions case
    // if INTEGRATE_SML, the sml is integrated and we only need to symmetrize the interactions later on
# if DEBUG_RHS
    printf("calling knnNeighbourSearch\n");
# endif
    cudaVerifyKernel((knnNeighbourSearch<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>(
                    interactions)));
    cudaVerify(cudaDeviceSynchronize());
#endif
#if DEAL_WITH_TOO_MANY_INTERACTIONS // make sure that a particle does not get more than MAX_NUM_INTERACTIONS
# if DEBUG_RHS
    printf("calling nearNeighbourSearch_modify_sml\n");
# endif
    cudaVerifyKernel((nearNeighbourSearch_modify_sml<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>(
                    interactions)));
#else // risk a termination if MAX_NUM_INTERACTIONS is reached for one particle
# if DEBUG_RHS
    printf("calling nearNeighbourSearch\n");
# endif
    cudaVerifyKernel((nearNeighbourSearch<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>(
                    interactions)));
#endif
    cudaVerify(cudaDeviceSynchronize());
#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration neighboursearch: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif
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

//#if !INTEGRATE_DENSITY
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((calculateDensity<<<numberOfMultiprocessors * 4, NUM_THREADS_DENSITY>>>( interactions)));
//    cudaVerifyKernel((calculateDensity<<<1,1>>>( interactions)));
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration density: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
//#endif

#if SHEPARD_CORRECTION
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((shepardCorrection<<<numberOfMultiprocessors*4, NUM_THREADS_256>>>( interactions)));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration shepard correction: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
    //cudaVerifyKernel((printTensorialCorrectionMatrix<<<1,1>>>( interactions)));
#endif

#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
#endif
    cudaVerifyKernel((calculateSoundSpeed<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
    cudaVerify(cudaDeviceSynchronize());
#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration soundspeed: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

#if (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH)
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((CalcDivvandCurlv<<<numberOfMultiprocessors * 4, NUM_THREADS_128>>>(
                    interactions)));
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration div v and curl v: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif

#if SIRONO_POROSITY
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((calculateCompressiveStrength<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
    cudaVerify(cudaDeviceSynchronize());
    cudaVerifyKernel((calculateTensileStrength<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration compressive, tensile and shear strength: %.2f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif

#if PURE_REGOLITH
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((plasticity<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration plasticity: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif

#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
#endif
    cudaVerifyKernel((calculatePressure<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));
    cudaVerify(cudaDeviceSynchronize());
#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration pressure: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif
/*  function is not in porosity.cu anymore but in timeintecration.cu internal forces
#if PALPHA_POROSITY
    cudaVerifyKernel((calculateDistensionChange<<<numberOfMultiprocessors * 4, NUM_THREADS_PALPHA_POROSITY>>>()));
    cudaVerify(cudaDeviceSynchronize());
#endif
*/

    if (param.selfgravity) {
#if DEBUG_RHS_RUNTIMES
        cudaEventRecord(start, 0);
#endif
        cudaVerifyKernel((calculateCentersOfMass<<<1, NUM_THREADS_CALC_CENTER_OF_MASS>>>()));
        cudaVerify(cudaDeviceSynchronize());
#if DEBUG_RHS_RUNTIMES
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[timerCounter], start, stop);
        printf("duration calc center of mass: %.7f ms\n", time[timerCounter]);
        totalTime += time[timerCounter++];
#endif
    }

#if INVISCID_SPH
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((betaviscosity<<<numberOfMultiprocessors * 4, NUM_THREADS_128>>>(
		    interactions)));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration betaviscosity: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif

#if (SYMMETRIC_STRESSTENSOR || FRAGMENTATION || PLASTICITY)
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((symmetrizeStress<<<4 * numberOfMultiprocessors, NUM_THREADS_512>>>()));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration symmetrize stress tensor: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif

#if FRAGMENTATION
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((damageLimit<<<numberOfMultiprocessors*4, NUM_THREADS_512>>>()));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration damage limit: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
    fflush(stdout);
#endif

#if PLASTICITY
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((plasticityModel<<<numberOfMultiprocessors * 4, NUM_THREADS_512>>>()));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration plasticityModel: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif

#if JC_PLASTICITY
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((JohnsonCookPlasticity<<<numberOfMultiprocessors * 4, NUM_THREADS_512>>>()));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration johnson-cook: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif

#if TENSORIAL_CORRECTION
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((tensorialCorrection<<<numberOfMultiprocessors*4, NUM_THREADS_256>>>( interactions)));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration tensorial correction: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
//    cudaVerifyKernel((printTensorialCorrectionMatrix<<<1,1>>>( interactions)));
#endif

#if VISCOUS_REGOLITH
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((calculatedeviatoricStress<<<numberOfMultiprocessors*4, NUM_THREADS_256>>>( interactions)));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration viscous regolith : %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
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
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((setQuantitiesGhostParticles<<<numberOfMultiprocessors, NUM_THREADS_BOUNDARY_CONDITIONS>>>()));
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration quantities ghost particles: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif

#if DEBUG_MISC
    fprintf(stdout, "checking correlation matrix\n");
    fflush(stdout);
    cudaVerifyKernel((checkNaNs<<<numberOfMultiprocessors, NUM_THREADS_128>>>(interactions)));
    cudaVerify(cudaDeviceSynchronize());
    fprintf(stdout, "starting internalForces\n");
    fflush(stdout);
#endif

#if SOLID
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((set_stress_tensor<<<numberOfMultiprocessors, NUM_THREADS_256>>>()));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration set stress tensor: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif

#if NAVIER_STOKES
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((calculate_kinematic_viscosity<<<numberOfMultiprocessors, NUM_THREADS_256>>>()));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration calculation kinematic viscosity: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif

#if NAVIER_STOKES
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((calculate_shear_stress_tensor<<<numberOfMultiprocessors, NUM_THREADS_256>>>(interactions)));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration calculation shear stress tensor: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif


#if ARTIFICIAL_STRESS
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((compute_artificial_stress<<<numberOfMultiprocessors, NUM_THREADS_256>>>(interactions)));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration artificial_stress: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif

    // the main loop, where all accelerations are calculated
#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
#endif
    cudaVerifyKernel((internalForces<<<numberOfMultiprocessors, NUM_THREADS_128>>>(interactions)));
    //cudaVerifyKernel((internalForces<<<1, 1 >>>(interactions)));
    cudaVerify(cudaDeviceSynchronize());
#if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration internal forces: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
#endif

#if GRAVITATING_POINT_MASSES
    // interaction with the point masses
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    cudaVerifyKernel((gravitation_from_point_masses<<<numberOfMultiprocessors, NUM_THREADS_128>>>(calculate_nbody)));
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration gravitation from point masses: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
    // back reaction from the disk
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(start, 0);
# endif
    backreaction_from_disk_to_point_masses(calculate_nbody);
    cudaVerify(cudaDeviceSynchronize());
# if DEBUG_RHS_RUNTIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time[timerCounter], start, stop);
    printf("duration backreaction from the particles on pointmasses: %.7f ms\n", time[timerCounter]);
    totalTime += time[timerCounter++];
# endif
#endif

#if DEBUG_MISC
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
#if DEBUG_RHS_RUNTIMES
        cudaEventRecord(start, 0);
#endif
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
#if DEBUG_GRAVITY
        fprintf(stdout, "%d particles change their nodes, this is a fraction of %g %% (currently allowed max is 0.1 %%)\n",
                movingparticles_host, changefraction*1e2);
#endif
        if (changefraction > 1e-3) {
            flag_force_gravity_calc = 1;
            cudaMemcpyToSymbol(reset_movingparticles, &flag_force_gravity_calc, sizeof(int));
        }
        /* free mem */
        cudaVerify(cudaFree(movingparticlesPerBlock));
#if DEBUG_RHS_RUNTIMES
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[timerCounter], start, stop);
        printf("duration tree changes: %.7f ms\n", time[timerCounter]);
        totalTime += time[timerCounter++];
#endif
    }

    /* self-gravitation using TREE */
    if (param.selfgravity) {
#if DEBUG_RHS_RUNTIMES
        cudaEventRecord(start, 0);
#endif
        if (!param.decouplegravity)
            flag_force_gravity_calc = 1;
        if (flag_force_gravity_calc) {
#if DEBUG_GRAVITY
            fprintf(stdout, "calculating self-gravity using new tree\n");
#endif
            cudaVerifyKernel((selfgravity<<<16*numberOfMultiprocessors, NUM_THREADS_SELFGRAVITY>>>()));
            flag_force_gravity_calc = 0;
            cudaMemcpyToSymbol(reset_movingparticles, &flag_force_gravity_calc, sizeof(int));
        } else {
#if DEBUG_GRAVITY
            printf("skipping calculation of self-gravity, using values from last timestep\n");
#endif
            cudaVerifyKernel((addoldselfgravity<<<16*numberOfMultiprocessors, NUM_THREADS_SELFGRAVITY>>>()));
        }
        cudaVerify(cudaDeviceSynchronize());
#if DEBUG_RHS_RUNTIMES
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[timerCounter], start, stop);
        printf("duration selfgravity: %.7f ms\n", time[timerCounter]);
        totalTime += time[timerCounter++];
#endif
        gravity_index++;
    }

    /* self gravitation using particle-particle forces */
    if (param.directselfgravity) {
#if DEBUG_GRAVITY
        fprintf(stdout, "calculating self-gravity using n**2 algorithm\n");
#endif
#if DEBUG_RHS_RUNTIMES
        cudaEventRecord(start, 0);
#endif
        cudaVerifyKernel((direct_selfgravity<<<numberOfMultiprocessors, NUM_THREADS_SELFGRAVITY>>>()));
        cudaVerify(cudaDeviceSynchronize());
#if DEBUG_RHS_RUNTIMES
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[timerCounter], start, stop);
        printf("duration selfgravity: %.7f ms\n", time[timerCounter]);
        totalTime += time[timerCounter++];
#endif
    }


    /* set any special particle values */
    cudaVerifyKernel((BoundaryConditionsAfterRHS<<<16 * numberOfMultiprocessors, NUM_THREADS_BOUNDARY_CONDITIONS>>>(interactions)));

    // set dx/dt = v or dx/dt = v + dxsph/dt
    cudaVerifyKernel((setlocationchanges<<<4 * numberOfMultiprocessors, NUM_THREADS_512>>>(interactions)));


#if 0 // disabled, cms 2019-12-03: should be sufficient to do this at start of rhs
#if VARIABLE_SML && !READ_INITIAL_SML_FROM_PARTICLE_FILE
    // boundary conditions for the smoothing lengths
# if DEBUG_RHS
    printf("calling check_sml_boundary\n");
# endif
    cudaVerifyKernel((check_sml_boundary<<<numberOfMultiprocessors * 4, NUM_THREADS_NEIGHBOURSEARCH>>>()));
    cudaVerify(cudaDeviceSynchronize());
#endif
#endif // 0

#if DEBUG_RHS_RUNTIMES
    fprintf(stdout, "total duration rhs: %.7f ms\n", totalTime);
    if (param.performanceTest)
        write_performance(time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
}
