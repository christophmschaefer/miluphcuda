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

#include "tree.h"
#include "timeintegration.h"
#include "config_parameter.h"
#include "parameter.h"
#include "miluph.h"
#include "pressure.h"


// do not iterate more than MAX_VARIABLE_SML_ITERATIONS times to get the desired number of interaction partners
// if VARIABLE_SML and FIXED_NOI is set
#define MAX_VARIABLE_SML_ITERATIONS 4
// tolerance value. if found number of interactions is as close as TOLERANCE_WANTED_NUMBER_OF_INTERACTIONS, we stop iterating
#define TOLERANCE_WANTED_NUMBER_OF_INTERACTIONS 5


__device__ int treeMaxDepth = 0;

extern __device__ double dt;
extern __device__ volatile double radius;
extern __device__ volatile int maxNodeIndex;
extern __device__ int blockCount;
extern __device__ double minx, maxx;
extern __device__ double miny, maxy;
#if DIM == 3
extern __device__ double minz, maxz;
#endif
extern __device__ int movingparticles;
extern __device__ int reset_movingparticles;
extern __constant__ volatile int *childList;



__device__ int childListIndex(int nodeIndex, int childNumber) {
    return (nodeIndex - numParticles) * numChildren + childNumber;
}


__global__ void setEmptyMassForInnerNodes(void) {
    int k;
    for(k = maxNodeIndex + (threadIdx.x + blockIdx.x * blockDim.x); k < numNodes; k += blockDim.x * gridDim.x) {
        p.m[k] = EMPTY;
    }
}




__global__ void buildTree()
{
	register int inc = blockDim.x * gridDim.x;
	register int i = threadIdx.x + blockIdx.x * blockDim.x;
	register int k;
	register int childIndex, child;
	register int lockedIndex;
	register double x;
#if DIM > 1
    register double y;
#endif
	register double r;
	register double dx;
#if DIM > 1
    register double dy;
#endif
	register double rootRadius = radius;
	register double rootX = p.x[numNodes-1];
#if DIM > 1
	register double rootY = p.y[numNodes-1];
#endif
    register int depth = 0;
	register int isNewParticle = TRUE;
	register int currentNodeIndex;
	register int newNodeIndex;
	register int subtreeNodeIndex;
#if DIM == 3
	register double z;
	register double dz;
	register double rootZ = p.z[numNodes-1];
#endif

    volatile double *px, *pm;
#if DIM > 1
    volatile double *py;
#if DIM == 3
    volatile double *pz;
#endif
#endif

    px  = p.x;
    pm = p.m;
#if DIM > 1
    py = p.y;
#if DIM == 3
    pz = p.z;
#endif
#endif

	while (i < numParticles) {
        depth = 0;

		if (isNewParticle) {
			isNewParticle = FALSE;
			// cache particle data
			x = px[i];
            p.ax[i] = 0.0;
#if DIM > 1
			y = py[i];
            p.ay[i] = 0.0;
#if DIM == 3
			z = pz[i];
            p.az[i] = 0.0;
#endif
#endif

			// start at root
			currentNodeIndex = numNodes-1;
			r = rootRadius;
			childIndex = 0;
			if (x > rootX) childIndex = 1;
#if DIM > 1
			if (y > rootY) childIndex += 2;
#if DIM == 3
			if (z > rootZ) childIndex += 4;
#endif
#endif
		}

		// follow path to leaf
		child = childList[childListIndex(currentNodeIndex, childIndex)];
        /* leaves are 0 ... numParticles */
		while (child >= numParticles) {
			currentNodeIndex = child;
            depth++;
			r *= 0.5;
			// which child?
			childIndex = 0;
			if (x > px[currentNodeIndex]) childIndex = 1;
#if DIM > 1
			if (y > py[currentNodeIndex]) childIndex += 2;
#if DIM > 2
			if (z > pz[currentNodeIndex]) childIndex += 4;
#endif
#endif
			child = childList[childListIndex(currentNodeIndex, childIndex)];
        }

		// we want to insert the current particle i into currentNodeIndex's child at position childIndex
		// where child is now empty, locked or a particle
		// if empty -> simply insert, if particle -> create new subtree
		if (child != LOCKED) {
			// the position where we want to place the particle gets locked
			lockedIndex = childListIndex(currentNodeIndex, childIndex);
			// atomic compare and save: compare if child is still the current value of childlist at the index lockedIndex, if so, lock it
			// atomicCAS returns the old value of child
			if (child == atomicCAS((int *) &childList[lockedIndex], child, LOCKED)) {
				// if the destination is empty, insert particle
				if (child == EMPTY) {
					// insert the particle into this leaf
					childList[lockedIndex] = i;
				} else {
					// there is already a particle, create new inner node
					subtreeNodeIndex = -1;
					do {
						// get the next free nodeIndex
						newNodeIndex = atomicSub((int * ) &maxNodeIndex, 1) - 1;

						// throw error if there aren't enough node indices available
						if (newNodeIndex <= numParticles) {
							printf("(thread %d): error during tree creation: not enough nodes. newNodeIndex %d, maxNodeIndex %d, numParticles: %d\n", threadIdx.x, newNodeIndex, maxNodeIndex, numParticles);
                            assert(0);
						}

						// the first available free nodeIndex will be the subtree node
						subtreeNodeIndex = max(subtreeNodeIndex, newNodeIndex);

						dx = (childIndex & 1) * r;
#if DIM > 1
						dy = ((childIndex >> 1) & 1) * r;
#if DIM == 3
						dz = ((childIndex >> 2) & 1) * r;
#endif
#endif
                        depth++;
						r *= 0.5;

						// we save the radius here, so we can use it during neighboursearch. we have to set it to EMPTY after the neighboursearch
						pm[newNodeIndex] = r;
						dx = px[newNodeIndex] = px[currentNodeIndex] - r + dx;
#if DIM > 1
						dy = py[newNodeIndex] = py[currentNodeIndex] - r + dy;
#if DIM == 3
						dz = pz[newNodeIndex] = pz[currentNodeIndex] - r + dz;
#endif
#endif

						for (k = 0; k < numChildren; k++) {
							childList[childListIndex(newNodeIndex, k)] = EMPTY;
						}

						if (subtreeNodeIndex != newNodeIndex) {
							// this condition is true when the two particles are so close to each other, that they are
							// again put into the same node, so we have to create another new inner node.
							// in this case, currentNodeIndex is the previous newNodeIndex
							// and childIndex is the place where the particle i belongs to, relative to the previous newNodeIndex
							childList[childListIndex(currentNodeIndex, childIndex)] = newNodeIndex;
						}

						childIndex = 0;
						if (px[child] > dx) childIndex = 1;
#if DIM > 1
						if (py[child] > dy) childIndex += 2;
#if DIM == 3
						if (pz[child] > dz) childIndex += 4;
#endif
#endif
						childList[childListIndex(newNodeIndex, childIndex)] = child;

						// compare positions of particle i to the new node
						currentNodeIndex = newNodeIndex;
						childIndex = 0;
						if (x > dx) childIndex = 1;
#if DIM > 1
						if (y > dy) childIndex += 2;
#if DIM == 3
						if (z > dz) childIndex += 4;
#endif
#endif
						child = childList[childListIndex(currentNodeIndex, childIndex)];
						// continue creating new nodes (with half radius each) until the other particle is not in the same spot in the tree
					} while (child >= 0);
					childList[childListIndex(currentNodeIndex, childIndex)] = i;
					__threadfence();
					//__threadfence() is used to halt the current thread until all previous writes to shared and global memory are visible
					// by other threads. It does not halt nor affect the position of other threads though!
					childList[lockedIndex] = subtreeNodeIndex;
				}
                p.depth[i] = depth;
				// continue with next particle
				i += inc;
				isNewParticle = TRUE;
			}
		}
		__syncthreads(); // child was locked, wait for other threads to unlock
	}
}



/* get the maximum tree depth */
__global__ void getTreeDepth(int *treeDepthPerBlock)
{
	register int i, j, k, m;
	__shared__ volatile int sharedtreeDepth[NUM_THREADS_TREEDEPTH];

    blockCount = 0;
    int localtreeDepth = 0;
	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += blockDim.x * gridDim.x) {
        localtreeDepth =  max(localtreeDepth, p.depth[i]);
    }

    i = threadIdx.x;
    sharedtreeDepth[i] = localtreeDepth;
    for (j = NUM_THREADS_TREEDEPTH / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i+j;
            sharedtreeDepth[i] = localtreeDepth = max(localtreeDepth, sharedtreeDepth[k]);
        }
    }

    // write block result to global memory
    if (i == 0) {
        k = blockIdx.x;
        treeDepthPerBlock[k] = localtreeDepth;
        m = gridDim.x-1;
        __threadfence();
        if (m == atomicInc((unsigned int *) &blockCount, m)) {
            for (j = 0; j <= m; j++) {
                localtreeDepth = max(localtreeDepth, treeDepthPerBlock[j]);
            }
            blockCount = 0;
        }
        treeMaxDepth = localtreeDepth;
    }

}



/* give an estimate how many particles will leave their leaves */
__global__ void measureTreeChange(int * movingparticlesPerBlock)
{
	register int i, j, k, m;
	__shared__ volatile int sharedmovingparticles[NUM_THREADS_TREECHANGE];
    double nodesize = 0;
    double distance = 0;

    blockCount = 0;

    int localmovingparticles = 0;
    int localdepth = 0;

	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += blockDim.x * gridDim.x) {
        localdepth = p.depth[i];
        nodesize = pow(0.5, localdepth) * radius;

// algorithm: determine if particle has moved more than 10% of cellsize of its original cell
        if (reset_movingparticles) {
            p_rhs.g_x[i] = p.x[i];
            p_rhs.g_local_cellsize[i] = nodesize*nodesize;
#if DIM > 1
            p_rhs.g_y[i] = p.y[i];
#if DIM > 2
            p_rhs.g_z[i] = p.z[i];
#endif
#endif
            distance = 0;
        } else {
            distance = (p.x[i] - p_rhs.g_x[i])*(p.x[i] - p_rhs.g_x[i]);
#if DIM > 1
            distance += (p.y[i] - p_rhs.g_y[i])*(p.y[i] - p_rhs.g_y[i]);
#if DIM > 2
            distance += (p.z[i] - p_rhs.g_z[i])*(p.z[i] - p_rhs.g_z[i]);
#endif
#endif
        }
        if (distance > p_rhs.g_local_cellsize[i]) {
            localmovingparticles++;
        }
    }
    i = threadIdx.x;
    sharedmovingparticles[i] = localmovingparticles;

    for (j = NUM_THREADS_TREECHANGE / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i+j;
            sharedmovingparticles[i] += sharedmovingparticles[k];
        }
    }

    // write block result to global memory
    if (i == 0) {
        localmovingparticles = 0;
        k = blockIdx.x;
        movingparticlesPerBlock[k] = sharedmovingparticles[i];
        m = gridDim.x - 1;
        __threadfence();
        if ((m == atomicInc((unsigned int *) &blockCount, m))) {
            /* last block, add all up */
            for (j = 0; j <= m; j++) {
                localmovingparticles += movingparticlesPerBlock[j];
            }
            blockCount = 0;
        }
        movingparticles = localmovingparticles;
    }

}



__global__ void calculateCentersOfMass()
{
	register int i, k, child, missing;
	register double m, cm, px;
#if DIM > 1
    register double py;
#endif
#if DIM == 3
	register double pz;
#endif
#if DIM == 3
	__shared__ volatile int sharedChildList[NUM_THREADS_CALC_CENTER_OF_MASS * 8];
#elif DIM == 2
	__shared__ volatile int sharedChildList[NUM_THREADS_CALC_CENTER_OF_MASS * 4];
#elif DIM == 1
	__shared__ volatile int sharedChildList[NUM_THREADS_CALC_CENTER_OF_MASS * 2];
#endif

	k = maxNodeIndex + (threadIdx.x + blockIdx.x * blockDim.x);

	missing = 0;
	while (k < numNodes) {
		if (missing == 0) {
			// new cell, so initialize
			cm = 0.0;
			px = 0.0;
#if DIM > 1
			py = 0.0;
#if DIM == 3
			pz = 0.0;
#endif
#endif
			for (i = 0; i < numChildren; i++) {
				child = childList[childListIndex(k, i)];
				if (child != EMPTY) {
					sharedChildList[missing * NUM_THREADS_CALC_CENTER_OF_MASS + threadIdx.x] = child; // cache missing children
					m = p.m[child];
					missing++;
					if (m >= 0.0) {
						// child is ready
						missing--;
						// add child's contribution
						cm += m;
						px += p.x[child] * m;
#if DIM > 1
						py += p.y[child] * m;
#if DIM == 3
						pz += p.z[child] * m;
#endif
#endif
					}
				}
			}
		}

		if (missing != 0) {
			do {
				// poll missing child
				child = sharedChildList[(missing - 1) * NUM_THREADS_CALC_CENTER_OF_MASS + threadIdx.x];
				m = p.m[child];
				if (m >= 0.0) {
					// child is now ready
					missing--;
					// add child's contribution
					cm += m;
					px += p.x[child] * m;
#if DIM > 1
					py += p.y[child] * m;
#if DIM == 3
					pz += p.z[child] * m;
#endif
#endif
				}
				// repeat until we are done or child is not ready
			} while ((m >= 0.0) && (missing != 0));
		}

		if (missing == 0) {
			// all children are ready, so store computed information
			m = 1.0 / cm;
			p.x[k] = px * m;
#if DIM > 1
			p.y[k] = py * m;
#if DIM == 3
			p.z[k] = pz * m;
#endif
#endif
			__threadfence();  // make sure data are visible before setting mass
			p.m[k] = cm;
			k += blockDim.x * gridDim.x;  // move on to next cell
		}
	}
}

/* checks interaction list for symmetry */
/*
   removes particle j from particle i's interaction list if particle i is not in
   particles j's interaction list


   awfully slow, not used for the time being
*/
__global__ void symmetrizeInteractions(int *interactions)
{
	register int64_t interactions_index;
	register int64_t interactions_index_2;
	int i, inc, indexP, j;
	int noi;
    int found;
    int k;
    int nod;

    int di[MAX_NUM_INTERACTIONS] = {0, };


	inc = blockDim.x * gridDim.x;
    /* loop over all particles */
	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        nod = 0;
        /* check the interaction list of particle i */
        noi = p.noi[i];
        for (j = 0; j < noi; j++) {
            /* index of interaction partner */
			interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + j;
            indexP = interactions[interactions_index];
            /* check if i is in interaction list of indexP */
            found = FALSE;
            /* loop over all interactions of interaction partner */
            for (k = 0; k < p.noi[indexP]; k++) {
				interactions_index_2 = (int64_t)indexP * MAX_NUM_INTERACTIONS + k;
                if (interactions[interactions_index_2] == i) {
                    found = TRUE;
                    break;
                }
            }
            /* if i was not found in interactions of indexP, delete indexP from interaction list of i */
            if (!found) {
                /*  remember index, that we want to delete */
                di[nod++] = j;
            }
        }
        /* remove deleted partners from interaction list */
        for (k = 0; k < nod; k++) {
			interactions_index_2 = (int64_t)i * MAX_NUM_INTERACTIONS + di[k];			
			interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + noi--;			
            interactions[interactions_index_2] = interactions[interactions_index];
            //interactions[i*MAX_NUM_INTERACTIONS+di[k]] = interactions[i*MAX_NUM_INTERACTIONS+noi--];
        }
        p.noi[i] = noi;
    } /* for loop over all particles */

}


#if VARIABLE_SML && FIXED_NOI
/* search interaction partners with variable smoothing length */
__global__ void knnNeighbourSearch(int *interactions)
{
	register int i, inc, nodeIndex, depth, childNumber, child;
	register double x, y, interactionDistance, dx, dy, r, d;
	register int currentNodeIndex[MAXDEPTH];
	register int currentChildNumber[MAXDEPTH];
	register int numberOfInteractions;
#if DIM == 3
	register double z, dz;
#endif
	inc = blockDim.x * gridDim.x;
    /* loop over all particles */
	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
		x = p.x[i];
		y = p.y[i];
#if DIM == 3
		z = p.z[i];
#endif

        volatile int found = FALSE;
        register int nit = -1;

	    double htmp, htmpold;
        volatile double htmpj;

        htmp = p.h[i];

        /* look for nice sml */
        while (!found) {
            numberOfInteractions = 0;
            nit++;
		    depth = 0;
		    currentNodeIndex[depth] = numNodes - 1;
		    currentChildNumber[depth] = 0;
		    numberOfInteractions = 0;
		    r = radius * 0.5; // because we start with root children
		    interactionDistance = (r + htmp);
    		do {

	    		childNumber = currentChildNumber[depth];
		    	nodeIndex = currentNodeIndex[depth];

			    while (childNumber < numChildren) {
				    child = childList[childListIndex(nodeIndex, childNumber)];
				    childNumber++;
				    if (child != EMPTY && child != i) {
					    dx = x - p.x[child];
					    dy = y - p.y[child];
#if DIM == 3
					    dz = z - p.z[child];
#endif
					    if (child < numParticles) {
						    d = dx*dx + dy*dy;
#if DIM == 3
						    d += dz*dz;
#endif
                            htmpj = p.h[child];

						    if (d < htmp*htmp && d < htmpj*htmpj) {
							    numberOfInteractions++;
						    }
					    } else if (fabs(dx) < interactionDistance && fabs(dy) < interactionDistance
#if DIM == 3
					        		&& fabs(dz) < interactionDistance
#endif
					    ) {
						// put child on stack
						    currentChildNumber[depth] = childNumber;
						    currentNodeIndex[depth] = nodeIndex;
						    depth++;
						    r *= 0.5;
						    interactionDistance = (r + htmp);
					    	if (depth >= MAXDEPTH) {
						    	printf("Error, maxdepth reached! problem in tree during interaction search");
                                assert(depth < MAXDEPTH);
						    }
						    childNumber = 0;
						    nodeIndex = child;
					    }
				    }
			    }
			    depth--;
			    r *= 2.0;
			    interactionDistance = (r + htmp);
		    } while (depth >= 0);

            htmpold = htmp;
//            printf("%d %d %e\n", i, numberOfInteractions, htmp);
            /* stop if we have the desired number of interaction partners \pm TOLERANCE_WANTED_NUMBER_OF_INTERACTIONS */
            if ((nit > MAX_VARIABLE_SML_ITERATIONS || abs(numberOfInteractions - matnoi[p_rhs.materialId[i]]) < TOLERANCE_WANTED_NUMBER_OF_INTERACTIONS ) && numberOfInteractions < MAX_NUM_INTERACTIONS) {
                found = TRUE;
                p.h[i] = htmp;
            } else if (numberOfInteractions >= MAX_NUM_INTERACTIONS) {
                htmpold = htmp;
                if (numberOfInteractions < 1)
                    numberOfInteractions = 1;
                htmp *= 0.5 *  ( 1.0 + pow( (double) matnoi[p_rhs.materialId[i]]/ (double) numberOfInteractions, 1./DIM));
            } else {
                /* lower or raise htmp accordingly */
                if (numberOfInteractions < 1)
                    numberOfInteractions = 1;

                htmpold = htmp;
                htmp *= 0.5 *  ( 1.0 + pow( (double) matnoi[p_rhs.materialId[i]]/ (double) numberOfInteractions, 1./DIM));
            }
#if DEBUG_MISC
            if (htmp < 1e-20) {
                printf("+++ particle: %d it: %d htmp: %e htmpold: %e wanted: %d current: %d mId: %d \n", i, nit,
                        htmp, htmpold, matnoi[p_rhs.materialId[i]], numberOfInteractions, p_rhs.materialId[i]);
            }
#endif

        }
    }

}

#endif


/* search interaction partners for each particle */
/* the smoothing length is changed if MAX_NUM_INTERACTIONS is reached */
__global__ void nearNeighbourSearch_modify_sml(int *interactions)
{
	register int i, inc, nodeIndex, depth, childNumber, child;
	register double x, interactionDistance, dx, r, d;
#if DIM > 1
    register double y, dy;
#endif
	register int currentNodeIndex[MAXDEPTH];
	register int currentChildNumber[MAXDEPTH];
	register int numberOfInteractions;
#if DIM == 3
	register double z, dz;
#endif
	inc = blockDim.x * gridDim.x;

    register int interactions_OK = 0;

	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
		x = p.x[i];
#if DIM > 1
		y = p.y[i];
#if DIM == 3
		z = p.z[i];
#endif
#endif
	    double sml; /* smoothing length of particle */
        volatile double smlj; /* smoothing length of potential interaction partner */



start_interaction_search_for_particle:
		// start at root
		depth = 0;
		currentNodeIndex[depth] = numNodes - 1;
		currentChildNumber[depth] = 0;
		numberOfInteractions = 0;
		r = radius * 0.5; // because we start with root children
        sml = p.h[i];
		interactionDistance = (r + sml);
        // flag for numberOfInteractions < MAX_NUM_INTERACTIONS
        interactions_OK = 0;

		do {
			childNumber = currentChildNumber[depth];
			nodeIndex = currentNodeIndex[depth];
			while (childNumber < numChildren) {
				child = childList[childListIndex(nodeIndex, childNumber)];
				childNumber++;
				if (child != EMPTY && child != i) {
					dx = x - p.x[child];
#if DIM > 1
					dy = y - p.y[child];
#if DIM == 3
					dz = z - p.z[child];
#endif
#endif
					if (child < numParticles) {
						d = dx*dx;
#if DIM > 1
                        d += dy*dy;
#if DIM == 3
						d += dz*dz;
#endif
#endif

                        smlj = p.h[child];

                        // make sure, all interactions are symmetric
						if (d < sml*sml && d < smlj*smlj) {
                            // check if we are still safe with the current numberOfInteractions
                            if (numberOfInteractions < MAX_NUM_INTERACTIONS) {
							    interactions[i * MAX_NUM_INTERACTIONS + numberOfInteractions] = child;
                            }
							numberOfInteractions++;
                        }
					} else if (fabs(dx) < interactionDistance
#if DIM > 1
                            && fabs(dy) < interactionDistance
#if DIM == 3
							&& fabs(dz) < interactionDistance
#endif
#endif
					) {
						// put child on stack
						currentChildNumber[depth] = childNumber;
						currentNodeIndex[depth] = nodeIndex;
						depth++;
						r *= 0.5;
						interactionDistance = (r + sml);
						if (depth >= MAXDEPTH) {
							printf("Error, maxdepth reached!");
                            assert(depth < MAXDEPTH);
						}
						childNumber = 0;
						nodeIndex = child;
					}
				}
			}

			depth--;
			r *= 2.0;
			interactionDistance = (r + sml);
		} while (depth >= 0);

		if (numberOfInteractions >= MAX_NUM_INTERACTIONS) {
            // now, we lower the sml according to the dimension and the ratio
            sml = pow((double) MAX_NUM_INTERACTIONS/(double) numberOfInteractions, 1./DIM) * p.h[i];
            // and remove another 20%
            if (threadIdx.x == 0)
                printf("WARNING: Maximum number of interactions exceeded: %d / %d, lower sml from %.16f to %.16f\n", numberOfInteractions, MAX_NUM_INTERACTIONS, p.h[i], 0.8*sml);
            p.h[i] = 0.8*sml;
            // do this search for particle i again
            goto start_interaction_search_for_particle;
		}
		p.noi[i] = numberOfInteractions;
	}
}





/* search interaction partners for each particle */
__global__ void nearNeighbourSearch(int *interactions)
{
	register int64_t interactions_index;
	register int i, inc, nodeIndex, depth, childNumber, child;
	register double x, interactionDistance, dx, r, d;
#if DIM > 1
    register double y, dy;
#endif
	register int currentNodeIndex[MAXDEPTH];
	register int currentChildNumber[MAXDEPTH];
	register int numberOfInteractions;
#if DIM == 3
	register double z, dz;
#endif


	inc = blockDim.x * gridDim.x;
	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
		x = p.x[i];
#if DIM > 1
		y = p.y[i];
#if DIM == 3
		z = p.z[i];
#endif
#endif
	    double sml; /* smoothing length of particle */
        double smlj; /* smoothing length of potential interaction partner */
		// start at root
		depth = 0;
		currentNodeIndex[depth] = numNodes - 1;
		currentChildNumber[depth] = 0;
		numberOfInteractions = 0;
		r = radius * 0.5; // because we start with root children
        sml = p.h[i];
        p.noi[i] = 0;
		interactionDistance = (r + sml);

		do {

			childNumber = currentChildNumber[depth];
			nodeIndex = currentNodeIndex[depth];

			while (childNumber < numChildren) {
#if DEBUG_DEVEL			
				register int childListIndex_int = childListIndex(nodeIndex, childNumber);
				assert(childListIndex_int > 0);
#endif 
				child = childList[childListIndex(nodeIndex, childNumber)];
#if DEBUG_DEVEL				
				if (child < LOCKED) {
					printf("child %d (depth %d) is broken\n", child, depth);
					assert(child > 0);
				}
#endif
				childNumber++;
				if (child != EMPTY && child != i) {
					dx = x - p.x[child];
#if DIM > 1
					dy = y - p.y[child];
#if DIM == 3
					dz = z - p.z[child];
#endif
#endif


					if (child < numParticles) {
                        if (p_rhs.materialId[child] == EOS_TYPE_IGNORE) {
                            continue;
                        }
						d = dx*dx;
#if DIM > 1
                        d += dy*dy;
#if DIM == 3
						d += dz*dz;
#endif
#endif

                        smlj = p.h[child];

						if (d < sml*sml && d < smlj*smlj) {
							interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + numberOfInteractions;
							// interactions[i * MAX_NUM_INTERACTIONS + numberOfInteractions] = child;
							interactions[interactions_index] = child;
							numberOfInteractions++;
#if TOO_MANY_INTERACTIONS_KILL_PARTICLE
                            if (numberOfInteractions >= MAX_NUM_INTERACTIONS) {
                                printf("setting the smoothing length for particle %d to 0!\n", i);
                                p.h[i] = 0.0;
                                p.noi[i] = 0;
                                sml = 0.0;
                                interactionDistance = 0.0;
                                p_rhs.materialId[i] = EOS_TYPE_IGNORE;
                                // continue with next particle by setting depth to -1
                                // cms 2018-01-19
                                depth = -1;
                                break;
                            }
#endif
						}
					} else if (fabs(dx) < interactionDistance
#if DIM > 1
                            && fabs(dy) < interactionDistance
#if DIM == 3
							&& fabs(dz) < interactionDistance
#endif
#endif
					) {
						// put child on stack
						currentChildNumber[depth] = childNumber;
						currentNodeIndex[depth] = nodeIndex;
						depth++;
						r *= 0.5;
						interactionDistance = (r + sml);
						if (depth >= MAXDEPTH) {
							printf("Error, maxdepth reached!");
                            assert(depth < MAXDEPTH);
						}
						childNumber = 0;
						nodeIndex = child;
					}
				}
			}

			depth--;
			r *= 2.0;
			interactionDistance = (r + sml);
		} while (depth >= 0);

		if (numberOfInteractions >= MAX_NUM_INTERACTIONS) {
			//printf("ERROR: Maximum number of interactions exceeded: %d / %d\n", numberOfInteractions, MAX_NUM_INTERACTIONS);
#if !TOO_MANY_INTERACTIONS_KILL_PARTICLE
            assert(numberOfInteractions < MAX_NUM_INTERACTIONS);
#endif
            /*
			for (child = 0; child < MAX_NUM_INTERACTIONS; child++) {
				printf("(thread %d): %d - %d\n", threadIdx.x, i, interactions[i*MAX_NUM_INTERACTIONS+child]);
			} */
		}
		p.noi[i] = numberOfInteractions;
	}
}

#if VARIABLE_SML
// checks if the smoothing length is too large or too small
__global__ void check_sml_boundary(void)
{
    int i, inc;
    int matId, d, e;
    double smlmin, smlmax;
    inc = blockDim.x * gridDim.x;
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        matId = p_rhs.materialId[i];
        smlmin = p_rhs.h0[i] * mat_f_sml_min[matId];
        smlmax = p_rhs.h0[i] * mat_f_sml_max[matId];
        if (p.h[i] < smlmin) {
            p.h[i] = smlmin;
#if INTEGRATE_SML
            p.dhdt[i] = 0.0;
#endif
        } else if (p.h[i] > smlmax) {
            p.h[i] = smlmax;
#if INTEGRATE_SML
            p.dhdt[i] = 0.0;
#endif
        }
    }
}
#endif

__global__ void computationalDomain(
		double *minxPerBlock, double *maxxPerBlock
#if DIM > 1
		, double *minyPerBlock, double *maxyPerBlock
#endif
#if DIM == 3
		, double *minzPerBlock, double *maxzPerBlock
#endif
		) {
	register int i, j, k, m;
	__shared__ volatile double sharedMinX[NUM_THREADS_COMPUTATIONAL_DOMAIN];
	__shared__ volatile double sharedMaxX[NUM_THREADS_COMPUTATIONAL_DOMAIN];
#if DIM > 1
	__shared__ volatile double sharedMinY[NUM_THREADS_COMPUTATIONAL_DOMAIN];
	__shared__ volatile double sharedMaxY[NUM_THREADS_COMPUTATIONAL_DOMAIN];
    register double localMinY, localMaxY;
#endif
	register double localMinX, localMaxX;
#if DIM == 3
	__shared__ volatile double sharedMinZ[NUM_THREADS_COMPUTATIONAL_DOMAIN];
	__shared__ volatile double sharedMaxZ[NUM_THREADS_COMPUTATIONAL_DOMAIN];
	register double localMinZ, localMaxZ;
#endif
	// init with valid data
	localMinX = p.x[0];
	localMaxX = p.x[0];
#if DIM > 1
	localMinY = p.y[0];
	localMaxY = p.y[0];
#if DIM == 3
	localMinZ = p.z[0];
	localMaxZ = p.z[0];
#endif
#endif
	// printf("DEBUG: threadId.x: %d, blockIdx.x: %d, blockDim.x: %d, gridDim.x: %d, threadIdx.x + blockIdx.x * blockDim.x: %d, numParticles: %d\n", threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, threadIdx.x + blockIdx.x * blockDim.x, numParticles);
	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
		// find minimum and maximum coordinates
		localMinX = min(localMinX, p.x[i]);
		localMaxX = max(localMaxX, p.x[i]);
#if DIM > 1
		localMinY = min(localMinY, p.y[i]);
		localMaxY = max(localMaxY, p.y[i]);
#if DIM == 3
		localMinZ = min(localMinZ, p.z[i]);
		localMaxZ = max(localMaxZ, p.z[i]);
#endif
#endif
	}
	i = threadIdx.x;
	sharedMinX[i] = localMinX;
	sharedMaxX[i] = localMaxX;
#if DIM > 1
	sharedMinY[i] = localMinY;
	sharedMaxY[i] = localMaxY;
#if DIM == 3
	sharedMinZ[i] = localMinZ;
	sharedMaxZ[i] = localMaxZ;
#endif
#endif
	// reduction
	for (j = NUM_THREADS_COMPUTATIONAL_DOMAIN / 2; j > 0; j /= 2) {
		__syncthreads();
		if (i < j) {
			k = i + j;
			sharedMinX[i] = localMinX = min(localMinX, sharedMinX[k]);
			sharedMaxX[i] = localMaxX = max(localMaxX, sharedMaxX[k]);
#if DIM > 1
			sharedMinY[i] = localMinY = min(localMinY, sharedMinY[k]);
			sharedMaxY[i] = localMaxY = max(localMaxY, sharedMaxY[k]);
#if DIM == 3
			sharedMinZ[i] = localMinZ = min(localMinZ, sharedMinZ[k]);
			sharedMaxZ[i] = localMaxZ = max(localMaxZ, sharedMaxZ[k]);
#endif
#endif
		}
	}
	// first thread writes block result to global memory
	if (i == 0) {
		k = blockIdx.x;
		minxPerBlock[k] = localMinX;
		maxxPerBlock[k] = localMaxX;
#if DIM > 1
		minyPerBlock[k] = localMinY;
		maxyPerBlock[k] = localMaxY;
#if DIM == 3
		minzPerBlock[k] = localMinZ;
		maxzPerBlock[k] = localMaxZ;
#endif
#endif
		m = gridDim.x - 1;
		if (m == atomicInc((unsigned int *) &blockCount, m)) {
			// last block, so combine all block results
			for (j = 0; j <= m; j++) {
				localMinX = min(localMinX, minxPerBlock[j]);
				localMaxX = max(localMaxX, maxxPerBlock[j]);
#if DIM > 1
				localMinY = min(localMinY, minyPerBlock[j]);
				localMaxY = max(localMaxY, maxyPerBlock[j]);
#if DIM == 3
				localMinZ = min(localMinZ, minzPerBlock[j]);
				localMaxZ = max(localMaxZ, maxzPerBlock[j]);
#endif
#endif
			}
			minx = localMinX;
			maxx = localMaxX;
#if DIM > 1
			miny = localMinY;
			maxy = localMaxY;
#if DIM == 3
			minz = localMinZ;
			maxz = localMaxZ;
#endif
#endif
			// create root node
			k = numNodes - 1;
			radius = localMaxX - localMinX;
#if DIM > 1
			radius = max(localMaxX - localMinX, localMaxY - localMinY);
#if DIM == 3
			radius = max(radius, localMaxZ - localMinZ);
#endif
#endif
			radius *= 0.5;
			// printf("DEBUG: Computational domain: minx: %e, maxx: %e, radius: %e\n", localMinX, localMaxX, radius);
			p.x[k] = 0.5 * (localMaxX + localMinX);
#if DIM > 1
			p.y[k] = 0.5 * (localMaxY + localMinY);
#if DIM == 3
			p.z[k] = 0.5 * (localMaxZ + localMinZ);
#endif
#endif
			p.m[k] = EMPTY;
			for (i = 0; i < numChildren; i++) childList[childListIndex(k, i)] = EMPTY;
			maxNodeIndex = k;
			// reset block count
			blockCount = 0;
		}
	}
}

#if SML_CORRECTION
/* redo NeighbourSearch for particular particle only: search for interaction partners */
__device__ void redo_NeighbourSearch(int particle_id, int *interactions)
{
	register int64_t interactions_index;
	register int i, inc, nodeIndex, depth, childNumber, child;
	register double x, y, interactionDistance, dx, dy, r, d;
	register int currentNodeIndex[MAXDEPTH];
	register int currentChildNumber[MAXDEPTH];
	register int numberOfInteractions;
#if DIM == 3
	register double z, dz;
#endif
    i = particle_id;
	x = p.x[i];
	y = p.y[i];
#if DIM == 3
	z = p.z[i];
#endif
    //printf("1) sml_new > h: noi: %d\n", p.noi[i]);

	double sml; /* smoothing length of particle */
    double smlj; /* smoothing length of potential interaction partner */
	// start at root
	depth = 0;
	currentNodeIndex[depth] = numNodes - 1;
	currentChildNumber[depth] = 0;
	numberOfInteractions = 0;
	r = radius * 0.5; // because we start with root children
    sml = p.h[i];
    p.noi[i] = 0;
	interactionDistance = (r + sml);

	do {
		childNumber = currentChildNumber[depth];
		nodeIndex = currentNodeIndex[depth];
		while (childNumber < numChildren) {
			child = childList[childListIndex(nodeIndex, childNumber)];
			childNumber++;
			if (child != EMPTY && child != i) {
				dx = x - p.x[child];
#if DIM > 1
				dy = y - p.y[child];
#if DIM == 3
				dz = z - p.z[child];
#endif
#endif

				if (child < numParticles) {
                    if (p_rhs.materialId[child] == EOS_TYPE_IGNORE) {
                        continue;
                    }
					d = dx*dx;
#if DIM > 1
                    d += dy*dy;
#if DIM == 3
					d += dz*dz;
#endif
#endif
                    smlj = p.h[child];

					if (d < sml*sml && d < smlj*smlj) {
						interactions_index = (int64_t)i * MAX_NUM_INTERACTIONS + numberOfInteractions;
						interactions[interactions_index] = child;
						// interactions[i * MAX_NUM_INTERACTIONS + numberOfInteractions] = child;
						numberOfInteractions++;
#if TOO_MANY_INTERACTIONS_KILL_PARTICLE
                        if (numberOfInteractions >= MAX_NUM_INTERACTIONS) {
                            printf("setting the smoothing length for particle %d to 0!\n", i);
                            p.h[i] = 0.0;
                            p.noi[i] = 0;
                            sml = 0.0;
                            interactionDistance = 0.0;
                            p_rhs.materialId[i] = EOS_TYPE_IGNORE;
                            // continue with next particle by setting depth to -1
                            // cms 2018-01-19
                            depth = -1;
                            break;
                        }
#endif
					}
				} else if (fabs(dx) < interactionDistance
#if DIM > 1
                        && fabs(dy) < interactionDistance
#if DIM == 3
						&& fabs(dz) < interactionDistance
#endif
#endif
				) {
					// put child on stack
					currentChildNumber[depth] = childNumber;
					currentNodeIndex[depth] = nodeIndex;
					depth++;
					r *= 0.5;
					interactionDistance = (r + sml);
					if (depth >= MAXDEPTH) {
						printf("wtf, maxdepth reached!");
                        assert(depth < MAXDEPTH);	
					}
						childNumber = 0;
						nodeIndex = child;
				}
			}
		}

		depth--;
		r *= 2.0;
		interactionDistance = (r + sml);
	} while (depth >= 0);

	if (numberOfInteractions >= MAX_NUM_INTERACTIONS) {
		printf("ERROR: Maximum number of interactions exceeded: %d / %d\n", numberOfInteractions, MAX_NUM_INTERACTIONS);
#if !TOO_MANY_INTERACTIONS_KILL_PARTICLE
       // assert(numberOfInteractions < MAX_NUM_INTERACTIONS);
#endif
	}
	p.noi[i] = numberOfInteractions;
}
#endif //SML_CORRECTION
