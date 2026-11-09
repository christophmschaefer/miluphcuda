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

#ifndef _TREE_H
#define _TREE_H
#include "parameter.h"

// support radius of the pair kernel W_ij (see AVERAGE_KERNELS in parameter.h and the pair loops in
// internal_forces.cu): particles i and j interact if |r_ij| < SML_PAIR_SUPPORT(h_i, h_j). Symmetric in i and j,
// and monotonic in both arguments, which the neighbour search relies on.
#if AVERAGE_KERNELS
// W_ij = (W(h_i) + W(h_j))/2
# define SML_PAIR_SUPPORT(hi, hj) (fmax((hi), (hj)))
#else
// W_ij = W((h_i + h_j)/2)
# define SML_PAIR_SUPPORT(hi, hj) (0.5*((hi) + (hj)))
#endif


extern __device__ int treeMaxDepth;


__global__ void buildTree();
__global__ void getTreeDepth(int *treeDepthPerBlock);
__global__ void measureTreeChange(int *movingparticlesPerBlock);

__global__ void calculateCentersOfMass();

__global__ void setEmptyMassForInnerNodes(void);

__global__ void calculateNodeSmlMax(void);

__global__ void nearNeighbourSearch(int *interactions);
__global__ void nearNeighbourSearch_modify_sml(int *interactions);

__global__ void knnNeighbourSearch(int *interactions);


__global__ void symmetrizeInteractions(int *interactions);

__global__ void check_sml_boundary(void);

__device__ void redo_NeighbourSearch(int particle_id, int *interactions);


__global__ void computationalDomain(
		double *minxPerBlock, double *maxxPerBlock
#if DIM > 1
		, double *minyPerBlock, double *maxyPerBlock
#endif
#if DIM == 3
		, double *minzPerBlock, double *maxzPerBlock
#endif
		);
#endif
