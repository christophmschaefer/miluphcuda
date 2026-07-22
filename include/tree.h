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


extern __device__ int treeMaxDepth;


__global__ void buildTree();
__global__ void getTreeDepth(int *treeDepthPerBlock);
__global__ void measureTreeChange(int *movingparticlesPerBlock);

__global__ void calculateCentersOfMass();

__global__ void setEmptyMassForInnerNodes(void);

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
