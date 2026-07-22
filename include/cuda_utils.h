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
#ifndef om_cuda_utils_
#define om_cuda_utils_

#include <stdio.h>
#include <stdlib.h>

#ifndef NDEBUG
#define cudaVerify(x) do {                                               \
    cudaError_t __cu_result = x;                                         \
    if (__cu_result!=cudaSuccess) {                                      \
      fprintf(stderr,"%s:%i: error: cuda function call failed:\n"        \
              "  %s;\nmessage: %s\n",                                    \
              __FILE__,__LINE__,#x,cudaGetErrorString(__cu_result));     \
      exit(1);                                                           \
    }                                                                    \
  } while(0)
#define cudaVerifyKernel(x) do {                                         \
    x;                                                                   \
    cudaError_t __cu_result = cudaGetLastError();                        \
    if (__cu_result!=cudaSuccess) {                                      \
      fprintf(stderr,"%s:%i: error: cuda function call failed:\n"        \
              "  %s;\nmessage: %s\n",                                    \
              __FILE__,__LINE__,#x,cudaGetErrorString(__cu_result));     \
      exit(1);                                                           \
    }                                                                    \
  } while(0)
#else
#define cudaVerify(x) do {                                               \
    x;                                                                   \
  } while(0)
#define cudaVerifyKernel(x) do {                                         \
    x;                                                                   \
  } while(0)
#endif

#endif
