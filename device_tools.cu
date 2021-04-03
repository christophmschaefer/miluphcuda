/**
 * @author      Daniel Thun and Christoph Schaefer
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

#include <stdio.h>
#include "device_tools.h"

/* 
        device informations
        authors: Daniel Thun and Christoph Schaefer
        mainly taken from cuda samples
*/


/* ********************************************************************* */
inline int _ConvertSMVer2Cores(int major, int minor)
/*!
 *  Helper function to calculate the number of CUDA core.
 *  Taken from cuda_samples/common/inc/helper_cuda.h
 *********************************************************************** */
{
    /* Defines for GPU Architecture types (using the SM version to determine the # of cores per SM */
    typedef struct {
        int SM; /* 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version */
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, /* Tesla Generation (SM 1.0) G80 class */
        { 0x11,  8 }, /* Tesla Generation (SM 1.1) G8x class */
        { 0x12,  8 }, /* Tesla Generation (SM 1.2) G9x class */
        { 0x13,  8 }, /* Tesla Generation (SM 1.3) GT200 class */
        { 0x20, 32 }, /* Fermi Generation (SM 2.0) GF100 class */
        { 0x21, 48 }, /* Fermi Generation (SM 2.1) GF10x class */
        { 0x30, 192}, /* Kepler Generation (SM 3.0) GK10x class */
        { 0x32, 192}, /* Kepler Generation (SM 3.2) GK10x class */
        { 0x35, 192}, /* Kepler Generation (SM 3.5) GK11x class */
        { 0x37, 192}, /* Kepler Generation (SM 3.7) GK21x class */
        { 0x50, 128}, /* Maxwell Generation (SM 5.0) GM10x class */
        { 0x52, 128},
        { 0x53, 128},
        { 0x60,  64},
        { 0x61, 128},
        { 0x62, 128},
        { 0x70,  64},
        { 0x72,  64},
        { 0x75,  64},
        { 0x80,  64},
        { 0x86, 128},
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }

    /* If we don't find the values, we default use the previous one to run properly */
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n",
            major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}



/* ********************************************************************* */
void printfDeviceInformation(void)
/*!
 *  printfs some basic information about detected CUDA devices. 
 *  Taken from cuda samples/1_Utilities/deviceQuery
 *  
 *********************************************************************** */
{
    int i, device_count, driverVersion = 0, runtimeVersion = 0;
    struct cudaDeviceProp prop;

    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        printf("\nNo device(s) that support CUDA found!\n");
        exit(1);
    }

    for (i = 0; i < device_count; i++) {   
      //  cudaSetDevice(i);
        cudaGetDeviceProperties(&prop, i);
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        printf("\nGeneral Information for Device %d -- %s\n\n", i, prop.name);
        printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
        printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("  Compute capability:                            %d.%d\n\n", prop.major, prop.minor);

        printf("  Multiprocessors:                               %d\n", prop.multiProcessorCount);
        printf("  CUDA Cores / Multiprocessor:                   %d\n", _ConvertSMVer2Cores(prop.major, prop.minor));
        printf("  Total amount of CUDA Cores:                    %d\n", _ConvertSMVer2Cores(prop.major, prop.minor)*prop.multiProcessorCount);
        printf("  GPU clock rate:                                %0.f MHz\n\n", prop.clockRate * 1e-3f);

#if CUDART_VERSION >= 5000
        /* This is supported in CUDA 5.0 (runtime API device properties) */
        printf("  Memory Clock rate:                             %.0f Mhz\n", prop.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",   prop.memoryBusWidth);

        if (prop.l2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n\n", prop.l2CacheSize);
        }
#endif

        printf("  Total amount of global memory:                 %.0f MBytes\n", (float)prop.totalGlobalMem/1048576.0f);
        printf("  Total amount of constant memory:               %lu bytes\n", prop.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", prop.sharedMemPerBlock);

        /* if prop.major >= 3 set shared memory bank size to 8 byte */
        if (prop.major >= 3) {
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
            printf("  Shared memory bank size                        %d bytes\n", 8);
        }
        else {
            printf("  Shared memory bank size                        %d bytes\n", 4);
        }

        printf("  Total number of registers available per block: %d\n", prop.regsPerBlock);
        printf("  Warp size:                                     %d\n\n", prop.warpSize);

        printf("  Maximum number of threads per multiprocessor:  %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", prop.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        printf("  Run time limit on kernels:                     %s\n", prop.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", prop.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", prop.canMapHostMemory ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n\n", prop.ECCEnabled ? "Enabled" : "Disabled");
    }
}
