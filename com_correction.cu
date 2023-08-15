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
 * You should have received a copy of the GNU General Public License;
 * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "com_correction.h"
#include "miluph.h"
#include "memory_handling.h"
#include "timeintegration.h"

#if MOVING_COM_CORRECTION

__global__ void COMcorrection()
{
    register int i, inc, n;
    inc = blockDim.x * gridDim.x;
    double total_mass;
    double xcm, ycm;
    double vxcm, vycm;
#if DIM > 2
    double zcm, vzcm;
#endif

    total_mass = 0.0;
    xcm = 0.0;
    ycm = 0.0;
    vxcm = 0.0;
    vycm = 0.0;
#if DIM > 2
    zcm = 0.0;
    vzcm = 0.0;
#endif
    
    //Loop over ptmasses to get new COM coordinates
//    for (n = threadIdx.x + blockIdx.x * blockDim.x; n < numPointmasses; n += inc) {

    for (n = 0; n < numPointmasses; n++) {
        total_mass += pointmass.m[n];
        xcm += pointmass.m[n] * pointmass.x[n];
        ycm += pointmass.m[n] * pointmass.y[n];

        vxcm += pointmass.m[n] * pointmass.vx[n];
        vycm += pointmass.m[n] * pointmass.vy[n];
#if DIM > 2
        zcm += pointmass.m[n] * pointmass.z[n];
        vzcm += pointmass.m[n] * pointmass.vz[n];
#endif
    }
    
    xcm /= total_mass;
    ycm /= total_mass;

    vxcm /= total_mass;
    vycm /= total_mass;
#if DIM > 2
    zcm /= total_mass;
    vzcm /= total_mass;
#endif
/*
    if (param.verbose) {
        printf("The total mass is %.5f \n", total_mass);
        printf("The COM is now @ (%.5f, %.5f)", xcm, ycm);
        printf("\t moving with velocity (%.5f, %.5f)", vxcm, vycm); 
    }
*/

    printf("The total mass is %.25f \n", total_mass);
    printf("The COM is now @ (%.15f, %.15f) \n", xcm, ycm);
    printf("Moving with velocity (%.15f, %.15f) \n", vxcm, vycm);
 
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
        p.x[i] -= xcm;
        p.y[i] -= ycm;

        p.vx[i] -= vxcm;
        p.vy[i] -= vycm;
#if DIM > 2
        p.z[i] -= zcm;
        p.vz[i] -= vzcm;
#endif
    }
}



#endif
