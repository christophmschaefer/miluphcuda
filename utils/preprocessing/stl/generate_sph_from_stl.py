#!/usr/bin/env python

"""
Generate SPH distribution from STL 3D data
author: Christoph Schäfer, ch.schaefer@uni-tuebingen.de


to fasten up the processing you might want to install pycuda 
see https://documen.tician.de/pycuda for instructions


last changes: 2017-06-30
"""

import sys
import os
import numpy as np
import stl


try:
    from mpl_toolkits import mplot3d
    from matplotlib import pyplot
    use_plot = True
    print("Found matplotlib, enabling plotting.")
except:
    print("no matplotlib found, disabling plotting.")
    use_plot = False


try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    use_gpu = True
    print("Found cuda support, using gpu. Make sure nvcc is in your PATH.")
except:
    print("no cuda support found, disabling gpu usage.")
    use_gpu = False


sqrt3 = np.sqrt(3)
sqrt6 = np.sqrt(6)

# the factor from HCP
fact = 2*sqrt6/3


def locations(i, j, k):
    x = 2*i  + (j+k)%2
    y = sqrt3*(j + 1./3*(k%2))
    z = fact*k
    return x, y, z




if len(sys.argv) != 4:
    print("Generate SPH particle locations from STL geometrical data.")
    print("e.g. get your STL data from https://nasa3d.arc.nasa.gov and generate point clouds for SPH simulations.")
    print("usage:")
    print("%s <file>.stl number_of_particles <c|hcp> " % sys.argv[0])
    print("where <file>.stl is a STL file and number_of_particles an integer, c is for a cubic grid, hcp for a hexagonal.")
    print("output file is foo.0000 and contains the locations of the particles in column-style x y z.")
    print("Note: the script takes quite a while unless cuda is used.")
    sys.exit(0)



iterations = 0
nowp = int(sys.argv[2])

if nowp < 0:
    print("Error: number of particles should be > 0.")
    sys.exit(1)


gridstyle = sys.argv[3]
if gridstyle == 'c':
    print("using cubic grid")
elif gridstyle == 'hcp':
    print("using hexagonal grid")
else:
    print("Error. no such gridstyle '%s'." % gridstyle)
    sys.exit(1)

# read stl file
try:
    stl_fn = sys.argv[1]
    mesh = stl.mesh.Mesh.from_file(stl_fn)
except:
    print("Error reading %s" % stl_fn)
    sys.exit(1)

# get the normals and the geometry of the mesh
normals = mesh.normals
# number of normals in file
nnormals = np.shape(normals)[0]
# volume
volume, cog, inertia = mesh.get_mass_properties()

# calculate midpoints of meshes
xs = np.zeros(nnormals)
ys = np.zeros(nnormals)
zs = np.zeros(nnormals)
for n in range(nnormals):
    xs[n] = mesh.points[n,0] + mesh.points[n,3] + mesh.points[n,6]
    ys[n] = mesh.points[n,1] + mesh.points[n,4] + mesh.points[n,7]
    zs[n] = mesh.points[n,2] + mesh.points[n,5] + mesh.points[n,8]
xs *= 1./3
ys *= 1./3
zs *= 1./3

minx = np.min(xs)
maxx = np.max(xs)
miny = np.min(ys)
maxy = np.max(ys)
minz = np.min(zs)
maxz = np.max(zs)

print("Minima and maxima from STL geometry.")
print("x: %e to %e" % (minx, maxx))
print("y: %e to %e" % (miny, maxy))
print("z: %e to %e" % (minz, maxz))

mmax = np.max([maxx, maxy, maxz])
mmin = np.min([minx, miny, minz])

if gridstyle == 'hcp':
    dr = (mmax - mmin)/(nowp**(1./3)) 
    NI = np.ceil((mmax - mmin)/(dr)).astype('int') + 1
    NJ = NI
    NK = NJ
    N = NI*NJ*NK
    print("max and min: %e to %e" % (mmin, mmax))
    while nowp != N and iterations < 10:
        iterations += 1
        dr *= (N/nowp)**(1./3)
        scale = dr*fact
        NI = np.ceil((mmax - mmin)/(scale)).astype('int') + 1
        NJ = NI
        NK = NJ
        N = NI*NJ*NK
    print("Starting with", NI*NJ*NK, "particles,", NI, "in each direction", "delta is", dr*fact)

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    c = 0
    for i in range(NI):
        for j in range(NJ):
            for k in range(NK):
                x[c], y[c], z[c] = locations(i, j, k)
                c += 1

    x *= dr
    y *= dr
    z *= dr
    offset = mmin
    x += offset
    y += offset
    z += offset

elif gridstyle == 'c':
    dr = (mmax - mmin)/(nowp**(1./3))
    NI = np.ceil((mmax - mmin)/dr).astype('int') + 1
    N = NI**3
    print("max and min: %e to %e" % (mmin, mmax))
    while nowp != N and iterations < 10:
        iterations += 1
        dr *= (N/nowp)**(1./3)
        NI = np.ceil((mmax - mmin)/(dr)).astype('int') + 1
        N = NI**3
    print("Starting", N, "particles, delta is", dr)

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    c = 0
    for i in range(NI):
        for j in range(NI):
            for k in range(NI):
                x[c], y[c], z[c] = i, j, k
                c += 1
    x *= dr
    y *= dr
    z *= dr
    offset = mmin
    x += offset
    y += offset
    z += offset



kernel_code = """
#include <stdio.h>
__global__ void choose_particles(double *x, double *y, double *z, 
    double *xs, double *ys, double *zs, double *nx, double *ny,
    double *nz, long *take_me, long *nop, long *nom)
{

    int i, n, inc;
    int check_me;

    double distance;
    double d;
    double dotproduct;
    double xv, yv, zv;

    inc = blockDim.x * gridDim.x;
    //printf("%d \\n", threadIdx);

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < *nop; i += inc) {
        distance = 1e30;
        dotproduct = 0;
        check_me = -1;
        d = 0;
        for (n = 0; n < *nom; n++) {
            d = (xs[n]-x[i])*(xs[n]-x[i]) + (ys[n]-y[i])*(ys[n]-y[i])
                + (zs[n]-z[i])*(zs[n]-z[i]);
            if (d < distance) {
                distance = d;
                check_me = n;
            }
        }
        xv = xs[check_me] - x[i];
        yv = ys[check_me] - y[i];
        zv = zs[check_me] - z[i];
        dotproduct = xv*nx[check_me] + yv*ny[check_me] + zv*nz[check_me];
        if (dotproduct > 0) {
            take_me[i] = i;
           // printf("+++ %e %e %e\\n", x[i], y[i], z[i]);     
        } else {
            take_me[i] = -1;
        }
    }
}
"""





if use_gpu:
    print("transferring data to the gpu")
    mod = SourceModule(kernel_code)
    function = mod.get_function('choose_particles')
    # allocate for index array
    take_me = np.arange(0, N)
    take_me_results = np.empty_like(take_me)
    take_me_gpu = cuda.mem_alloc(take_me.nbytes)
    x_gpu = cuda.mem_alloc(x.nbytes)
    y_gpu = cuda.mem_alloc(y.nbytes)
    z_gpu = cuda.mem_alloc(z.nbytes)
    xs_gpu = cuda.mem_alloc(xs.nbytes)
    ys_gpu = cuda.mem_alloc(ys.nbytes)
    zs_gpu = cuda.mem_alloc(zs.nbytes)
    # the components of the normals
    nx = np.empty_like(xs)
    ny = np.empty_like(xs)
    nz = np.empty_like(xs)
    # we need contiguous memory, hence, copy everthing smoothly...
    N = len(x)
    M = len(nx)
    N = np.array(N, dtype='int')
    M = np.array(M, dtype='int')
    for n in range(0, len(nx)):
        nx[n] = normals[n,0]
        ny[n] = normals[n,1]
        nz[n] = normals[n,2]
    nx_gpu = cuda.mem_alloc(nx.nbytes)
    ny_gpu = cuda.mem_alloc(ny.nbytes)
    nz_gpu = cuda.mem_alloc(nz.nbytes)
    nop_gpu = cuda.mem_alloc(N.nbytes)
    nom_gpu = cuda.mem_alloc(M.nbytes)

    cuda.memcpy_htod(nop_gpu, N)
    cuda.memcpy_htod(nom_gpu, M)
    
    # now copy everything on the gpu
    cuda.memcpy_htod(x_gpu, x)
    cuda.memcpy_htod(y_gpu, y)
    cuda.memcpy_htod(z_gpu, z)
    cuda.memcpy_htod(xs_gpu, xs)
    cuda.memcpy_htod(ys_gpu, ys)
    cuda.memcpy_htod(zs_gpu, zs)
    cuda.memcpy_htod(nx_gpu, nx)
    cuda.memcpy_htod(ny_gpu, ny)
    cuda.memcpy_htod(nz_gpu, nz)
    # call the kernel that does the job
    function(x_gpu, y_gpu, z_gpu, xs_gpu, ys_gpu, zs_gpu, nx_gpu, ny_gpu, nz_gpu, take_me_gpu, nop_gpu, nom_gpu, block=(256,1,1))
    print("kernel called")
    cuda.memcpy_dtoh(take_me, take_me_gpu)
    take_me = take_me[take_me > -1]
    take_me = list(take_me)
else:
    # now check for each particle if it's inside the mesh volume or not
    take_me = []
    for c in range(N):
        yeah_me = 1
        print("Processing particle no.", c, (c+1)/N*100, '% done                 ',  end='\r', flush=True)
        # find closest mesh
        distance = 2*(mmax - mmin)
        for n in range(nnormals):
            d = (xs[n] - x[c])**2 + (ys[n] - y[c])**2 + (zs[n] - z[c])**2
            if d < distance:
                distance = d
                check_mesh = n
        n = check_mesh
        xv = xs[n] - x[c]
        yv = ys[n] - y[c]
        zv = zs[n] - z[c]
        dotproduct = xv*normals[n,0] + yv*normals[n,1] + zv*normals[n,2]
        if (dotproduct > 0):
            take_me.append(c)
# end of if use_gpu
        


x = np.take(x, take_me)
y = np.take(y, take_me)
z = np.take(z, take_me)

fnumber = len(x)
print("Finally", fnumber, "particles chosen.")
print("Total volume from STL data:", volume)
print("-> particle volume:", volume/fnumber)


np.savetxt('foo.0000', np.transpose([x,y,z]), delimiter=' ', newline=os.linesep)
print("foo.0000 saved with coordinates.")

if use_plot:
    print("Now plotting.")
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)
#    axes.add_collection3d(mplot3d.art3d.Line3DCollection(mesh.vectors, alpha=0.1))
    axes.scatter(x,y,z,c='r')
    # Auto scale to the mesh size
    scale = mesh.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    # Show the plot to the screen
    pyplot.show()

print("End.")


