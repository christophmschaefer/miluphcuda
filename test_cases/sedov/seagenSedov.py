import seagen

import seagen
import numpy as np
import h5py

def cubicSpline(dx_vec, sml):

    dWdx = np.zeros(3)
    r = 0
    for d in range(3):
        r += dx_vec[d] * dx_vec[d]
        dWdx[d] = 0

    r = np.sqrt(r)
    dWdr = 0
    W = 0
    q = r/sml

    f = 8./np.pi * 1./(sml * sml * sml);

    if q > 1:
        W = 0
        dWdr = 0.0
    elif q > 0.5:
        W = 2. * f * (1.-q) * (1.-q) * (1-q)
        dWdr = -6. * f * 1./sml * (1.-q) * (1.-q)
    elif q <= 0.5:
        W = f * (6. * q * q * q - 6. * q * q + 1.)
        dWdr = 6. * f/sml * (3 * q * q - 2 * q)
    for d in range(3):
        dWdx[d] = dWdr/r * dx_vec[d]

    return W, dWdr, dWdx


if __name__ == '__main__':

    """
    Create particle distribution using seagen
    """
    N = 100000
    radius = 0.5
    radius_sedov = radius * 0.06
    radii = np.arange(0.01, radius, 0.01)
    densities = np.ones(len(radii))

    particles = seagen.GenSphere(N, radii, densities)


    """
    Postprocess particle distribution
    """
    # masses
    m = particles.m

    # positions
    x = particles.x
    y = particles.y
    z = particles.z

    # velocities
    vx = np.zeros(len(x))
    vy = np.zeros(len(x))
    vz = np.zeros(len(x))

    r = np.sqrt(x**2 + y**2 + z**2)

    u = np.zeros(len(x))

    n_sedov = 0
    verify = 0

    for i in range(len(x)):
        if r[i] < radius_sedov:
            W, dWdr, dWdx = cubicSpline([x[i], y[i], z[i]], radius_sedov)
            verify += W * m[i]
            n_sedov += 1
            u[i] = W

    print("u_mean = {}".format(u.mean()))
    u_max = u.max()
    u_ground = u_max * 1e-6
    for i in range(len(x)):
        if u[i] == 0:
            u[i] = u_ground

    print("n_sedov: {}".format(n_sedov))
    print("verify: {}".format(verify))
    print("u_max = {} | u_ground = {} | u_mean = {}".format(u_max, u_ground, u.mean()))


    """
    Write to h5 file
    """
    f = h5py.File("sedov_seagen_{}.h5".format(len(x)), "w")

    pos = np.array([[x[i], y[i], z[i]] for i in range(len(x))])
    vel = np.array([[float(vx[i]), float(vy[i]), float(vz[i])] for i in range(len(x))])
    materialId = np.array([int(0) for i in range(len(x))])

    f.create_dataset("x", data=pos)
    f.create_dataset("v", data=vel)
    f.create_dataset("m", data=m)
    f.create_dataset("materialId", data=materialId)
    f.create_dataset("u", data=u)

    f.close()
