#!/usr/bin/env python3

"""
Creates initial conditions for tensile rod test case, see Gray (2001) and Schaefer (2005).

authors: Christoph Schäfer, Christoph Burger
last updated: 11/May/2021
"""


import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import argparse
except Exception as e:
    print("ERROR! Cannot properly import modules. Exiting ...")
    print(str(type(e).__name__))
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)


parser = argparse.ArgumentParser(description = "Creates initial conditions for tensile rod test case, see Gray (2001) and Schaefer (2005).")
parser.add_argument("--dx", help = "particle spacing in cm (default: 0.0125)", type=float, default = 0.0125)
parser.add_argument("--weibull_m", help = "default: 8.5", type=float, default = 8.5)
parser.add_argument("--weibull_k", help = "default: 1.4e19", type=float, default = 1.4e19)
parser.add_argument("--outfile", help = "default: rod.0000", type=str, default = "rod.0000")
parser.add_argument("--plot", help = "additionally plot flaw distribution (default: false)", action="store_true")
args = parser.parse_args()

print("use a square grid with grid spacing: {}".format(args.dx) )

# scaling is cm, rod geometry and setup from Gray (2001) and Schaefer (2005)
length = 3.
width = 1.
rho = 1.
mat_type = 0
print("rod length: {}".format(length) )
print("rod width: {}".format(width) )

area = length*width
mass = rho * args.dx**2


# create particles
xp = np.arange(-length/2,length/2,args.dx)
yp = np.arange(-width/2,width/2,args.dx)

N = len(xp) * len(yp)
print("using {} SPH particles for whole rod...".format(N) )

x = np.zeros(N)
y = np.zeros(N)

k = 0
for i in range(len(xp)):
    for j in range(len(yp)):
        x[k] = xp[i]
        y[k] = yp[j]
        k += 1


# distribute flaws
flaws = []
for i in range(N):
    flaws.append([])

particle_picked = np.zeros(N, dtype=int)
N_picked = 0
i = 0
while True:
    # abort once each particle has been picked at least once
    if N_picked == N:
        break
    i += 1
    # randomly pick a particle    
    index = int(np.random.randint(N, size=1))
    if particle_picked[index] == 0:
        particle_picked[index] = 1
        N_picked += 1
    # don't set flaws outside 1.1 cm
    # (note: the boundary conditions are imposed outside 1.3 cm, see boundary.cu, where this transition zone avoids fracture at the interface)
    if x[index] < -1.1 or x[index] > 1.1:
        continue
    else:
        # set flaw
        flaw = (i/(args.weibull_k * area))**(1./args.weibull_m)
        flaws[index].append(flaw)

print("picked {} flaws, expectation value (N*logN): {}".format(i, int(np.ceil(N * np.log(N)))) )


"""
output file format:

1:x[0] 2:x[1] 3:v[0] 4:v[1] 5:mass 6:density 7:material type 8:number of flaws 9:DIM-root of damage 10:S/sigma[0][0] 11:S/sigma[0][1] 12:S/sigma[1][0] 13:S/sigma[1][1] 14->14+number of flaws:activation thresholds for this particle
"""

out = open(args.outfile, 'w')
print("writing to output file {}...".format(args.outfile) )

plf = np.zeros_like(x)

for k in range(N):
    plf[k] = len(flaws[k])
    print(x[k], y[k], "0.0", "0.0", mass, rho, mat_type, len(flaws[k]), "0.0", "0.0", "0.0", "0.0", "0.0", *flaws[k], file=out)

print("distributed flaws with Weibull m: {}   k: {}".format(args.weibull_m, args.weibull_k) )
print("max no. flaws per particle: {}".format(int(np.max(plf))) )

out.close()


if args.plot:
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_xlim(-length/2, length/2)
    ax.set_ylim(-width/2, width/2)
    ax.set_ylabel('y')
    ax.set_title('2D tensile rod, flaw distribution')
    ax.set_aspect('equal')
    #ax.grid()
    #s = ax.scatter(x, y, c=plf, s=0.5, cmap='gnuplot')
    #s = ax.scatter(x, y, c=plf, s=0.5, cmap='magma')
    s = ax.scatter(x, y, c=plf, s=0.7)
    fig.colorbar(s, orientation='horizontal', label='no. assigned flaws')
    plt.show()

