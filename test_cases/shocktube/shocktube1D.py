#!/usr/bin/env python3


import numpy as np
import sys


outfn = "shocktube.0000"
outfile = open(outfn, 'w')

xmin=-1
xmax=2

# setup following http://www.tat.physik.uni-tuebingen.de/~kley/lehre/cp-prakt/projekte/projekt-kley.pdf
rho0 = 1
p0 = 1
e0 = 2.5
gamma = 1.4

rho1 = 0.125
p1 = 0.1
e1 = 2.0

vx = 0
mt = 0

dx = 5e-4
print(dx*rho0/rho1)
m = rho0 * dx
x = xmin
while x < xmax:
    if x > 0.5:
        x += rho0/rho1*dx
        p = p1
        e = e1
    else:
        x += dx
        p = p0
        e = e0
# 1:x[0] 2:x[1] 3:v[0] 4:v[1] 5:mass 6:density 7:energy 8:material type 
# 1:x[0] 2:v[0] 3:mass 4:energy 5:material type 
    #print("%e %e %e %e %e %d" % (x[i], vx, m, rho, e, mt), file=outfile)
    print("%e %e %e %e %d" % (x, vx, m, e, mt), file=outfile)




outfile.close()
