#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt


# scaling is cm, rod geometry and setup from Gray 2001 and Schaefer 2005
length = 3
width = 1

A = length*width

dx = 0.0125

rho = 1
mass = rho * dx**2

material_type = 0

xp = np.arange(-length/2,length/2,dx)
yp = np.arange(-width/2,width/2,dx)

N = len(xp) * len(yp)

x = np.zeros(N)
y = np.zeros(N)

flaws = []
for i in range(N):
    flaws.append([])

# weibull parameters
m_wei = 8.5
k_wei = 1.4e19


Nflaws = int(np.ceil(N * np.log(N)))

k = 0
for i in range(len(xp)):
    for j in range(len(yp)):
        x[k] = xp[i]
        y[k] = yp[j]
        k = k+1

for i in range(Nflaws):
    # arbitray choose a particle    
    index = int(np.random.randint(N, size=1))
    if x[index] < -1.2 or x[index] > 1.2:
        continue

    flaw = (i/(k_wei*A))**(1./m_wei)
    flaws[index].append(flaw)


"""
1:x[0] 2:x[1] 3:v[0] 4:v[1] 5:mass 6:density 7:material type 8:number of flaws 9:DIM-root of damage 10:S/sigma[0][0] 11:S/sigma[0][1] 12:S/sigma[1][0] 13:S/sigma[1][1] 14->14+number of flaws:activation thresholds for this particle
"""



out = open("rod.0000", 'w')

plf = np.zeros_like(x)

for k in range(N):
    plf[k] = len(flaws[k])
    print(x[k], y[k], "0.0", "0.0", mass, rho, material_type, len(flaws[k]), "0.0", "0.0", "0.0", "0.0", "0.0", *flaws[k], file=out)


print(np.max(plf))

out.close()

fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_xlim(-length/2, length/2)
ax.set_ylim(-width/2, width/2)
ax.set_ylabel('y')
ax.set_title('2D rod, flaw distribution')
ax.set_aspect('equal')
#ax.grid()
#s = ax.scatter(x, y, c=plf, s=0.5, cmap='gnuplot')
#s = ax.scatter(x, y, c=plf, s=0.5, cmap='magma')
s = ax.scatter(x, y, c=plf, s=0.7)
fig.colorbar(s, orientation='horizontal', label='number of assigned flaws')
plt.show()


