import matplotlib.pyplot as plt
import numpy as np
import sys

N = 25**2*2

filename = sys.argv[1]

x, y  = np.loadtxt(filename,  usecols=(0, 1), unpack=True)

x1 = []
x2 = []
y1 = []
y2 = []
for i in range(N):
    if i<N/2:
        x1.append(x[i])
        y1.append(y[i])
    else:
        x2.append(x[i])
        y2.append(y[i])

x_bp = []
y_bp = []
for j in range(N, len(x)):
    x_bp.append(x[j])
    y_bp.append(y[j])

plt.plot(x_bp, y_bp, '.k')
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.axis('equal')
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

fileout = filename+'.png'
plt.savefig(fileout)
plt.close()
