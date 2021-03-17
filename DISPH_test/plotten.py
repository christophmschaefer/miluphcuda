import matplotlib.pyplot as plt
import numpy as np
import os

n = 128
N = 25**2*2
dn = 1
real_n = int(n/dn)

for i in range(real_n):
    k = dn*i

    if k<10000:
        n0 = 4 - len(str(k))
    else:
        n0 = 5 - len(str(k))
    OO = ''
    for j in range(n0):
        OO += str(0)
    filename = 'fluids.' + OO + str(k)

    b = 'making picture from file' + filename
    print(b, end="\r")

    x, y = np.loadtxt(filename,  usecols=(0, 1), unpack=True)

    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for j in range(N):
        if j<N/2:
            x1.append(x[j])
            y1.append(y[j])
        else:
            x2.append(x[j])
            y2.append(y[j])

    x_bp = []
    y_bp = []
    for l in range(N, len(x)):
        x_bp.append(x[l])
        y_bp.append(y[l])

    plt.plot(x_bp, y_bp, '.k')
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    plt.axis('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()

    fileout = filename+'.png'
    plt.savefig(os.path.join('pics',fileout))
    plt.close()
