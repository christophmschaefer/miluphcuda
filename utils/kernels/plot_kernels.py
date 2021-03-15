#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-pastel')


def wendlandc2(x):
    y = 5./4. * (1-x)**3  *(1+3*x)*(x<1)
    return y

def wendlandc4(x):
    y = 3./2. * (1-x)**5 * (1+5*x+8*x**2) * (x < 1)
    return y

def wendlandc6(x):
    y = 55./32. * (1-x)**7 * (1+7*x+19*x**2+21*x**3) * (x < 1)
    return y



dx = 1e-3
minx = dx
maxx = 2.0
x = np.arange(minx, maxx+dx, dx);

fig, ax = plt.subplots(2)


fig.suptitle("The Wendland Kernel Functions")

ax[1].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$W(x)$')
ax[1].set_ylabel(r'$\frac{\mathrm{d}}{\mathrm{d}x}W(x)$')
for a in ax:
    a.set_xlim(0,1.1)

ax[0].plot(x, wendlandc2(x), label='Wendland C2')
ax[1].plot(x[1:], np.diff(wendlandc2(x)), label=r'$\frac{\mathrm{d}}{\mathrm{d}x}$ Wendland C2')
ax[0].plot(x, wendlandc4(x), label='Wendland C4')
ax[1].plot(x[1:], np.diff(wendlandc4(x)), label=r'$\frac{\mathrm{d}}{\mathrm{d}x}$ Wendland C4')
ax[0].plot(x, wendlandc6(x), label='Wendland C6')
ax[1].plot(x[1:], np.diff(wendlandc6(x)), label=r'$\frac{\mathrm{d}}{\mathrm{d}x}$ Wendland C6')

# test the normalization
c2 = 2*np.trapz(wendlandc2(x), dx=dx)
print("Integral -1:1 wendland c2:", c2)
c4 = 2*np.trapz(wendlandc4(x), dx=dx)
print("Integral -1:1 wendland c4:", c4)
c6 = 2*np.trapz(wendlandc6(x), dx=dx)
print("Integral -1:1 wendland c6:", c6)

ax[0].legend()
ax[1].legend()
plt.savefig('kernels.pdf')
