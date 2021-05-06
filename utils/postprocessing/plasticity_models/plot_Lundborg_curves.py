#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt


pressure = np.linspace(0, 1e12, 1000)


Y_M = 3.5e9
Y_0 = np.array([1e6,1e7,1e8,1e9])
mu = 1.0
yield_strength = np.zeros([len(Y_0), len(pressure)])

for i in range(len(yield_strength[:,0])):
    yield_strength[i,:] = Y_0[i] + mu * pressure / (1 + mu *pressure / (Y_M - Y_0[i]))

fig, ax = plt.subplots()

for i in range(len(yield_strength[:,0])):
    label = "%g" % (Y_0[i])
    ax.plot(pressure, yield_strength[i,:], '--', label=label)
plt.grid()
ax.legend()
ax.set_xscale('log')
ax.set_xlabel('Pressure [Pa]')
ax.set_ylabel('Yield Strength [Pa]')
plt.show()
