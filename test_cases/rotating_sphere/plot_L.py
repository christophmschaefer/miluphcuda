#!/usr/bin/env python3

import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
except:
    print("Required modules not found. Exiting")
    sys.exit(1)

# plt.style.use('dark_background')

dpi = 100
fig = plt.figure(figsize=(600/dpi, 600/dpi))
time, Ltot = np.loadtxt("conserved_quantities.log", usecols = (0,11), unpack=True)
ax = fig.add_subplot(111)

ax.plot(time, Ltot, ':')
ax.set_xlabel("Time [code units]")
ax.set_ylabel("Total angular momentum [code units]")
ax.grid()

fig.savefig("angular_momentum.png", dpi=dpi)



