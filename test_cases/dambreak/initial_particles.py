#!/usr/bin/env python

"""
Generate initial condition for dambreak simulation
"""

# unity density
density = 1.0
dx = 0.01
m = dx**2 * density

outputf = open("dam.0000", 'w')

ymax = 0.5
xmax = 1.0
ymin = 0.0
xmin = 0.0

x = xmin
while x < xmax:
    y = ymin
    while y < ymax:
        # format: x y vx vy mass material_type
        print("%.17lf %.17lf 0.0 0.0 %.17lf 0" % (x, y, m), file=outputf)
        y += dx
    x += dx

outputf.close()
