#!/usr/bin/env python

"""
Generate initial condition for dambreak simulation
"""

# unity density
density = 1.0
dx = 0.01
m = dx**2 * density

outputf = open("dam.0000", 'w')

xbound = 100
ymax = 0.5
xmax = 1.0 + xbound
ymin = 0.0
xmin = 0.0

x = xmin - 3*dx
while x < xmax:
    y = ymin - 3*dx
    while y < ymax:
        # format: x y vx vy mass material_type
        material_type = 0
        if x < xmin or y < ymin:
            material_type = 1
        if x < xmax - xbound:
            print("%.17lf %.17lf 0.0 0.0 %.17lf %.17lf %d" % (x, y, m, density, material_type), file=outputf)
        elif y < ymin:
            material_type = 1
            print("%.17lf %.17lf 0.0 0.0 %.17lf %.17lf %d" % (x, y, m, density, material_type), file=outputf)
        y += dx
    x += dx

outputf.close()
