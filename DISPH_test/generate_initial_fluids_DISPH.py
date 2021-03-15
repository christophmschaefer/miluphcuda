import numpy as np
import matplotlib.pyplot as plt

N1 = 25**2
N2 = 25**2
N = N1 + N2

rho1 = 1.0e3
rho2 = 8.0e3
m1 = rho1/N1
m2 = rho2/N2
m_bp = (m1+m2)/2

u1 = 8.0e3
u2 = 1.0e3
q1 = rho1*u1
q2 = rho2*u2
U1 = m1*u1
U2 = m2*u2

q_bp = (q1 + q2)/2
U_bp = (U1 + U2)/2


v1 = 0.0
v2 = 0.0
x_left = 0.0
x_right = 1.0
y_bottom = 0.0
y_top = 2.5
p_distance_y1 = 1/np.sqrt(N1)
p_distance_y2 = 1/np.sqrt(N2)
p_distance_x = (x_right - x_left)/np.sqrt(N1)


outputf = open("fluids.0000", 'w')

x1 = []
y1 = []
x2 = []
y2 = []


num_of_rows = 3
dx_bp = 0.05
x_b_left = - num_of_rows*dx_bp
y_b_down = - num_of_rows*dx_bp
y_b_top = y_top + dx_bp

x_bp_down = []
y_bp_down = []

N_bp_x_row = int((1 + 2*num_of_rows*dx_bp)/dx_bp)+1
N_bp_y_row = num_of_rows + 1
N_bp_column = int(y_top/dx_bp) 

# floor of the jar
for i in range(N_bp_x_row):
    for j in range(N_bp_y_row):
        x_bp_down.append(x_b_left + i*dx_bp)
        y_bp_down.append(y_b_down + j*dx_bp)

# sides of the char
x_bp_sides = []
y_bp_sides = []
#for k in range(1,num_of_rows): 
for i in range(N_bp_y_row):
    for j in range(1,N_bp_column):
        x_bp_sides.append(x_b_left + i*dx_bp)
        y_bp_sides.append(y_bottom + j*dx_bp)
        x_bp_sides.append(x_right + i*dx_bp)
        y_bp_sides.append(y_bottom + j*dx_bp)
    
# top of the jar
x_bp_top = []
y_bp_top = []
for i in range(N_bp_x_row):
    for j in range(N_bp_y_row):
        x_bp_top.append(x_b_left + i*dx_bp)
        y_bp_top.append(y_top + j*dx_bp)
        
        
x_bp = x_bp_down + x_bp_sides + x_bp_top
y_bp = y_bp_down + y_bp_sides + y_bp_top


v_trigger = 2.0


for i in range(int(np.floor(np.sqrt(N1)))):
    for j in range(int(np.floor(np.sqrt(N1)))):
        x = p_distance_x/2 + i*p_distance_x
        y = p_distance_y1/2 + j*p_distance_y1
        if x < 0.5 and y > 0.9:
            print("%e %e %e %e %e %e %e 0" % (x, y, v1, v_trigger,  m1, q1, U1), file=outputf)
        else:
            print("%e %e %e %e %e %e %e 0" % (x, y, v1, v1, m1, q1, U1), file=outputf)
        x1.append(p_distance_x/2 + i*p_distance_x)
        y1.append(p_distance_y1/2 + j*p_distance_y1)

for i in range(int(np.floor(np.sqrt(N2)))):
    for j in range(int(np.floor(np.sqrt(N2)))):
        x = p_distance_x/2 + i*p_distance_x
        y = 1 +  p_distance_y2/2 + j*p_distance_y2
        if x > 0.5 and y < 1.1:
            print("%e %e %e %e %e %e %e 1" % (x, y, v2, -v_trigger, m2, q2, U2), file=outputf)
        else:
            print("%e %e %e %e %e %e %e 1" % (x,y, v2, v2, m2, q2, U2), file=outputf)
        x2.append(p_distance_x/2 + i*p_distance_x)
        y2.append(1 + p_distance_y2/2 + j*p_distance_y2)



for i in range(len(x_bp)):
    print("%e %e 0.0 0.0 %e %e %e 2" % (x_bp[i], y_bp[i], m_bp, q_bp, U_bp), file=outputf)
        
plt.plot(x_bp, y_bp, '.k')
    
    

outputf.close()

plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.axis('equal')
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

plt.savefig("fluids_initial.png")
