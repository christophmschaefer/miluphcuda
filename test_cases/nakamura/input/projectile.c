/* simple c-routine to build input file for miluph */
/* collision between two basaltic objects */
// Christoph Schaefer, dec 2012
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TRUE (!0)
#define FALSE 0

int main (int argc, char *argv[]) 
{

    double delta = 5e-4;
    double rho = 1.18e3;
    double e, f, g;
    double xmin = -1e-1;
    double ymin = -1e-1;
    double zmin = -1e-1;
    double xmax = 1e-1; 
    double ymax = 1e-1;
    double r;
    double zmax = 1e-1;
    double offset = 0.0;
    double vz = -3.2e3;
    int draw;
    double tmass = 0.2e-3;
    double radius = 0.0;
    double radiussq;
    double mass = 0.0;

    radius = pow((0.75/M_PI * tmass / rho),(1./3));
    offset = delta*3.04 + 3e-2 + radius*2.0;
    radiussq = radius*radius;

    mass = rho * delta*delta*delta;

    // warning: nof = 0 for the projectile        mt nof

    e = xmin;
    while (e < xmax) {
        f = ymin;
        while (f < ymax) {
            g = zmin;
            while (g < zmax) {
                draw = FALSE;
                r = e*e + f*f + g*g;
                if (r <= radiussq) draw = TRUE;
                if (draw) 
                    fprintf(stdout, "%e %e %e 0.0 0.0 %e %e %e 0.0 1 0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n", e, f, g+offset, vz, mass, rho);
                g += delta;
            }
            f += delta;
        }
        e += delta;
    }

	return 0;
}




