/* simple c-routine to build input file for miluph */
/* Nakamurara & Fujiwara 1991 impact experiment, 0.2 g nylon bullet impacting 3 cm radius basalt sphere @ 3.2 km/s */
// Christoph Schaefer, dec 2012
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TRUE (!0)
#define FALSE 0

int main (int argc, char *argv[]) 
{

    int draw = FALSE;
	double velocity = 0.0;
    double delta = 5e-4;
    double rho = 2.7e3;
    double e, f, g;
    double xmin = -1e-1;
    double ymin = -1e-1;
    double zmin = -1e-1;
    double xmax = 1e-1; 
    double ymax = 1e-1;
    double zmax = 1e-1;
    double r;
    double deltax = 0;


    double angle;

    angle = 30; 
    angle *= M_PI/180;

    //fprintf(stderr, "angle: %f\n", sin(angle));
    double radius = 3e-2;

    double radiussq = radius*radius;
    deltax = sin(angle) * radius;
    
    e = xmin;
    while (e < xmax) {
        f = ymin;
        while (f < ymax) {
            g = zmin;
            while (g < zmax) {
                draw = FALSE;
                r = e*e + f*f + g*g;
                if (r <= radiussq) draw = TRUE;
                if (draw) fprintf(stdout, "%e %e %e 0.0 0.0 0.0 %e %e 0.0 0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n", e+deltax, f, g, rho*delta*delta*delta, rho);
                g += delta;
            }
            f += delta;
        }
        e += delta;
    }
	return 0;
}
