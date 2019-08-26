/* simple c-routine to build input file for miluph/parasph */
/* 2 rings */
// Christoph Schaefer, April 2012
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TRUE (!0)
#define FALSE 0

int main (int argc, char *argv[]) 
{
	double x, y;
	double x1, y1;
	int draw = FALSE;
	double rmax = 4.0;
	double rmin = 3.0;
    double r;
    double speed = 0.059;
    double delta = 0.075;
    double ydelta = delta;
    double mass;
    int i = 0;
    int j = 0;


    x1 = 5.0;
    y1 = 0;

    mass = delta*delta;



    for (i = -1000; i < 1000; i++) {
        for (j = -1000; j < 1000; j++) {
            y = i*ydelta;
            x = j*delta;

            r = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1));
            if (r > rmin && r < rmax)
                draw = TRUE;

            if (draw) {
			    fprintf(stdout, "%e %e %e 0.0 %e 1.0 0 0.0 0.0 0.0 0.0\n", x, y, -speed, mass);
			    fprintf(stdout, "%e %e %e 0.0 %e 1.0 0 0.0 0.0 0.0 0.0\n", -x, y, speed, mass);
                draw = FALSE;
            }
        }
    }

	return 0;
}
