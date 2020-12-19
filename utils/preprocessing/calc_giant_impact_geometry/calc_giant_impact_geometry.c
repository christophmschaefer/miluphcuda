// Program for calculating the analytical (relative) orbit for 2 bodies, where various options for input parameters are available.
// Finally some important parameters at a different position on this orbit (specified via the bodies distance) are calculated.
// All units are SI
// Christoph Burger 11/Jul/2016

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#define TRUE 1
#define FALSE 0
#define DIM 3
#define EPS6 1.0e-6
#define PATHLENGTH 256

#define ERRORTEXT(x) {fprintf(stderr,x); exit(1);}
#define ERRORVAR(x,y) {fprintf(stderr,x,y); exit(1);}


void help(char* programname)
{
	fprintf(stdout, "\n  Usage: %s [Options]\n\n", programname);
	fprintf(stdout, "    Program for calculating the analytical (relative) orbit for 2 bodies, where various options for input parameters are available.\n");
	fprintf(stdout, "    Finally some important parameters at a different position on this orbit (specified via the bodies distance) are calculated.\n");
	fprintf(stdout, "\n  Options:\n");
	fprintf(stdout, "    -?                    displays this help message and exits\n");
	fprintf(stdout, "    -m projectile-mass    specify mass of the projectile (mandatory)\n");
	fprintf(stdout, "    -M target-mass        specify mass of the target (mandatory)\n\n");
	fprintf(stdout, "    -c                    set this for (impact-angle,distance,relative-velocity) as input parameters\n");
	fprintf(stdout, "    -i                    set this for (impact-parameter,distance,relative-velocity) as input parameters\n");
	fprintf(stdout, "    -v                    set this for (inertial) position and velocity vectors of both bodies as input parameters\n");
	fprintf(stdout, "    -f filename           only for input option -v: this reads (inertial) position and velocity vectors from file\n");
	fprintf(stdout, "                          (2 lines with x1 x2 x3 v1 v2 v3, first one for projectile, second one for target).\n");
	fprintf(stdout, "                          If not set they will be read directly from user input.\n");
	fprintf(stdout, "\n");
}


int main(int argc, char* argv[])
{
	int j;
	const double G = 6.6741e-11;	//gravitational constant
	double m_p = -1.0;	//projectile mass
	double m_t = -1.0;	//target mass
	int c_input = FALSE;	//input parameter definition
	int i_input = FALSE;	//input parameter definition
	int v_input = FALSE;	//input parameter definition
	int f_input = FALSE;	//input parameter definition
	char vectorfile[PATHLENGTH];	//declare string for filename
	FILE *vfl;
	double mu;	//gravitational parameter = G*(m_p+m_t)
	double r, v, alpha, b;	//distance, velocity, impact angle (in rad!) and impact parameter (first used for input and then reused for output)
	double a, e;	//orbital parameters, in case of a parabolic orbit p is saved to a
	double pos_vec_p[DIM], pos_vec_t[DIM];	//position vectors for projectile/target (used as input only)
	double vel_vec_p[DIM], vel_vec_t[DIM];	//velocity vectors for projectile/target (used as input only)
	double vec1[DIM], vec2[DIM];	//auxiliary vectors
	double v_esc;	//escape velocity (first used for input and then reused for output)
	int orbit_shape;	//1 for parabolic, 2 for hyperbolic, 3 for elliptic
	
// process command line options:
	while ( ( j = getopt(argc, argv, "?m:M:civf:") ) != -1 )	//int-representations of command line options are successively saved in j
	switch((char)j)
	{
		case '?':
			help(*argv);
			exit(0);
		case 'm':
			m_p = atof(optarg);
			break;
		case 'M':
			m_t = atof(optarg);
			break;
		case 'c':
			if( (i_input==TRUE) || (v_input==TRUE) )
				ERRORTEXT("ERROR! Only one input parameter definition possible!\n")
			else
				c_input = TRUE;
			break;
		case 'i':
			if( (c_input==TRUE) || (v_input==TRUE) )
				ERRORTEXT("ERROR! Only one input parameter definition possible!\n")
			else
				i_input = TRUE;
			break;
		case 'v':
			if( (c_input==TRUE) || (i_input==TRUE) )
				ERRORTEXT("ERROR! Only one input parameter definition possible!\n")
			else
				v_input = TRUE;
			break;
		case 'f':
			f_input = TRUE;
			strncpy(vectorfile,optarg,PATHLENGTH);
			break;
		default:
			help(*argv);
			exit(1);
	}
	
	mu = G*(m_p+m_t);
	
// read input parameters and convert them if necessary - to always end up with known distance r, velocity v, impact angle alpha, and impact parameter b: 
	fprintf(stdout, "  --------------------------------\n");
	fprintf(stdout, "Enter input parameters:\n");
	if( (c_input == TRUE) || (i_input == TRUE) )
	{
		fprintf(stdout, "    distance (in m) = ");
		fscanf(stdin, "%le", &r);
		fprintf(stdout, "    relative velocity (in m/s) = ");
		fscanf(stdin, "%le", &v);
		if( c_input == TRUE )
		{
			fprintf(stdout, "    impact angle (in °) = ");
			fscanf(stdin, "%le", &alpha);
			alpha = alpha * M_PI / 180.0;	//convert alpha to rad
			b = r * sin(alpha);
			fprintf(stdout, "    (calculated) impact parameter = %e m\n", b);
		}
		else
		{
			fprintf(stdout, "    impact parameter (in m) = ");
			fscanf(stdin, "%le", &b);
			alpha = asin(b/r);	//alpha in rad!
			fprintf(stdout, "    (calculated) impact angle = %e°\n", alpha*180.0/M_PI);
		}
	}
	else if( v_input == TRUE )
	{
		if( f_input == FALSE )	//vectors are read directly from user input
		{
			fprintf(stdout, "  Position and velocity vector of the projectile:\n");
			for(j=0; j<DIM; j++)
			{
				fprintf(stdout, "    x%d (in m) = ", j+1);
				fscanf(stdin, "%le", pos_vec_p+j);
			}
			for(j=0; j<DIM; j++)
			{
				fprintf(stdout, "    v%d (in m/s) = ", j+1);
				fscanf(stdin, "%le", vel_vec_p+j);
			}
			fprintf(stdout, "  Position and velocity vector of the target:\n");
			for(j=0; j<DIM; j++)
			{
				fprintf(stdout, "    x%d (in m) = ", j+1);
				fscanf(stdin, "%le", pos_vec_t+j);
			}
			for(j=0; j<DIM; j++)
			{
				fprintf(stdout, "    v%d (in m/s) = ", j+1);
				fscanf(stdin, "%le", vel_vec_t+j);
			}
		}
		else if( f_input == TRUE )	//vectors are read from file
		{
			fprintf(stdout, "  Position and velocity vectors are read from file: %s\n", vectorfile);
			if ( (vfl = fopen(vectorfile,"r")) == NULL )
				ERRORVAR("FILE ERROR! Cannot open %s for reading!\n",vectorfile)
			fscanf(vfl, "%le %le %le %le %le %le%*[^\n]\n", pos_vec_p, pos_vec_p+1, pos_vec_p+2, vel_vec_p, vel_vec_p+1, vel_vec_p+2);
			fscanf(vfl, "%le %le %le %le %le %le", pos_vec_t, pos_vec_t+1, pos_vec_t+2, vel_vec_t, vel_vec_t+1, vel_vec_t+2);
			fclose(vfl);
		}
		//calculate the (appropriate) relative position vector (vec1) and velocity vector (vec2):
		for(j=0; j<DIM; j++)
			vec1[j] = pos_vec_t[j] - pos_vec_p[j];
		for(j=0; j<DIM; j++)
			vec2[j] = vel_vec_p[j] - vel_vec_t[j];
		//calculate distance and velocity as norm of these vectors:
		r=0.0;
		v=0.0;
		for(j=0; j<DIM; j++)
			r += vec1[j] * vec1[j];
		r = sqrt(r);
		for(j=0; j<DIM; j++)
			v += vec2[j] * vec2[j];
		v = sqrt(v);
		//calculate the impact angle alpha (in rad) via the scalar product:
		alpha = 0.0;
		for(j=0; j<DIM; j++)
			alpha += vec1[j] * vec2[j];
		alpha = acos( alpha/(r*v) );	//alpha in rad!
		b = r * sin(alpha);
		//output calculated values:
		fprintf(stdout, "      ----------------\n");
		fprintf(stdout, "    (calculated) distance = %e m\n", r);
		fprintf(stdout, "    (calculated) relative velocity = %e m/s\n", v);
		fprintf(stdout, "    (calculated) impact angle = %e°\n", alpha*180.0/M_PI);
		fprintf(stdout, "    (calculated) impact parameter = %e m\n", b);
	}
	else
		ERRORTEXT("ERROR! You didn't choose a certain set of input parameters!\n")
	
// calculate v/v_esc and calculate orbit shape for the respective conic section:
	v_esc = sqrt( 2.0*mu/r );
	fprintf(stdout, "  --------------------------------\n");
	fprintf(stdout, "The bodies' mutual escape velocity at these positions is %e m/s, the relative velocity (%e m/s) is %e times this value.\n", v_esc, v, v/v_esc);
	if( (v/v_esc > 1.0-EPS6) && (v/v_esc < 1.0+EPS6) )	//the orbit is parabolic
	{
		orbit_shape = 1;
		a = 2*r*sin(alpha)*sin(alpha);
		fprintf(stdout, "    This is treated as a parabolic orbit with p = %e m. However, since parabolic orbits are just a limiting case, make sure that your orbit is indeed (sufficiently close to) parabolic!\n", a);
	}
	else if( v/v_esc > 1.0 )	//the orbit is hyperbolic
	{
		orbit_shape = 2;
		a = 1.0 / ( v*v/mu - 2.0/r );
		e = sqrt( 1.0 + r/a/a*(2*a+r)*sin(alpha)*sin(alpha) );
		fprintf(stdout, "    This is a hyperbolic orbit with a = %e m and e = %e.\n", a, e);
	}
	else if( (v/v_esc < 1.0) && (v/v_esc > 0.0) )	//the orbit is elliptic
	{
		orbit_shape = 3;
		a = 1.0 / ( 2.0/r - v*v/mu );
		e = sqrt( 1.0 - r/a/a*(2*a-r)*sin(alpha)*sin(alpha) );
		fprintf(stdout, "    This is an elliptic orbit with a = %e m and e = %e.\n", a, e);
	}
	else
		ERRORTEXT("ERROR! Invalid result for velocity over escape velocity!\n")
	
// calculate parameters at desired position on the orbit:
	fprintf(stdout, "  --------------------------------\n");
	fprintf(stdout, "Distance (in m) on this orbit for which output parameters shall be calculated = ");
	fscanf(stdin, "%le", &r);
	
	v_esc = sqrt(2.0*mu/r);
	
	if( orbit_shape == 1 )	//the orbit is parabolic
	{
		v = v_esc;
		alpha = asin( sqrt(a/2.0/r) );
	}
	else if( orbit_shape == 2 )	//the orbit is hyperbolic
	{
		v = sqrt( mu*( 2.0/r + 1.0/a ) );
		alpha = asin( sqrt( a*a*(e*e-1.0)/r/(2.0*a+r) ) );
	}
	else	//the orbit is elliptic
	{
		v = sqrt( mu*( 2.0/r - 1.0/a ) );
		alpha = asin( sqrt( a*a*(1.0-e*e)/r/(2.0*a-r) ) );
	}
	
	b = r*sin(alpha);
	
	fprintf(stdout, "  --------------------------------\n");
	fprintf(stdout, "Results:\n");
	fprintf(stdout, "    The bodies' mutual escape velocity at this distance is %e m/s, the relative velocity (%e m/s) is %e times this value.\n", v_esc, v, v/v_esc);
	fprintf(stdout, "    impact angle = %e°\n", alpha*180.0/M_PI);
	fprintf(stdout, "    impact parameter = %e m\n", b);
	fprintf(stdout, "  --------------------------------\n");
	
	exit(0);
}







