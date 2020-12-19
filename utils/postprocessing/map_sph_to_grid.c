/*

    map SPH particles to grid

    author: Christoph Schaefer

*/


#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include <string.h>

#define TRUE 1
#define FALSE 0


#define PACKAGE "map_sph_to_grid"
#define VERSION "0.001-z"


#define INTEGRATE_ENERGY
#define SOLID
#undef FRAGMENTATION
#define POROUS_JUTZI
#define FRAGMENTATION

/* mainly for file format */
#define DIM 2

struct file {
	FILE *data;
	char name[256];
} disk, to, vtkfile;


struct Particle {
	int 		xgrid[DIM];
	int 		noi;
	double		x[DIM];
	double		v[DIM];
	double 		m;
	double 		rho;
	double		p;
	double		e;
	double 		sml;
    double      T;
    double      plastic_strain;
    double      alpha_jutzi;
	int		gsml[DIM];
	int		material_type;
#ifdef SOLID
	double		S[DIM][DIM];
#endif
#ifdef FRAGMENTATION
	double 		damage;
	struct {	
		int 	nof;
		int	noaf;
		double	local_strain;
		double  *activation_threshold;
	} flaws;
#endif
} *g, p;

double gridlength[DIM], offset[DIM];
int gsml[DIM];
double sml = 0.0;
int beautify = FALSE;
int gnuplot = FALSE;
int gridsize = 100;

/* reads a particle in disk.data */
void read_particle(void);
void write_vtkfile(int, double*);
/* does the sph algorithm and stuff */
void rhs(void);
/* returns the kernel value */
double kernel(double, double);
void add_values(double, struct Particle *, struct Particle *);

static void usage(char *name) 
{
	fprintf(stderr, "Usage %s [options]\n"
			"\tmap sph data to grid.\n"
			"Available options:\n"
			"\t-h, --help\t\t\t This message.\n"
			"\t-V, --version\t\t\t Print out version number.\n"
			"\t-N, --numberofparticles\t Number of particles in input filenames.\n"
			"\t-H, --sml\t\t Smoothing length. (default: use particle's sml)\n"
			"\t-b, --beautify\t\t Use a 30%% longer sml. Works only if --sml is not set.\n"
			"\t-{x,y,z}, --{x,y,z}sizemin\t\t Size of the grid. (default: 0.0)\n"
			"\t-{X,Y,Z}, --{x,y,z}sizemax\t\t Size of the grid. (default: 0.0)\n"
			"\t-g, --grid\t\t\t Size of the grid (default: 100 grid cells).\n"
			"\t-G, --gnuplot\t\t\t Output can be used to plot using the pm3d module in gnuplot.\n"
			"\t-f, --filename\t\t\t Input file name. (default: disk.0000)\n"
			"File format is 'x y [z] v_x v_y [v_z] mass [rho] [e] [sml] [noi] [material_type] [number_of_flaws] [number_of_activated_flaws] [local_strain] [damage] [p] [S_xx [S_xy S_xz]] [[flaw_1] ... [flaw_number_of_flaws]]'.\n"
			, name);

	exit(1);
}





int main(int argc, char *argv[])
{

	int c;
	int we_want_it;
	int k, l, m;
	int i;
	int d;
	int nop = 0;
	int not_taken = 0;
	double x[2*DIM] = { 0.0, };
	static struct option opts[] = {
		{ "help", 0, NULL, 'h' },
		{ "version", 0, NULL, 'V' },
		{ "sml", 1, NULL, 'H' },
		{ "numberofparticles", 1, NULL, 'N' },
		{ "grid", 1, NULL, 'g' },
		{ "gnuplot", 0, NULL, 'G' },
		{ "beautify", 0, NULL, 'b' },
		{ "filename", 1, NULL, 'f' },
		{ "xsizemin", 1, NULL, 'x' },
		{ "xsizemax", 1, NULL, 'X' },
		{ "ysizemin", 1, NULL, 'y' },
		{ "ysizemax", 1, NULL, 'Y' },
		{ "zsizemin", 1, NULL, 'z' },
		{ "zsizemmax", 1, NULL, 'Z' }
	};
	
	if (argc == 1) 
		usage(argv[0]);

	strcpy(disk.name, "disk.0000");
	
	while ((c=getopt_long(argc, argv, "bN:Gg:x:X:y:Y:z:Z:H:f:vh", opts, &i)) != -1) {
		switch (c) {
			case 'N':
				nop = atoi(optarg);
				if (nop <= 0) {
					fprintf(stderr, "The number of particles in the file should be greater than zero.\n");
					exit(1);
				}
				break;
			case 'g':
				gridsize = atoi(optarg);
				if (gridsize < 0) {
					fprintf(stderr, "Er? Number of grid cells should be positive.\n");
					exit(1);
				}
				break;
			case 'H':
				sml = atof(optarg);
				if (sml < 0) {
					fprintf(stderr, "Huh: Smoothing length less than zero? Eventing new physics ....\n");
					exit(1);
				}
				break;
			case 'b':
				beautify = TRUE;
				break;
			case 'G':
				gnuplot = TRUE;
				break;
			case 'x':
				x[0] = atof(optarg);
				break;
			case 'X':
				x[1] = atof(optarg);
				break;
			case 'y':
				if (DIM == 1) {
					exit(1);
				}
				x[2] = atof(optarg);
				break;	
			case 'Y':
				if (DIM == 1) {
					fprintf(stderr, "Wrong dimension.\n");
					exit(1);
				}
				x[3] = atof(optarg);
				break;	
			case 'z':
				if (DIM != 3) {
					fprintf(stderr, "Wrong dimension.\n");
					exit(1);
				}
				x[4] = atof(optarg);
				break;
			case 'Z':
				if (DIM != 3) {
					fprintf(stderr, "Wrong dimension.\n");
					exit(1);
				}
				x[5] = atof(optarg);
				break;
			case 'V':
				fprintf(stdout, "This is " PACKAGE " version " VERSION ".\n");
				exit(0);
				break;
			case 'f':
				if (!strcpy(disk.name, optarg))
					exit(1);
				break;
			case 'h':
				usage(argv[0]);
				exit(0);
			default:
				usage(argv[0]);
				exit(0);
		}
	}

	/* check the grid size */
	for (d = 0; d < 2*DIM; d = d+2) {
		if (x[d+1] < x[d]) { 
			fprintf(stderr, "Check your {x,y,z}size{min,max} values.\n");
	                exit(1);
		}
	}
	strcpy(to.name, disk.name);
	strcpy(vtkfile.name, disk.name);
	strcat(to.name, ".grid");
	strcat(vtkfile.name, ".vtk");
	/* alloc memory for the grid points */	
#if (DIM == 1)
	 g = (struct Particle *) calloc(gridsize, sizeof(struct Particle));
#elif (DIM == 2)
	 g = (struct Particle *) calloc(gridsize*gridsize, sizeof(struct Particle));
#elif (DIM == 3)
	 g = (struct Particle *) calloc(gridsize*gridsize*gridsize, sizeof(struct Particle));
#else
# error "No such dimension."
#endif

	/* set gridlength for each dimension */
	gridlength[0] = (x[1]-x[0])/(gridsize-1);
	offset[0] = x[0];
#if (DIM==2)
	gridlength[1] = (x[3]-x[2])/(gridsize-1);
	offset[1] = x[2];
#endif
#if (DIM==3)
	gridlength[1] = (x[3]-x[2])/(gridsize-1);
	offset[1] = x[2];
	gridlength[2] = (x[5]-x[4])/(gridsize-1);
	offset[2] = x[4];
#endif
	/* smoothing length in grid size units */
	for (d = 0; d < DIM; d++) {
		//printf("gridlength[%d]: %e\n", d, gridlength[d]);
		//printf("%e\n", sml);
		gsml[d] = (int) floor(sml/gridlength[d]) + 1;
		//printf("gsml[%d]: %d\n", d, gsml[d]);
	}


	/* initialize gridpoints */
#if (DIM==1)
	for (k = 0; k < gridsize; k++) {
		g[k].xgrid[0] = k;
		g[k].x[0] = k * gridlength[0] + offset[0];
	}
#elif (DIM==2)
	for (k = 0; k < gridsize; k++) {
		for (l = 0; l < gridsize; l++) {
			g[k*gridsize+l].xgrid[0] = k; 
			g[k*gridsize+l].xgrid[1] = l; 
			g[k*gridsize+l].x[0] = k * gridlength[0] + offset[0];
			g[k*gridsize+l].x[1] = l * gridlength[1] + offset[1];
		}
	}
#elif (DIM==3)
	for (k = 0; k < gridsize; k++) {
		for (l = 0; l < gridsize; l++) {
			for (m = 0; m < gridsize; m++) {
				g[k*gridsize*gridsize+l*gridsize+m].xgrid[0] = k; 
				g[k*gridsize*gridsize+l*gridsize+m].xgrid[1] = l; 
				g[k*gridsize*gridsize+l*gridsize+m].xgrid[2] = m; 
				g[k*gridsize*gridsize+l*gridsize+m].x[0] = k * gridlength[0] + offset[0];
				g[k*gridsize*gridsize+l*gridsize+m].x[1] = l * gridlength[1] + offset[1];
				g[k*gridsize*gridsize+l*gridsize+m].x[2] = m * gridlength[2] + offset[2];
			}
		}
	}
#endif
	fprintf(stdout, "Grid initialized.\n");
	fprintf(stdout, "Opening the file %s.\n", disk.name);
	
	disk.data = fopen(disk.name, "r");
	if (disk.data == NULL) {
		fprintf(stderr, "wtf! Cannot open %s\n", disk.name);
		exit(1);
	}
	fprintf(stdout, "Trying to map %d particles to the grid.\n", nop);
	for (i = 0; i < nop; i++) {
		if (!(i%10000)) {
			fprintf(stdout, "Processing particle %d.\n", i);
		}
	    read_particle();

		/* check if we want to use this particle */
		we_want_it = check_particle();
		if (!we_want_it) {
			not_taken++;
			continue;
		}

		if (beautify) {
			p.sml += 0.3*p.sml;
		}
		/* rhs */	
		rhs();
	}
	fclose(disk.data);
	fprintf(stdout, "All particles read and mapped. Skipped %d particles. Now writing to %s.\n", not_taken, to.name);
	to.data = fopen(to.name, "w");
	if (to.data == NULL) {
		fprintf(stderr, "Err! Cannot write to %s\n", to.name);
		exit(1);
	}



#if (DIM == 1)	
	for (k = 0; k < gridsize; k++) {
		fprintf(to.data, "%e\t", g[k].x[0]);
		fprintf(to.data, "%e\t", g[k].rho);

		fprintf(to.data, "\n");
	}
#elif (DIM == 2)
	for (k = 0; k < gridsize; k++) {
		for (l = 0; l < gridsize; l++) {
			for (d = 0; d < DIM; d++) {
				fprintf(to.data, "%e\t", g[k*gridsize+l].x[d]);
			}
			fprintf(to.data, "%e\t", g[k*gridsize+l].rho);
			fprintf(to.data, "%e\t", g[k*gridsize+l].e);
			fprintf(to.data, "\n");
		}
		if (gnuplot)
			fprintf(to.data, "\n");
	}
#elif (DIM == 3)
	for (k = 0; k < gridsize; k++) {
		for (l = 0; l < gridsize; l++) {
			for (m = 0; m < gridsize; m++) {
				for (d = 0; d < DIM; d++) {
					fprintf(to.data, "%e\t", g[k*gridsize*gridsize+l*gridsize+m].x[d]);
				}
				fprintf(to.data, "%e\t", g[k*gridsize*gridsize+l*gridsize+m].rho);
#ifdef FRAGMENTATION
				fprintf(to.data, "%e\t", g[k*gridsize*gridsize+l*gridsize+m].damage);
#endif
				fprintf(to.data, "\n");
			}
		}
	}
#endif
	fclose(to.data);
    write_vtkfile(gridsize, x);
}



void write_vtkfile(int gridsize, double *x)
{


    int k, l, m;

    

    vtkfile.data = fopen(vtkfile.name, "w");
    fprintf(vtkfile.data, "# vtk DataFile Version 3.0\n");
    fprintf(vtkfile.data, "vtkfile\n");
    fprintf(vtkfile.data, "ASCII\n");
    fprintf(vtkfile.data, "DATASET STRUCTURED_POINTS\n");
    fprintf(vtkfile.data, "DIMENSIONS %d", gridsize); 
#if DIM > 1
    fprintf(vtkfile.data, " %d", gridsize); 
#if DIM > 2
    fprintf(vtkfile.data, " %d", gridsize); 
#endif
#endif
    fprintf(vtkfile.data, "\n");
    fprintf(vtkfile.data, "ORIGIN %e", x[0]);
#if DIM > 1
    fprintf(vtkfile.data, " %e", x[2]);
#if DIM > 2
    fprintf(vtkfile.data, " %e", x[4]);
#endif
#endif
    fprintf(vtkfile.data, "\n");
    fprintf(vtkfile.data, "SPACING %e", (x[1]-x[0])/gridsize);
#if DIM > 1
    fprintf(vtkfile.data, " %e", (x[3]-x[2])/gridsize);
#if DIM > 2
    fprintf(vtkfile.data, " %e", (x[5]-x[4])/gridsize);
#endif
#endif
    fprintf(vtkfile.data, "\n");
    fprintf(vtkfile.data, "POINT_DATA %d\n", 
#if DIM == 1
        gridsize
#elif DIM == 2
        gridsize*gridsize
#elif DIM == 3
        gridsize*gridsize*gridsize
#endif
    );
    fprintf(vtkfile.data, "SCALARS density float 1\n");
    fprintf(vtkfile.data, "LOOKUP_TABLE default\n");

#if (DIM == 1)	
	for (k = 0; k < gridsize; k++) {
		fprintf(vtkfile.data, "%e ", g[k].rho);
	}
    fprintf(vtkfile.data, "\n");
#elif (DIM == 2)
	for (l = 0; l < gridsize; l++) {
		for (k = 0; k < gridsize; k++) {
			fprintf(vtkfile.data, "%e ", g[k*gridsize+l].rho);
		}
	    fprintf(vtkfile.data, "\n");
	}
#elif (DIM == 3)
	for (m = 0; m < gridsize; m++) {
		for (l = 0; l < gridsize; l++) {
			for (k = 0; k < gridsize; k++) {
				fprintf(vtkfile.data, "%e ", g[k*gridsize*gridsize+l*gridsize+m].rho);
			}
			fprintf(vtkfile.data, "\n");
		}
	    fprintf(vtkfile.data, "\n");
	}
#endif
    

    fclose(vtkfile.data);


}


void read_particle(void) 
{
	int d, e;
	int c;
	/* read in coordinates */
	for (d = 0; d < DIM; d++) 
		fscanf(disk.data, "%le", &p.x[d]);
	/* read in velocities */
	for (d = 0; d < DIM; d++) 
		fscanf(disk.data, "%le", &p.v[d]);
	/* read in mass */
	fscanf(disk.data, "%le", &p.m);
	/* read in density */
	fscanf(disk.data, "%le", &p.rho);
#ifdef INTEGRATE_ENERGY
	/* read in energy */
	fscanf(disk.data, "%le", &p.e);
#endif
	fscanf(disk.data, "%le", &p.sml);
	fscanf(disk.data, "%d", &p.noi);
	/* read in material type */
	fscanf(disk.data, "%d", &p.material_type);
#ifdef PLASTICITY
	fscanf(disk.data, "%e", &p.plastic_strain);
	fscanf(disk.data, "%e", &p.T);
#endif
#ifdef FRAGMENTATION
	fscanf(disk.data, "%d", &p.flaws.nof); 
	fscanf(disk.data, "%d", &p.flaws.noaf);
	fscanf(disk.data, "%le", &p.flaws.local_strain);
	/* read in actual damage */
	fscanf(disk.data, "%le", &p.damage);
#endif
	fscanf(disk.data, "%le", &p.p);
#ifdef SOLID
	/* read in tensor */
	// FixMe: there will be only 5 components
	for (d = 0; d < DIM; d++)
		for (e = 0; e < DIM; e++)
			fscanf(disk.data, "%le", &p.S[d][e]);
#endif
#ifdef POROUS_JUTZI
	fscanf(disk.data, "%le", &p.alpha_jutzi);
#endif
	/* skip until end of line */
	do {
	        c = fgetc(disk.data);
        } while (c != EOF && c != (int) '\n');

}



int check_particle(void) 
{
	int d;
	int e;	
	int we_want_it = TRUE;
	for (d = 0; d < DIM; d++) {
		p.xgrid[d] = (int) floor((p.x[d] - offset[d])/gridlength[d]);
		if (sml == 0) {
			p.gsml[d] = (int) floor(p.sml/gridlength[d]) + 1;
		}
		if (p.xgrid[d] < 0 || p.xgrid[d] >= gridsize)
			we_want_it = FALSE;
	}
	return we_want_it;
}


/* the sph algorithm */
void rhs(void)
{
	int k, l, m;
	int d;
	int kmin, kmax, lmin, lmax, mmin, mmax;
	double dr[DIM], rr;
	double hh, h;
	struct Particle *r;
	double w;

    int interacted = FALSE;
	
	/* determine indices for the sum */	
	hh = sml * sml;
	if (sml == 0) {
		for (d = 0; d < DIM; d++)
			gsml[d] = p.gsml[d];
		hh = p.sml*p.sml;
	}
	k = p.xgrid[0];
#if (DIM > 1)
	l = p.xgrid[1];
#endif
#if (DIM > 2)
	m = p.xgrid[2];
#endif


	(k - gsml[0]) >= 0 ? (kmin = k - gsml[0]) : (kmin = 0);
	(k + gsml[0]) <= (gridsize-1) ? (kmax = k + gsml[0]) : (kmax = gridsize-1);
#if (DIM > 1)
	(l - gsml[1]) >= 0 ? (lmin = l - gsml[1]) : (lmin = 0);
	(l + gsml[1]) <= (gridsize-1) ? (lmax = l + gsml[1]) : (lmax = gridsize-1);
#endif
#if (DIM > 2)
	(m - gsml[2]) >= 0 ? (mmin = m - gsml[2]) : (mmin = 0);
	(m + gsml[2]) <= (gridsize-1) ? (mmax = m + gsml[2]) : (mmax = gridsize-1);
#endif
#if (DIM == 1)
	/* search interaction partners */
	h = sqrt(hh);
	for (k = kmin; k < kmax+1; k++) {
		r = &g[k];
		rr = (r->x[0] - p.x[0])*(r->x[0] - p.x[0]);	
		/* check if there is interaction */
		if (rr < hh) {
            interacted = TRUE;
			/* get the kernel for this interaction */
			rr = sqrt(rr);
			w =  kernel(rr, h);
			/* add the stuff to the gridpoint */
			add_values(w, r, &p);
		}
	}
#elif (DIM == 2)
	h = sqrt(hh);
	for (k = kmin; k <= kmax; k++) {
		for (l = lmin; l <= lmax; l++) {
			r = &g[k*gridsize+l];
			/* distance to this grid point */
			rr = 0;
			for (d = 0; d < DIM; d++) {
				dr[d] = r->x[d] - p.x[d];
				rr += dr[d]*dr[d];
			}
			/* check if there is interaction */
			if (rr < hh) {
                interacted = TRUE;
				/* get the kernel for this interaction */
				rr = sqrt(rr);
				w =  kernel(rr, h);
				/* add the stuff to the gridpoint */
				add_values(w, r, &p);
			}
		}
	}
#elif (DIM == 3)
	h = sqrt(hh);
	for (k = kmin; k <= kmax; k++) {
		for (l = lmin; l <= lmax; l++) {
			for (m = mmin; m <= mmax; m++) {
				r = &g[k*gridsize*gridsize+l*gridsize+m];
				/* distance to this grid point */
				rr = 0;
				for (d = 0; d < DIM; d++) {
					dr[d] = r->x[d] - p.x[d];
					rr += dr[d]*dr[d];
				}
				/* check if there is interaction */
				if (rr < hh) {
                    interacted = TRUE;
					/* get the kernel for this interaction */
					rr = sqrt(rr);
					w =  kernel(rr, h);
					/* add the stuff to the gridpoint */
					add_values(w, r, &p);
				}
			}
		}
	}
#endif
    if (!interacted) {
        fprintf(stderr, "Warning: Did not find any interacting grid point for particle.\n");
    }
}

/* returns kernel values */
double kernel(double r, double h)
{
	int i;
	int d;
	double h2, h3, h4;
	double f1, f2;
	double tmp, tmp2;
	double w;


	h2 = h*h;
	h3 = h2*h;
	h4 = h3*h;
#if (DIM == 2)
	f1 = 40.0/(7.0*M_PI*h4*h);
#elif (DIM == 3)
	f1 = 8.0/(M_PI*h4*h2);
#elif (DIM == 1)
	f1 = 4.0/(3.0*h4);
#endif	
	f2 = 6.0*f1;

	/* distinction of cases */
	/* this is getting ugly - yea - yes */
	if (r > h) {
		w = 0;
	} else if (r < 0.5*h) {
		w = f1*(6.0*r*r*(r-h) + h3);
	} else {
		tmp2 = h - r;
		w = f1*2.0*tmp2*tmp2*tmp2;
	}
	return w;	
}

void add_values(double w, struct Particle *r, struct Particle *p)
{ 
	int d, e;
	r->rho += p->m * w;
	r->p += p->m/p->rho * p->p * w;
	for (d = 0; d < DIM; d++) {
		r->v[d] += p->m/p->rho * p->v[d] * w;
	}
#ifdef SOLID
	for (d = 0; d < DIM; d++) {
		for (e = 0; e < DIM; e++) {
			r->S[d][e] += p->m/p->rho * p->S[d][e] * w;
		}
	}
#endif
#ifdef FRAGMENTATION
	r->flaws.local_strain += p->m/p->rho * p->flaws.local_strain * w;
	r->damage += p->m/p->rho * p->damage * w;
#endif
	r->e += p->m/p->rho * p->e * w;
	
}



