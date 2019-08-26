/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com and Thomas I. Maindl
 *
 * @section     LICENSE
 * Copyright (c) 2019 Christoph Schaefer
 *
 * This file is part of miluphcuda.
 *
 * miluphcuda is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * miluphcuda is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "miluph.h"
#include "pressure.h"
#include "memory_handling.h"
#include "device_tools.h"
#include "kernel.h"
#include "little_helpers.h"
#include <cuda_runtime.h>

#if USE_SIGNAL_HANDLER
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <stdio.h>
volatile int terminate_flag = 0;
#endif



RunParameter param;

// the pointers to the arrays on the host
struct Particle p_host;
// the pointers to the arrays on the device
__constant__ struct Particle p;
// helper pointers for immutables
__constant__ struct Particle p_rhs;
// the pointers to the arrays on the device residung on the host
struct Particle p_device;
// the pointers for the runge-kutta substeps
__constant__ struct Particle rk[3];
struct Particle rk_device[3];

// the pointers for the predictor-corrector scheme
struct Particle predictor_device;
__constant__ struct Particle predictor;

// the stuff for gravitating point masses
struct Pointmass pointmass_host;
__constant__ struct Pointmass pointmass;
struct Pointmass pointmass_device;
struct Pointmass rk_pointmass_device[3];
__constant__ struct Pointmass rk_pointmass[3];
struct Pointmass predictor_pointmass_device;
__constant__ struct Pointmass predictor_pointmass;
int numberOfPointmasses;
int memorySizeForPointmasses;


int restartedRun = FALSE;

extern double startTime;

double treeTheta;

int maxNumFlaws_host;
int *interactions;
int *interactions_host;

int *childList_host;

int numberOfParticles;
int numberOfRealParticles;
int maxNumberOfParticles;

int memorySizeForTree;
int memorySizeForParticles;
int memorySizeForInteractions;
int memorySizeForChildren;
int memorySizeForStress;
#if FRAGMENTATION
int memorySizeForActivationThreshold;
#endif

int numberOfMultiprocessors;

int numberOfChildren = pow(2, DIM);
int numberOfNodes;


// the sph-kernel function pointers
extern __device__ SPH_kernel kernel;
extern __device__ SPH_kernel wendlandc2_p;
extern __device__ SPH_kernel wendlandc4_p;
extern __device__ SPH_kernel wendlandc6_p;
extern __device__ SPH_kernel cubic_spline_p;
extern __device__ SPH_kernel spiky_p;
extern __device__ SPH_kernel quartic_spline_p;
SPH_kernel kernel_h;




static void print_compile_information(void)
{
    /* give info about physical model */
    char yesno[10];
    fprintf(stdout, "Parameters: \n"
                    "Number of dimensions: %d\n", DIM);
#if INTEGRATE_ENERGY
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Solve energy equation:\t  %s\n", yesno);
#if INTEGRATE_DENSITY
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Solve continuity equation:\t  %s\n", yesno);
#if NAVIER_STOKES
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Solve Navier-Stokes equation:\t  %s\n", yesno);
#if DENSITY_FLOOR
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "using density floor: \t    %s\n", yesno);
#if SOLID
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Solid mechanics:\t  %s\n", yesno);
#if GRAVITATING_POINT_MASSES
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Additional point masses: \t %s\n", yesno);
#if FRAGMENTATION
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Fracture model:\t  %s\n", yesno);
#if PALPHA_POROSITY
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "P-alpha porosity model:\t  %s\n", yesno);
#if SIRONO_POROSITY
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Sirono porosity model:\t  %s\n", yesno);
#if EPSALPHA_POROSITY
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Epsilon-alpha porosity model:\t  %s\n", yesno);
#if VON_MISES_PLASTICITY
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Plasticity model:\t  %s", yesno);
#if VON_MISES_PLASTICITY
#if MOHR_COULOMB_PLASTICITY
    fprintf(stdout, "\t\t\t Mohr-Coulomb\n");
#elif DRUCKER_PRAGER_PLASTICITY
    fprintf(stdout, "\t\t\t Drucker-Prager\n");
#elif COLLINS_PRESSURE_DEPENDENT_YIELD_STRENGTH
    fprintf(stdout, "\t\t\t Pressure dependent yield strength with cohesion for damaged material\n");
#else
    fprintf(stdout, "\t\t\t simple von Mises yield criterion\n");
#endif
#else
    fprintf(stdout, "\n");
#endif
#if JC_PLASTICITY
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Plasticity model from Johnson - Cook:\t   %s\n", yesno);
#if TENSORIAL_CORRECTION
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "SPH linear consistency for strain rate and rotation rate tensor only:\t  %s\n", yesno);
#if ARTIFICIAL_VISCOSITY
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Using SPH representation of momentum and energy conservation version: \t"
#if SPHEQUATIONS == SPH_VERSION1
                        "SPH_VERSION1"
#elif SPHEQUATIONS == SPH_VERSION2
                        "SPH_VERSION2"
#endif
                        "\n");
    fprintf(stdout, "Standard SPH artificial viscosity:\t  %s\n", yesno);
#if XSPH
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "XSPH velocity smoothing:\t  %s\n", yesno);
#if VARIABLE_SML
    fprintf(stdout, "Using variable smoothing:\t yes\n");
#if FIXED_NOI
    fprintf(stdout, "\t\t with fixed number of interaction partners.\n");
#elif INTEGRATE_SML
    fprintf(stdout, "\t\t with integration of the smoothing length.\n");
#else
#error no such scheme for VARIABLE_SML
#endif
    fprintf(stdout, "Using fixed smoothing lengths: \t no\n");
#else
    fprintf(stdout, "Using variable smoothing:\t no\n");
    fprintf(stdout, "Using fixed smoothing lengths: \t yes\n");
#endif
#if READ_INITIAL_SML_FROM_PARTICLE_FILE
    fprintf(stdout, "Reading initial smoothing length for each particle.\n");
#endif
#if GHOST_BOUNDARIES
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "ghost boundaries:\t  %s\n", yesno);
    fprintf(stdout, "boundary particle id: %d\n", BOUNDARY_PARTICLE_ID);
#if HDF5IO
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "HDF5 i/o:\t  %s\n", yesno);
    if (param.verbose) {
        if (param.hdf5output) {
            strcpy(yesno, "yes");
        } else {
            strcpy(yesno, "no");
        }
        fprintf(stdout, "using HDF5 output: \t %s \n", yesno);
        if (param.hdf5input) {
            strcpy(yesno, "yes");
        } else {
            strcpy(yesno, "no");
        }
        fprintf(stdout, "using HDF5 input: \t %s \n", yesno);
        if (param.ascii_output) {
            strcpy(yesno, "yes");
        } else {
            strcpy(yesno, "no");
        }
        fprintf(stdout, "using ASCII output: \t %s \n", yesno);
    }

    fprintf(stdout, "implemented equations of state and corresponding eos type entry in material.cfg:\n");
    fprintf(stdout, "EOS_TYPE_IGNORE          \t\t\t %d\n", EOS_TYPE_IGNORE);
    fprintf(stdout, "EOS_TYPE_POLYTROPIC_GAS  \t\t\t %d\n", EOS_TYPE_POLYTROPIC_GAS);
    fprintf(stdout, "EOS_TYPE_MURNAGHAN       \t\t\t %d\n", EOS_TYPE_MURNAGHAN);
    fprintf(stdout, "EOS_TYPE_TILLOTSON       \t\t\t %d\n", EOS_TYPE_TILLOTSON);
    fprintf(stdout, "EOS_TYPE_ISOTHERMAL_GAS  \t\t\t %d\n", EOS_TYPE_ISOTHERMAL_GAS);
    fprintf(stdout, "EOS_TYPE_REGOLITH        \t\t\t %d\n", EOS_TYPE_REGOLITH);
    fprintf(stdout, "EOS_TYPE_JUTZI           \t\t\t %d\n", EOS_TYPE_JUTZI);
    fprintf(stdout, "EOS_TYPE_JUTZI_MURNAGHAN \t\t\t %d\n", EOS_TYPE_JUTZI_MURNAGHAN);
    fprintf(stdout, "EOS_TYPE_ANEOS           \t\t\t %d\n", EOS_TYPE_ANEOS);
    fprintf(stdout, "EOS_TYPE_VISCOUS_REGOLITH\t\t\t %d\n", EOS_TYPE_VISCOUS_REGOLITH);
    fprintf(stdout, "EOS_TYPE_IDEAL_GAS       \t\t\t %d\n", EOS_TYPE_IDEAL_GAS);
    fprintf(stdout, "EOS_TYPE_SIRONO          \t\t\t %d\n", EOS_TYPE_SIRONO);
    fprintf(stdout, "EOS_TYPE_EPSILON         \t\t\t %d\n", EOS_TYPE_EPSILON);
    fprintf(stdout, "EOS_TYPE_LOCALLY_ISOTHERMAL_GAS \t\t %d\n", EOS_TYPE_LOCALLY_ISOTHERMAL_GAS);


}

static void format_information(char *name)
{
    char physics[10];
    int noc = 0;
    int i, j, k;
#if SOLID
    strcpy(physics, "solid");
#else
    strcpy(physics, "hydro");
#endif

    fprintf(stdout, "Data file format for %s\n", name);
    fprintf(stdout, "dimension = %d\n", DIM);
    fprintf(stdout, "%s version (hydro or solid): %s\n", name, physics);
    fprintf(stdout, "\n");
    fprintf(stdout, "input file format for file <string.XXXX>:\n");
    for (i = 0; i < DIM; i++)
        fprintf(stdout, "%d:x[%d] ", i+1, i);
    noc = DIM; /* x */
    for (i = noc, j = 0; i < noc+DIM; i++, j++)
        fprintf(stdout, "%d:v[%d] ", i+1, j);
    noc += DIM; /* v */
    noc++;  /* m */
    fprintf(stdout, "%d:mass ", noc);
#if INTEGRATE_DENSITY
    noc++;   /* rho */
    fprintf(stdout, "%d:density ", noc);
#endif
#if INTEGRATE_ENERGY
    noc++;   /* e */
    fprintf(stdout, "%d:energy ", noc);
#endif
#if READ_INITIAL_SML_FROM_PARTICLE_FILE
    noc++; /* smoothing length */
    fprintf(stdout, "%d:smoothing length ", noc);
#endif
    noc++; /* material_type */
    fprintf(stdout, "%d:material type ", noc);
#if JC_PLASTICITY
    noc++;   /* ep */
    fprintf(stdout, "%d:strain ", noc);
    noc++;   /* T */
    fprintf(stdout, "%d:temperature ", noc);
#endif
#if FRAGMENTATION
    noc++; /* number of flaws */
    fprintf(stdout, "%d:number of flaws ", noc);
    noc++; /* damage */
    fprintf(stdout, "%d:DIM-root of tensile damage ", noc);
#endif
#if SOLID
    k = noc+1;
    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            fprintf(stdout, "%d:S/sigma[%d][%d] ", k, i, j);
            k++;
        }
    }
    noc += DIM*DIM; /* S */
#endif
#if SIRONO_POROSITY
    noc++;  /* rho_0prime */
    fprintf(stdout, "%d:rho_0prime ", noc);
    noc++;  /* rho_c_plus */
    fprintf(stdout, "%d:rho_c_plus ", noc);
    noc++;  /* rho_c_minus */
    fprintf(stdout, "%d:rho_c_minus ", noc);
    noc++;  /* compressive_strength */
    fprintf(stdout, "%d:compressive_strength ", noc);
    noc++;  /* tensile_strength */
    fprintf(stdout, "%d:tensile_strength ", noc);
    noc++;  /* bulk modulus K */
    fprintf(stdout, "%d:bulk modulus K ", noc);
    noc++;  /* flag_rho_0prime */
    fprintf(stdout, "%d:flag_rho_0prime ", noc);
    noc++;  /* flag_plastic */
    fprintf(stdout, "%d:flag_plastic ", noc);
    noc++;  /* shear_strength */
    fprintf(stdout, "%d:shear_strength ", noc);
#endif
#if PALPHA_POROSITY
    noc++; /* alpha_jutzi */
    fprintf(stdout, "%d:alpha_jutzi ", noc);
    noc++; /* pressure */
    fprintf(stdout, "%d:pressure ", noc);
#endif
#if EPSALPHA_POROSITY
    noc++; /* alpha_epspor */
    fprintf(stdout, "%d:alpha_epspor ", noc);
    noc++; /* epsilon_v */
    fprintf(stdout, "%d:epsilon_v ", noc);
#endif
#if FRAGMENTATION
    noc++;
    fprintf(stdout, "%d->%d+number of flaws:activation thresholds for this particle\n", noc, noc);
#endif
    fprintf(stdout, "\n");
#if HDF5IO
    fprintf(stdout, "output file format: (non-hdf5, for hdf5 use h5ls):\n");
#else
    fprintf(stdout, "output file format: only ascii since HDF5IO was not defined during compile time:\n");
#endif

    for (i = 0; i < DIM; i++)
        fprintf(stdout, "%d:x[%d] ", i+1, i);
    noc = DIM; /* x */
    for (i = noc, j = 0; i < noc+DIM; i++, j++)
        fprintf(stdout, "%d:v[%d] ", i+1, j);
    noc += DIM; /* v */
    noc++;  /* m */
    fprintf(stdout, "%d:mass ", noc);
    noc++;   /* rho */
    fprintf(stdout, "%d:density ", noc);
    noc++;   /* e */
#if INTEGRATE_ENERGY
    fprintf(stdout, "%d:energy ", noc);
    noc++; /* sml */
#endif
    fprintf(stdout, "%d:smoothing length ", noc);
    noc++; /* number of interaction partners */
    fprintf(stdout, "%d:number of interaction partners ", noc);
    noc++; /* material_type */
    fprintf(stdout, "%d:material type ", noc);
#if JC_PLASTICITY
    noc++;   /* ep */
    fprintf(stdout, "%d:strain ", noc);
    noc++;   /* T */
    fprintf(stdout, "%d:temperature ", noc);
#endif
#if FRAGMENTATION
    noc++; /* number of flaws */
    fprintf(stdout, "%d:number of flaws ", noc);
    noc++; /* number of activated flaws */
    fprintf(stdout, "%d:number of activated flaws ", noc);
    noc++; /* damage */
    fprintf(stdout, "%d:DIM-root of tensile damage ", noc);
#endif
#if !PALPHA_POROSITY
    noc++; /* pressure */
    fprintf(stdout, "%d:pressure ", noc);
#endif
#if SOLID
    noc++; /* local_strain  */
    fprintf(stdout, "%d:local_strain ", noc);
    k = noc+1;
    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            fprintf(stdout, "%d:S/sigma[%d][%d] ", k, i, j);
            k++;
        }
    }
    noc += DIM*DIM; /* S */
#endif
#if NAVIER_STOKES
    k = noc+1;
    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            fprintf(stdout, "%d:Tshear[%d][%d] ", k, i, j);
            k++;
        }
    }
    noc += DIM*DIM; /* Tshear */
#endif
#if SIRONO_POROSITY
    noc++;  /* rho_0prime */
    fprintf(stdout, "%d:rho_0prime ", noc);
    noc++;  /* rho_c_plus */
    fprintf(stdout, "%d:rho_c_plus ", noc);
    noc++;  /* rho_c_minus */
    fprintf(stdout, "%d:rho_c_minus ", noc);
    noc++;  /* compressive_strength */
    fprintf(stdout, "%d:compressive_strength ", noc);
    noc++;  /* tensile_strength */
    fprintf(stdout, "%d:tensile_strength ", noc);
    noc++;  /* bulk modulus K */
    fprintf(stdout, "%d:bulk modulus K ", noc);
    noc++;  /* flag_rho_0prime */
    fprintf(stdout, "%d:flag_rho_0prime ", noc);
    noc++;  /* flag_plastic */
    fprintf(stdout, "%d:flag_plastic ", noc);
    noc++;  /* shear_strength */
    fprintf(stdout, "%d:shear_strength ", noc);
#endif
#if PALPHA_POROSITY
    noc++; /* alpha_jutzi */
    fprintf(stdout, "%d:alpha_jutzi ", noc);
    noc++; /* pressure */
    fprintf(stdout, "%d:pressure ", noc);
#endif
#if EPSALPHA_POROSITY
    noc++; /* alpha_epspor */
    fprintf(stdout, "%d:alpha_epspor ", noc);
    noc++; /* epsilon_v */
    fprintf(stdout, "%d:epsilon_v ", noc);
#endif
#if FRAGMENTATION
    noc++;
    fprintf(stdout, "%d->%d+number of flaws:activation thresholds for this particle\n", noc, noc);
#endif
    fprintf(stdout, "\n");
    fprintf(stdout, "Additional information (such as time, momentum, energy, and angular momentum of the particle distribution is stored in <string.XXXX>.info\n");

#if GRAVITATING_POINT_MASSES
    fprintf(stdout, "Data file format for <string.XXXX>.mass\n");
    for (i = 0; i < DIM; i++) {
        fprintf(stdout, "%d:x[%d] ", i+1, i);
    }
    for (i = 0; i < DIM; i++) {
        fprintf(stdout, "%d:v[%d] ", i+DIM+1, i);
    }
    fprintf(stdout, "%d:mass %d:rmin %d:rmax", DIM+DIM+1, DIM+DIM+2, DIM+DIM+3);
    fprintf(stdout, "\n");
#if HDF5IO
    fprintf(stdout, "output file format for <string.XXXX>.mass: (non-hdf5, for hdf5 use h5ls):\n");
#else
    fprintf(stdout, "output file format for <string.XXXX>.mass: only ascii since HDF5IO was not defined during compile time:\n");
#endif
    for (i = 0; i < DIM; i++) {
        fprintf(stdout, "%d:x[%d] ", i+1, i);
    }
    for (i = 0; i < DIM; i++) {
        fprintf(stdout, "%d:v[%d] ", i+DIM+1, i);
    }
    fprintf(stdout, "%d:mass %d:rmin %d:rmax", DIM+DIM+1, DIM+DIM+2, DIM+DIM+3);
    fprintf(stdout, "\n");
#endif

}



void usage(char *name) {
    fprintf(stderr,
            "Usage %s [options]\n"
            "	sph program, version %s.\n"
            "Available options:\n"
            "\t-h, --help\t\t This message.\n"
            "\t-G, --information\t Print information about detected nvidia GPUs on this host.\n"
            "\t-d, --device_id <int> \t Try to use device with id <int> for computation (default: 0).\n"
            "\t-Y, --format\t\t Print information about input and output format of the data files,\n"
            "\t\t\t\t and about the compile time options of the binary.\n"
            "\t-v, --verbose\t\t Be talkative (stdout).\n"
            "\t-N, --numberofparticles\t Number of particles in input file.\n"
#if GRAVITATING_POINT_MASSES
            "\t-P, --nofpointmasses\t Number of pointmasses in mass input file.\n"
#endif
            "\t-I, --integrator\t Available Integrators are euler (1st order), euler_pc and monaghan_pc (2nd order),\n"
            "\t\t\t\t rk2_adaptive (2nd order with adaptive time step).\n"
            "\t-Q, --precision\t\t Precision of the rk2_adaptive integrator (default: 1e-6).\n"
            "\t-n, --num\t\t Number of simulation steps.\n"
#if HDF5IO
            "\t-H, --hdf5_output \t Use hdf5 for output (default is FALSE).\n"
            "\t-X, --hdf5_input \t Use hdf5 for input (default is FALSE), file 'string'.XXXX.h5 will be opened.\n"
#endif
            "\t-A, --no_ascii_output \t Disable ASCII output files (default is FALSE).\n"
            "\t-a, --theta\t\t Theta Criterion for Barnes-Hut Tree (default: 0.5)\n"
            "\t-t, --timeperstep\t Time for one simulation step.\n"
            "\t-M, --maxtimestep\t Upper limit for the timestep (rk2_integrator), timestep size for euler, respectively.\n"
            "\t-T, --starttime\t\t Start time of simulation.\n"
            "\t-f, --filename\t\t Name of input data file (default: disk.0000).\n"
            "\t\t\t\t Input data file name format is something like 'string'.XXXX, where\n"
            "\t\t\t\t XXXX means runlevel and zeros.\n"
            "\t-r, --restart\t\t Assume that ascii input file is old output file.\n"
            "\t-m, --materialconfig\t Name of config file including material config\n"
            "\t-k, --kernel\t\t use kernel function (default: cubic_spline)\n"
            "\t      \t\t\t possible values: wendlandc2, wendlandc4, wendlandc6, cubic_spline, quartic_spline, spiky.\n"
            "\t-s, --selfgravity\t Use selfgravity.\n"
            "\t-D, --directselfgravity\t Calculate selfgravity using direct particle-particle force and not the tree (slower).\n"
            "\t-g, --decouplegravity\t Decouple hydro time scale from gravitational time scale.\n"
            "\t-L, --angular_momentum <value> \t Check for conservation of angular momentum. (default: off)\n"
            "\t\t\t\t Simulations stops once the relative difference between current angular momentum and initial angular momentum is larger than <value>.\n"
            "\t-b, --boundary_ratio\t Ratio of additional ghost boundary particles (default: 0).\n"
            "Take a deep look at parameter.h. There you do necessary physical settings.\n"
            "Authors: Christoph Schaefer, Sven Riecker, Oliver Wandel, Samuel Scherrer.\n",
        name, VERSION);
    exit(0);
}

int main(int argc, char *argv[]) {
    // default run parameter
    param.performanceTest = FALSE;
    numberOfParticles = 0;
    numberOfPointmasses = 0;
    timePerStep = 1.0;
    startTime = 0.0;
    int wanted_device = 0;
    char configFile[255];
    strcpy(configFile, "material.cfg");
    // default integration scheme
    char integrationscheme[255] = "rk2_adaptive";

    static struct option opts[] = {
        { "verbose", 0, NULL, 'v' },
        { "restart", 0, NULL, 'r' },
        { "numberoftimesteps", 1, NULL, 'n' },
        { "device_id", 1, NULL, 'd' },
        { "numberofparticles", 1, NULL, 'N' },
        { "nofpointmasses", 1, NULL, 'P' },
        { "timeperstep", 1, NULL, 't' },
        { "maxtimestep", 1, NULL, 'M' },
        { "starttime", 1, NULL, 'T' },
        { "theta", 1, NULL, 'a' },
        { "precision", 1, NULL, 'Q' },
        { "hdf5_output", 0, NULL, 'H' },
        { "hdf5_input", 0, NULL, 'X' },
        { "no_ascii_output", 0, NULL, 'A' },
        { "decouplegravity", 0, NULL, 'g' },
        { "format", 0, NULL, 'Y' },
        { "filename", 1,	NULL, 'f' },
        { "angular_momentum", 1,	NULL, 'L' },
        { "kernel", 1,	NULL, 'k' },
        { "materialconfig", 1, NULL, 'm'},
        { "selfgravity", 0, NULL, 's' },
        { "directselfgravity", 0, NULL, 'D' },
        { "help", 0, NULL, 'h' },
        { "information", 0, NULL, 'G' },
        { "integrator", 1, NULL, 'I' },
        { "boundary_ratio", 0, NULL, 'b'},
        { NULL, 0, 0, 0 }
    };

    if (argc == 1) {
        usage(argv[0]);
    }

    param.hdf5input = FALSE;
    param.hdf5output = FALSE;
    param.restart = FALSE;
    param.ascii_output = TRUE;
    param.maxtimestep = -1;
    param.rk_epsrel = 1e-6;
    param.angular_momentum_check = -1.0;
    strcpy(param.kernel, "cubic_spline");
    param.boundary_ratio = 0;

    treeTheta = 0.5; // default theta
    param.selfgravity = FALSE;
    param.directselfgravity = FALSE;
    param.decouplegravity = 0;

#if USE_SIGNAL_HANDLER
    signal(SIGINT, signal_handler);
#endif

    int i, c;
    while ((c = getopt_long(argc, argv, "P:Q:d:M:b:m:N:L:k:T:DI:t:a:n:f:b:rXYvhHshVgGA", opts, &i)) != -1) {
        switch (c) {
            case 'M':
                param.maxtimestep = atof(optarg);
                if (param.maxtimestep < 0) {
                    fprintf(stderr, "Error. Maximum possible timestep should be > 0.\n");
                    exit(1);
                }
                break;
            case 'Q':
                param.rk_epsrel = atof(optarg);
                if (param.rk_epsrel < 0 || param.rk_epsrel >= 1) {
                    fprintf(stderr, "Error. Accuracy of the rk2 integrator should be 0 < rk_epsrel < 1.");
                    exit(1);
                }
                break;
            case 'G':
                printfDeviceInformation();
                exit(0);
                break;
            case 'd':
                wanted_device = atoi(optarg);
                printfDeviceInformation();
                fprintf(stdout, "Trying to use CUDA device %d\n", wanted_device);
                cudaSetDevice(wanted_device);
                break;
            case 'g':
                param.decouplegravity = 1;
                break;
            case 'A':
                param.ascii_output = FALSE;
                break;
            case 'P':
                numberOfPointmasses = atoi(optarg);
                if (numberOfPointmasses < 0) {
                    fprintf(stderr,
                            "Grmpf. Something's wrong with the number of point masses.\n ");
                    exit(1);
                }
                break;
            case 'N':
                numberOfParticles = atoi(optarg);
                if (numberOfParticles < 0) {
                    fprintf(stderr,
                            "Grmpf. Something's wrong with the number of particles.\n ");
                    exit(1);
                }
                break;
            case 'L':
                param.angular_momentum_check = atof(optarg);
                if (param.angular_momentum_check < 0) {
                    fprintf(stderr, "angular_momentum value should be > 0.\n");
                    exit(1);
                }
                break;
            case 'b':
                param.boundary_ratio = atof(optarg);
                if (param.boundary_ratio < 0) {
                    fprintf(stderr, "Boundary particle ratio should be positive.\n");
                    exit(1);
                }
                break;
            case 'a':
                treeTheta = atof(optarg);
                param.selfgravity = TRUE;
                if (treeTheta < 0 || treeTheta >= 1) {
                    fprintf(stderr, "Er? Check theta.\n");
                    exit(1);
                }
                break;
            case 'v':
                param.verbose = TRUE;
                break;
            case 'r':
                param.restart = TRUE;
                break;
            case 'T':
                startTime = atof(optarg);
                if (startTime < 0) {
                    fprintf(stderr, "Hm? Negative start time?\n");
                    exit(1);
                }
                break;
            case 't':
                timePerStep = atof(optarg);
                if (timePerStep < 0) {
                    fprintf(stderr, "Huh? Check time per step.\n");
                    exit(1);
                }
                break;
            case 'f':
                if (!strcpy(inputFile.name, optarg))
                    exit(1);
                break;
            case 'n':
                numberOfTimesteps = atoi(optarg);
                if (numberOfTimesteps < 0) {
                    fprintf(stderr, "Invalid number of simulation steps.\n");
                    exit(1);
                }
                break;
            case 'm':
                if (!strcpy(configFile, optarg)) {
                    fprintf(stderr, "Something wrong with material config file.");
                    exit(1);
                }
                break;
            case 's':
                param.selfgravity = TRUE;
                break;
            case 'D':
                param.directselfgravity = TRUE;
                break;
            case 'H':
                param.hdf5output = TRUE;
                break;
            case 'X':
                param.hdf5input = TRUE;
                break;
            case 'I':
                if (!strcpy(integrationscheme, optarg)) {
                    fprintf(stderr, "Something's wrong with the integrator name.\n");
                    exit(1);
                }
                break;
            case 'k':
                if (!strcpy(param.kernel, optarg)) {
                    fprintf(stderr, "Something's wrong with the kernel function.\n");
                    exit(1);
                }
                break;
            case 'Y':
                format_information(argv[0]);
                print_compile_information();
                exit(0);
            case 'h':
                usage(argv[0]);
                exit(0);
            default:
                usage(argv[0]);
                exit(0);
        }
    }

    maxNumberOfParticles = (int) ( (1+param.boundary_ratio) * numberOfParticles);
    numberOfRealParticles = numberOfParticles;

    if (param.verbose) {
        print_compile_information();
    }

    if (param.selfgravity && param.directselfgravity) {
        fprintf(stderr, "Warning: both selfgravity and directselfgravity parameters are set.\n");
        fprintf(stderr, "unsetting selfgravity and using directselfgravity.\n");
        param.selfgravity = FALSE;
    }


    // check for plasticity model
#if VON_MISES_PLASTICITY && JC_PLASTICITY
    fprintf(stderr, "Error: Can't use both Von Mises and Johnson-Cook Plasticity Models at the same time. Decide for one and recompile.\n");
    exit(1);
#endif

    // choose integrator
    fprintf(stdout, "Integrator information\n");
    if (0 == strcmp(integrationscheme, "rk2_adaptive")) {
        fprintf(stdout, "using rk2 adaptive\n");
        integrator = &rk2Adaptive;
        param.integrator_type = RK2_ADAPTIVE;
        printf("with accurary rk_epsrel: %g\n", param.rk_epsrel);
    } else if (0 == strcmp(integrationscheme, "euler")) {
        fprintf(stdout, "using euler\n");
        integrator = &euler;
        param.integrator_type = EULER;
    } else if (0 == strcmp(integrationscheme, "monaghan_pc")) {
        fprintf(stdout, "using monaghan_pc\n");
        integrator = &predictor_corrector;
        param.integrator_type = MONAGHAN_PC;
    } else if (0 == strcmp(integrationscheme, "euler_pc")) {
        fprintf(stdout, "using euler_pc\n");
        integrator = &predictor_corrector_euler;
        param.integrator_type = EULER_PC;
    } else {
        fprintf(stderr, "Err. No such integration scheme implemented yet.\n");
        exit(1);
    }

    // choose SPH kernel
    fprintf(stdout, "SPH kernel information\t");
    if (0 == strcmp(param.kernel, "wendlandc2")) {
        fprintf(stdout, "using wendlandc2 kernel\n");
        cudaMemcpyFromSymbol(&kernel_h, wendlandc2_p, sizeof(SPH_kernel));
        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
    } else if (0 == strcmp(param.kernel, "wendlandc4")) {
        fprintf(stdout, "using wendlandc4 kernel\n");
        cudaMemcpyFromSymbol(&kernel_h, wendlandc4_p, sizeof(SPH_kernel));
        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
    } else if (0 == strcmp(param.kernel, "wendlandc6")) {
        fprintf(stdout, "using wendlandc6 kernel\n");
        cudaMemcpyFromSymbol(&kernel_h, wendlandc6_p, sizeof(SPH_kernel));
        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
    } else if (0 == strcmp(param.kernel, "cubic_spline")) {
        fprintf(stdout, "using cubic_spline kernel\n");
        cudaMemcpyFromSymbol(&kernel_h, cubic_spline_p, sizeof(SPH_kernel));
        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
    } else if (0 == strcmp(param.kernel, "spiky")) {
        fprintf(stdout, "using spiky kernel\n");
        cudaMemcpyFromSymbol(&kernel_h, spiky_p, sizeof(SPH_kernel));
        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
    } else if (0 == strcmp(param.kernel, "quartic_spline")) {
        fprintf(stdout, "using quartic_spline kernel\n");
        cudaMemcpyFromSymbol(&kernel_h, quartic_spline_p, sizeof(SPH_kernel));
        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
    } else {
        fprintf(stderr, "Err. No such kernel function implemented yet: %s.\n", param.kernel);
        exit(1);
    }

    // print out selfgravity information
    fprintf(stdout, "Self gravity information\t");
    if (param.selfgravity) {
        fprintf(stdout, "calculating selfgravity with Barnes Hut tree with theta: %g\n", treeTheta);
    } else if (param.directselfgravity) {
        fprintf(stdout, "calculating selfgravity using direct particle-particle force.\n");
    } else {
        fprintf(stdout, "neglecting selfgravity.\n");
    }


    if (param.maxtimestep < 0)
        param.maxtimestep = timePerStep;

    if (param.verbose) printf("loading config file...\n");

    loadConfigFromFile(configFile);

    if (param.verbose) printf("clearing performance file...\n");

    if (param.performanceTest) clear_performance_file();

    if (param.verbose) printf("N = %d\n", numberOfParticles);
    if (param.verbose) printf("Allocating memory for %d particles\n", maxNumberOfParticles);

    if (param.verbose) printf("checking for cuda devices...\n");

    // check cuda
    cudaDeviceProp deviceProp;
    int cnt;
    cudaVerify(cudaGetDeviceProperties(&deviceProp, wanted_device));
    cudaGetDeviceCount(&cnt);
    if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
        fprintf(stderr, "There is no CUDA capable device\n");
        exit(-1);
    }
    fprintf(stdout, "Found compute capability %d.%d\n", deviceProp.major, deviceProp.minor);
    fprintf(stdout, "Need at least compute capability 2.0\n");
    fprintf(stdout, "Found #gpus %d: %s\n", cnt, deviceProp.name);
    numberOfMultiprocessors = deviceProp.multiProcessorCount;
    if (param.verbose) printf("found cuda device with %d multiprocessors.\n", numberOfMultiprocessors);


    /* initialise the memory */
    init_allocate_memory();


    // read particle data from input file
    if (param.verbose) printf("reading input file %s ...\n", inputFile.name);
    if ((inputFile.data = fopen(inputFile.name, "r")) == NULL) {
        fprintf(stderr, "Wtf? File %s not found.\n", inputFile.name);
        if (param.hdf5input) {
            fprintf(stderr, "Hope you know what you're up to and search for a h5 file\n");
            read_particles_from_file(inputFile);
        } else {
            exit(1);
        }
    } else {
        read_particles_from_file(inputFile);
        fclose(inputFile.data);
    }

    /* init some values */
    init_values();
    // copy the particles to the gpu
    copy_particle_data_to_device();
    if (cudaSuccess != cudaMemcpyToSymbol(childList, &childListd, sizeof(void*))) {
        fprintf(stderr, "copying of childList to device failed\n");
        exit(1);
    }

    /* if hdf5 output is enabled and no hdf5 input is set, write the ascii input file to hdf5 */
    if (param.hdf5output && !param.hdf5input) {
        if (param.verbose) {
            fprintf(stdout, "Writing input ascii file to hdf5 %s.h5\n", inputFile.name);
        }
        int asciiflag = param.ascii_output;
        param.ascii_output = 0;
        h5time = startTime;
        write_particles_to_file(inputFile);
        param.ascii_output = asciiflag;
    }

    if (param.verbose) {
        printf("Simulation time start: %g\n", startTime);
    }

    if (param.verbose) printf("starting time integration...\n\n");
    cudaProfilerStart();
    timeIntegration();
    cudaProfilerStop();

    /* free memory */
    if (param.verbose) printf("freeing memory\n");
    free_memory();

    if (param.verbose) printf("resetting GPU...\n");
    cudaVerify(cudaDeviceReset());


    if (param.verbose) printf("kthxbye.\n");


    return 0;

}
