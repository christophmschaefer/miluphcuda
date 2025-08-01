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
#include "rk2adaptive.h"
#include <cuda_runtime.h>

#if HDF5IO
#include <hdf5.h>
#endif

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
struct Pointmass rk4_pointmass_device[4];
__constant__ struct Pointmass rk4_pointmass[4];
struct Pointmass predictor_pointmass_device;
__constant__ struct Pointmass predictor_pointmass;
__constant__ struct Pointmass pointmass_rhs;
int numberOfPointmasses;
size_t memorySizeForPointmasses;
size_t integermemorySizeForPointmasses;


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

size_t memorySizeForTree;
size_t memorySizeForParticles;
size_t memorySizeForInteractions;
size_t memorySizeForChildren;
size_t memorySizeForStress;
#if FRAGMENTATION
size_t memorySizeForActivationThreshold;
#endif

int numberOfMultiprocessors;

int numberOfChildren = pow(2, DIM);
// size_t numberOfNodes;
int numberOfNodes;


// the sph-kernel function pointers
extern __device__ SPH_kernel kernel;
extern __device__ SPH_kernel wendlandc2_p;
extern __device__ SPH_kernel wendlandc4_p;
extern __device__ SPH_kernel wendlandc6_p;
extern __device__ SPH_kernel cubic_spline_p;
extern __device__ SPH_kernel spiky_p;
SPH_kernel kernel_h;



/* print info about physical model */
static void print_compile_information(void)
{
    char yesno[10];

    fprintf(stdout, "\n");
    fprintf(stdout, "Number of dimensions: %d\n", DIM);
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
#if FRAGMENTATION
# if DAMAGE_ACTS_ON_S
    strcpy(yesno, "yes");
# else
    strcpy(yesno, "no");
# endif
    fprintf(stdout, "Damage acts on S tensor:\t  %s\n", yesno);
#endif

#if PALPHA_POROSITY
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "P-alpha porosity model:\t  %s\n", yesno);
#if STRESS_PALPHA_POROSITY
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "P-alpha model (distention) affects S tensor:\t  %s\n", yesno);
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

#if PLASTICITY
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Plasticity model:\t  %s\n", yesno);
#if MOHR_COULOMB_PLASTICITY
    fprintf(stdout, "    Mohr-Coulomb");
# if VON_MISES_PLASTICITY
    fprintf(stdout, " + von Mises yield limit");
# endif
    fprintf(stdout, "\n");
#elif DRUCKER_PRAGER_PLASTICITY
    fprintf(stdout, "    Drucker-Prager");
# if VON_MISES_PLASTICITY
    fprintf(stdout, " + von Mises yield limit");
# endif
    fprintf(stdout, "\n");
#elif COLLINS_PLASTICITY
    fprintf(stdout, "    Collins model: pressure dependent yield strength with friction model for damaged material");
# if COLLINS_PLASTICITY_INCLUDE_MELT_ENERGY
    fprintf(stdout, " + strength reduction based on (single) melt energy");
# endif
    fprintf(stdout, "\n");
#elif COLLINS_PLASTICITY_SIMPLE
    fprintf(stdout, "    simplified version of Collins model: only Lundborg yield strength curve + simple negative-pressure cap (i.e., no dynamic fragmentation/damage)\n");
#elif VON_MISES_PLASTICITY
    fprintf(stdout, "    simple von Mises yield strength, independent of pressure\n");
#endif

#if JC_PLASTICITY
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Plasticity model from Johnson-Cook:\t   %s\n", yesno);

#if LOW_DENSITY_WEAKENING
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Low-density weakening model:\t   %s\n", yesno);

    fprintf(stdout, "Consistency switches for the SPH algorithm:\n");
#if SHEPARD_CORRECTION
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "SPH zeroth order consistency (aka Shepard correction):\t  %s\n", yesno);
#if TENSORIAL_CORRECTION
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "SPH linear consistency (aka tensorial correction):\t  %s\n", yesno);
#if ARTIFICIAL_VISCOSITY
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Using SPH representation of momentum and energy conservation version: \t"
#if SPH_EQU_VERSION == 1
                        "1"
#elif SPH_EQU_VERSION == 2
                        "2"
#endif
                        "\n");
    fprintf(stdout, "Standard SPH artificial viscosity:\t  %s\n", yesno);
#if XSPH
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "XSPH velocity smoothing:\t  %s\n", yesno);
#if READ_INITIAL_SML_FROM_PARTICLE_FILE
    fprintf(stdout, "Initial smoothing lengths: read from input file for each particle.\n");
#else
    fprintf(stdout, "Initial smoothing lengths: use a single one per material.\n");
#endif
#if VARIABLE_SML
    fprintf(stdout, "Using (time) variable smoothing lengths:\t yes\n");
# if FIXED_NOI
    fprintf(stdout, "    with fixed number of interaction partners\n");
# elif INTEGRATE_SML
    fprintf(stdout, "    with integration of the smoothing length\n");
# endif
#else
    fprintf(stdout, "Using (time) variable smoothing lengths:\t no\n");
#endif
#if AVERAGE_KERNELS
    fprintf(stdout, "Kernel for interaction is calculated by averaging kernels for each particle: \t W_ij = 0.5 ( W(h_i) + W(h_j) )\n");
#else
    fprintf(stdout, "Kernel for interaction is calculated using averaged smoothing length: \t W_ij = W(0.5 (h_i + h_j))\n");
#endif
#if GHOST_BOUNDARIES
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "Boundary conditions:\n");
    fprintf(stdout, "    ghost boundaries:\t  %s\n", yesno);
    fprintf(stdout, "    boundary particle id: %d\n", BOUNDARY_PARTICLE_ID);

    fprintf(stdout, "I/O settings:\n");
#if HDF5IO
    strcpy(yesno, "yes");
#else
    strcpy(yesno, "no");
#endif
    fprintf(stdout, "    HDF5 i/o:\t  %s\n", yesno);
    if (param.hdf5output) {
        strcpy(yesno, "yes");
    } else {
        strcpy(yesno, "no");
    }
    fprintf(stdout, "    using HDF5 output: \t %s \n", yesno);
    if (param.hdf5input) {
        strcpy(yesno, "yes");
    } else {
        strcpy(yesno, "no");
    }
    fprintf(stdout, "    using HDF5 input: \t %s \n", yesno);
    if (param.ascii_output) {
        strcpy(yesno, "yes");
    } else {
        strcpy(yesno, "no");
    }
    fprintf(stdout, "    using ASCII output: \t %s \n", yesno);

    fprintf(stdout, "\nImplemented equations of state and corresponding eos type entry in material.cfg:\n");
    fprintf(stdout, "EOS_TYPE_IGNORE          \t\t\t %d\n", EOS_TYPE_IGNORE);
    fprintf(stdout, "EOS_TYPE_POLYTROPIC_GAS  \t\t\t %d\n", EOS_TYPE_POLYTROPIC_GAS);
    fprintf(stdout, "EOS_TYPE_MURNAGHAN       \t\t\t %d\n", EOS_TYPE_MURNAGHAN);
    fprintf(stdout, "EOS_TYPE_TILLOTSON       \t\t\t %d\n", EOS_TYPE_TILLOTSON);
    fprintf(stdout, "EOS_TYPE_ISOTHERMAL_GAS  \t\t\t %d\n", EOS_TYPE_ISOTHERMAL_GAS);
    fprintf(stdout, "EOS_TYPE_REGOLITH        \t\t\t %d\n", EOS_TYPE_REGOLITH);
    fprintf(stdout, "EOS_TYPE_JUTZI           \t\t\t %d\n", EOS_TYPE_JUTZI);
    fprintf(stdout, "EOS_TYPE_JUTZI_MURNAGHAN \t\t\t %d\n", EOS_TYPE_JUTZI_MURNAGHAN);
	fprintf(stdout, "EOS_TYPE_JUTZI_ANEOS	  \t\t\t %d\n", EOS_TYPE_JUTZI_ANEOS);
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
    fprintf(stdout, "output file format: (non-HDF5, for HDF5 use h5ls):\n");
#else
    fprintf(stdout, "output file format: only ASCII since HDF5IO was not defined during compile time:\n");
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
    fprintf(stdout, "output file format for <string.XXXX>.mass: (non-HDF5, for HDF5 use h5ls):\n");
#else
    fprintf(stdout, "output file format for <string.XXXX>.mass: only ASCII since HDF5IO was not defined during compile time:\n");
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



void usage(char *name)
{
    fprintf(stderr,
            "\nmiluphcuda is a multi-rheology, multi-material SPH code, developed mainly for astrophysical applications.\n"
            "This is version %s.\n\n"
            "Usage:\n"
            "\t%s [options]\n\n"
            "Best options:\n"
            "\t-h, --help\t\t\t This message.\n"
            "\t-v, --verbose\t\t\t Be talkative (stdout).\n\n"
            "Available options:\n"
            "\t-a, --theta <value>\t\t Theta Criterion for Barnes-Hut Tree (default: 0.5).\n"
            "\t-A, --no_ascii_output \t\t Disable ASCII output files (default: not set).\n"
            "\t-b, --boundary_ratio <value>\t Ratio of additional ghost boundary particles (default: 0).\n"
            "\t-c, --cons_qu_file <name>\t Name of logfile for conserved quantities (default: conserved_quantities.log).\n"
            "\t-d, --device_id <int> \t\t Try to use device with id <int> for computation (default: 0).\n"
            "\t-D, --directselfgravity\t\t Calculate selfgravity using direct particle-particle force and not the tree (slower).\n"
            "\t-f, --filename <name>\t\t Name of input data file (default: disk.0000).\n"
            "\t\t\t\t\t File name format is something like 'string'.XXXX, where XXXX means runlevel and zeros.\n"
            "\t\t\t\t\t By default, an ASCII input file is assumed (unless -X is used).\n"
            "\t-F, --firsttimestep\t\t For rk2_adaptive, set initial timestep at integration start\n"
            "\t\t\t\t\t (default: maxtimestep, set by -M, or otherwise timeperstep, set by -t).\n"
            "\t-g, --decouplegravity\t\t Decouple hydro time scale from gravitational time scale.\n"
            "\t-G, --information\t\t Print information about detected Nvidia GPUs.\n"
#if HDF5IO
            "\t-H, --hdf5_output \t\t Use HDF5 for output (default: not set).\n"
            "\t\t\t\t\t If set, HDF5 files are produced in addition to ASCII files. Use -A to get only HDF5 output.\n"
#endif
            "\t-I, --integrator <name>\t\t Available options are 'euler' (1st order), 'euler_pc' and 'monaghan_pc' (2nd order),\n"
            "\t\t\t\t\t 'rk2_adaptive' (default, 2nd order with adaptive time step),\n"
            "\t\t\t\t\t 'heun_rk4' (2nd order for sph coupled with fourth order for n-body).\n"
            "\t-k, --kernel <name>\t\t Set kernel function (default: 'cubic_spline').\n"
            "\t      \t\t\t\t Options: wendlandc2, wendlandc4, wendlandc6, cubic_spline, spiky.\n"
            "\t-L, --angular_momentum <value> \t Check for conservation of angular momentum (default: off).\n"
            "\t\t\t\t\t Simulation stops once the relative difference between current and initial angular momentum is larger than <value>.\n"
            "\t-m, --materialconfig <name>\t Name of libconfig file including material config (default: material.cfg)\n"
            "\t-M, --maxtimestep <value>\t Upper limit for the timestep (rk2_adaptive), or timestep size (euler), respectively.\n"
            "\t-n, --num <int>\t\t\t Number of simulation steps (additional ones in case of restart).\n"
            "\t-Q, --precision <value>\t\t Precision of the rk2_adaptive integrator (default: 1e-5).\n"
            "\t-r, --restart\t\t\t Assume that ASCII input file (specified with -f) is old output file.\n"
            "\t\t\t\t\t To restart from old HDF5 output file use -X instead of this.\n"
            "\t-s, --selfgravity\t\t Activate selfgravity.\n"
            "\t\t\t\t\t Selfgravity is computed via tree by default (unless -D is used).\n"
            "\t-t, --timeperstep <value>\t Time for one simulation step.\n"
            "\t-T, --starttime <value>\t\t Start time of simulation.\n"
#if HDF5IO
            "\t-X, --hdf5_input \t\t Use HDF5 file for input (default: off).\n"
            "\t\t\t\t\t The filename itself - but without the .h5 ending (!) - has to be set with -f.\n"
#endif
            "\t-Y, --format\t\t\t Print information about input and output format of the data files,\n"
            "\t\t\t\t\t and about the compile time options of the binary.\n\n"
            "Take a deep look at parameter.h. There you find most of the physics and numerics settings.\n\n"
            "More information on github: https://github.com/christophmschaefer/miluphcuda\n\n",
        MILUPHCUDA_VERSION, name);
    exit(0);
}



int main(int argc, char *argv[])
{
    numberOfParticles = 0;
    numberOfPointmasses = 0;
    timePerStep = 1.0;
    startTime = 0.0;
    int wanted_device = 0;
    char configFile[255];
    strcpy(configFile, "material.cfg");
    // default integration scheme
    char integrationscheme[255] = "rk2_adaptive";
    FILE *conservedquantitiesfile;
    FILE *binarysystemfile;

    static struct option opts[] = {
        { "verbose", 0, NULL, 'v' },
        { "restart", 0, NULL, 'r' },
        { "numberoftimesteps", 1, NULL, 'n' },
        { "device_id", 1, NULL, 'd' },
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
        { "firsttimestep", 1,	NULL, 'F' },
        { "cons_qu_file", 1, NULL, 'c' },
        { "angular_momentum", 1,	NULL, 'L' },
        { "kernel", 1,	NULL, 'k' },
        { "materialconfig", 1, NULL, 'm' },
        { "selfgravity", 0, NULL, 's' },
        { "directselfgravity", 0, NULL, 'D' },
        { "help", 0, NULL, 'h' },
        { "information", 0, NULL, 'G' },
        { "integrator", 1, NULL, 'I' },
        { "boundary_ratio", 0, NULL, 'b' },
        { NULL, 0, 0, 0 }
    };

    if (argc == 1) {
        usage(argv[0]);
    }

    // default run parameter
    param.verbose = FALSE;
    param.hdf5input = FALSE;
    param.hdf5output = FALSE;
    param.restart = FALSE;
    param.ascii_output = TRUE;
    param.maxtimestep = -1.0;
    param.firsttimestep = -1.0;
    param.rk_epsrel = 1e-5;
    param.angular_momentum_check = -1.0;
    strcpy(inputFile.name, "disk.0000");
    strcpy(configFile, "material.cfg");
    strcpy(param.kernel, "cubic_spline");
    strcpy(param.conservedquantitiesfilename, "conserved_quantities.log");
    strcpy(param.binarysystemfilename, "binary_system.log");
    param.boundary_ratio = 0;
    treeTheta = 0.5; // default theta
    param.selfgravity = FALSE;
    param.directselfgravity = FALSE;
    param.decouplegravity = 0;
    param.performanceTest = FALSE;

#if USE_SIGNAL_HANDLER
    signal(SIGINT, signal_handler);
#endif

    int i, c;
    while ((c = getopt_long(argc, argv, "Q:d:M:b:m:L:k:T:DI:t:a:n:f:F:c:e:b:rXYvhHshVgGA", opts, &i)) != -1) {
        switch (c) {
            case 'F':
                param.firsttimestep = atof(optarg);
                if (param.firsttimestep < 0) {
                    fprintf(stderr, "Error. First timestep should be > 0.\n");
                    exit(1);
                }
                break;
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
                fprintf(stdout, "Trying to use CUDA device %d\n", wanted_device);
                cudaSetDevice(wanted_device);
                break;
            case 'g':
                param.decouplegravity = TRUE;
                break;
            case 'A':
                param.ascii_output = FALSE;
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
                if (!strcpy(inputFile.name, optarg)) {
                    fprintf(stderr, "Something's wrong with the input file.\n");
                    exit(1);
                }
                break;
            case 'c':
                if( !strcpy(param.conservedquantitiesfilename, optarg) ) {
                    fprintf(stderr, "Something's wrong with the name of the logfile for conserved quantities.\n");
                    exit(1);
                }
                break;
            case 'e':
                if( !strcpy(param.binarysystemfilename, optarg) ) {
                    fprintf(stderr, "Something's wrong with the name of the logfile for binary system.\n");
                    exit(1);
                }
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
                exit(1);
        }
    }

    // print device information
    if (param.verbose)
        printfDeviceInformation();

    // get the information about the number of particles in the file
    if ((inputFile.data = fopen(inputFile.name, "r")) == NULL) {
        if (param.hdf5input) {
#if HDF5IO
            char h5filename[256];
            strcpy(h5filename, inputFile.name);
            strcat(h5filename, ".h5");

            hid_t file_id = H5Fopen (h5filename, H5F_ACC_RDONLY, H5P_DEFAULT);
            if (file_id < 0) {
                fprintf(stderr, "********************** Error opening file %s\n", h5filename);
                exit(1);
            } else {
                fprintf(stdout, "\nUsing HDF5 input file %s.\n", h5filename);
            }

            /* open the dataset for the positions */
            hid_t x_id = H5Dopen(file_id, "/x", H5P_DEFAULT);
            if (x_id < 0) {
                fprintf(stderr, "Could not find locations in HDF5 file. Exiting...\n");
            }
            /* determine number of particles stored in HDF5 file */
            hid_t dspace = H5Dget_space(x_id);
            const int ndims = H5Sget_simple_extent_ndims(dspace);
            hsize_t dims[ndims];
            H5Sget_simple_extent_dims(dspace, dims, NULL);
            int my_anop = dims[0];
            fprintf(stdout, "Found %d particles in %s.\n", my_anop, h5filename);
            numberOfParticles = my_anop;
            H5Fclose(file_id);
#endif
        } else {
            fprintf(stderr, "Error: File %s not found.\n", inputFile.name);
            exit(1);
        }
    } else {
        // reading number of lines in file
        int count = 0;
        int datacnt = 0;
        char c;
        for (c = getc(inputFile.data); c != EOF; c = getc(inputFile.data)) {
            if (c == '\n') {
                if (datacnt == 0) {
                    fprintf(stderr, "Error, found empty line in inputfile %s. This does not work.\n", inputFile.name);
                    exit(1);
                }
                datacnt = 0;
                count++;
            } else {
                datacnt++;
            }
        }
        fprintf(stdout, "\nFound %d particles in %s.\n", count, inputFile.name);
        fclose(inputFile.data);
        numberOfParticles = count;
    }

#if GRAVITATING_POINT_MASSES
    // get the information about the number of particles in mass file
    char massfilename[256];
    FILE *inputf;
    strcpy(massfilename, inputFile.name);
    strcat(massfilename, ".mass");

    if ((inputf = fopen(massfilename, "r")) == NULL) {
        if (param.hdf5input) {
# if HDF5IO
            char h5filename[256];
            strcpy(h5filename, inputFile.name);
            strcat(h5filename, ".mass.h5");

            hid_t file_id = H5Fopen (h5filename, H5F_ACC_RDONLY, H5P_DEFAULT);
            if (file_id < 0) {
                fprintf(stderr, "********************** Error opening file %s\n", h5filename);
                exit(1);
            } else {
                fprintf(stdout, "Using HDF5 input file %s.\n", h5filename);
            }

            /* open the dataset for the positions */
            hid_t x_id = H5Dopen(file_id, "/x", H5P_DEFAULT);
            if (x_id < 0) {
                fprintf(stderr, "Could not find locations in HDF5 file. Exiting...\n");
            }
            /* determine number of particles stored in HDF5 file */
            hid_t dspace = H5Dget_space(x_id);
            const int ndims = H5Sget_simple_extent_ndims(dspace);
            hsize_t dims[ndims];
            H5Sget_simple_extent_dims(dspace, dims, NULL);
            int my_anop = dims[0];
            fprintf(stdout, "Found %d point masses in %s.\n", my_anop, h5filename);
            numberOfPointmasses = my_anop;
            H5Fclose(file_id);
# endif
        } else {
            fprintf(stderr, "File for the point masses %s not found.\n", massfilename);
            exit(1);
        }
    } else {
        // reading number of lines in file
        int count = 0;
        char c;
        for (c = getc(inputf); c != EOF; c = getc(inputf)) {
            if (c == '\n') {
                count++;
            }
        }
        fprintf(stdout, "Found %d point masses in %s.\n", count, massfilename);
        fclose(inputf);
        numberOfPointmasses = count;
    }

# if BINARY_INFO
    /* create binary system file and write header */
    if(param.hdf5input){
        if( (binarysystemfile = fopen(param.binarysystemfilename, "a")) == NULL ) {
            fprintf(stderr, "Cannot open '%s' for writing. Abort...\n", param.binarysystemfilename);
            exit(1);
        }
    }
    else {
        if( (binarysystemfile = fopen(param.binarysystemfilename, "w")) == NULL ) {
            fprintf(stderr, "Cannot open '%s' for writing. Abort...\n", param.binarysystemfilename);
            exit(1);
        }
        fprintf(binarysystemfile, "#         1.time            2.semi-major-axis   3.eccentricity       4.Binary angular momentum");
        fprintf(binarysystemfile, "\n");
    }
    fclose(binarysystemfile);    
# endif

#endif // GRAVITATING_POINT_MASSES

    maxNumberOfParticles = (int) ( (1+param.boundary_ratio) * numberOfParticles);
    numberOfRealParticles = numberOfParticles;

    print_compile_information();

    if (param.selfgravity && param.directselfgravity) {
        fprintf(stderr, "Warning: both selfgravity and directselfgravity parameters are set.\n");
        fprintf(stderr, "Unsetting selfgravity and using directselfgravity.\n");
        param.selfgravity = FALSE;
    }

    // choose integrator
    fprintf(stdout, "\nTime integrator: ");
    if (0 == strcmp(integrationscheme, "rk2_adaptive")) {
        integrator = &rk2Adaptive;
        param.integrator_type = RK2_ADAPTIVE;
        fprintf(stdout, "rk2_adaptive - with accuracy rk_epsrel: %g\n", param.rk_epsrel);
    } else if (0 == strcmp(integrationscheme, "euler")) {
        fprintf(stdout, "euler\n");
        integrator = &euler;
        param.integrator_type = EULER;
    } else if (0 == strcmp(integrationscheme, "monaghan_pc")) {
        fprintf(stdout, "monaghan_pc\n");
        integrator = &predictor_corrector;
        param.integrator_type = MONAGHAN_PC;
    } else if (0 == strcmp(integrationscheme, "heun_rk4")) {
        fprintf(stdout, "heun_rk4\n");
        integrator = &heun_rk4;
        param.integrator_type = HEUN_RK4;
    } else if (0 == strcmp(integrationscheme, "euler_pc")) {
        fprintf(stdout, "euler_pc\n");
        integrator = &predictor_corrector_euler;
        param.integrator_type = EULER_PC;
    } else {
        fprintf(stderr, "Err. No such time integration scheme implemented yet.\n");
        exit(1);
    }

    // choose SPH kernel
    fprintf(stdout, "SPH kernel: ");
    if (0 == strcmp(param.kernel, "wendlandc2")) {
        fprintf(stdout, "wendlandc2\n");
        cudaMemcpyFromSymbol(&kernel_h, wendlandc2_p, sizeof(SPH_kernel));
        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
    } else if (0 == strcmp(param.kernel, "wendlandc4")) {
        fprintf(stdout, "wendlandc4\n");
        cudaMemcpyFromSymbol(&kernel_h, wendlandc4_p, sizeof(SPH_kernel));
        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
    } else if (0 == strcmp(param.kernel, "wendlandc6")) {
        fprintf(stdout, "wendlandc6\n");
        cudaMemcpyFromSymbol(&kernel_h, wendlandc6_p, sizeof(SPH_kernel));
        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
    } else if (0 == strcmp(param.kernel, "cubic_spline")) {
        fprintf(stdout, "cubic_spline\n");
        cudaMemcpyFromSymbol(&kernel_h, cubic_spline_p, sizeof(SPH_kernel));
        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
    } else if (0 == strcmp(param.kernel, "spiky")) {
        fprintf(stdout, "spiky\n");
        cudaMemcpyFromSymbol(&kernel_h, spiky_p, sizeof(SPH_kernel));
        cudaMemcpyToSymbol(kernel, &kernel_h, sizeof(SPH_kernel));
    } else {
        fprintf(stderr, "Err. No such kernel function implemented yet: %s.\n", param.kernel);
        exit(1);
    }

    // print out self-gravity information
    fprintf(stdout, "Self-gravity: ");
    if (param.selfgravity) {
        fprintf(stdout, "yes - calculated with Barnes Hut tree with theta: %g\n", treeTheta);
    } else if (param.directselfgravity) {
        fprintf(stdout, "yes - calculated using direct particle-particle force.\n");
    } else {
        fprintf(stdout, "none.\n");
    }

    if (param.maxtimestep < 0)
        param.maxtimestep = timePerStep;
    if (param.maxtimestep > timePerStep) {
        fprintf(stderr, "ERROR. maxtimestep cannot be larger than time between outputs...\n");
        exit(1);
    }

    if (param.verbose)
        fprintf(stdout, "\nLoading config file...\n");
    loadConfigFromFile(configFile);

    if (param.performanceTest) {
        if (param.verbose)
            fprintf(stdout, "Clearing performance file...\n");
        clear_performance_file();
    }

    if (param.verbose) {
        fprintf(stdout, "\nNo particles: %d\n", numberOfParticles);
        fprintf(stdout, "Allocating memory for %d particles...\n", maxNumberOfParticles);
    }

    // query GPU(s)
    fprintf(stdout, "\nChecking for cuda devices...\n");
    cudaDeviceProp deviceProp;
    int cnt;
    cudaVerify(cudaGetDeviceProperties(&deviceProp, wanted_device));
    cudaGetDeviceCount(&cnt);
    if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
        fprintf(stderr, "There is no CUDA capable device. Exiting...\n");
        exit(-1);
    }
    fprintf(stdout, "Found compute capability %d.%d (need at least 3.5).\n", deviceProp.major, deviceProp.minor);
    fprintf(stdout, "Found #gpus: %d: %s\n", cnt, deviceProp.name);
    numberOfMultiprocessors = deviceProp.multiProcessorCount;
    fprintf(stdout, "Found cuda device with %d multiprocessors.\n", numberOfMultiprocessors);

    // initialise the memory
    init_allocate_memory();

    if ((inputFile.data = fopen(inputFile.name, "r")) == NULL) {
        if (param.hdf5input) {
            fprintf(stdout, "\nReading input file %s.h5\n", inputFile.name);
            read_particles_from_file(inputFile);
        } else {
            exit(1);
        }
    } else {
        fprintf(stdout, "\nReading input file %s\n", inputFile.name);
        read_particles_from_file(inputFile);
        fclose(inputFile.data);
    }

    // read/initialize material constants and copy them to the GPU + init some values
    init_values();

    // print information about time integrator settings
    if( param.integrator_type == RK2_ADAPTIVE )
        print_rk2_adaptive_settings();

    // copy the particles to the GPU
    copy_particle_data_to_device();
    if (cudaSuccess != cudaMemcpyToSymbol(childList, &childListd, sizeof(void*))) {
        fprintf(stderr, "copying of childList to device failed\n");
        exit(1);
    }

    // create conserved quantities logfile and write header
    if(param.hdf5input){
        if( (conservedquantitiesfile = fopen(param.conservedquantitiesfilename, "a")) == NULL ) {
            fprintf(stderr, "Ohoh... Cannot open '%s' for writing. Abort...\n", param.conservedquantitiesfilename);
            exit(1);
        }
    }
    else {
        if( (conservedquantitiesfile = fopen(param.conservedquantitiesfilename, "w")) == NULL ) {
            fprintf(stderr, "Ohoh... Cannot open '%s' for writing. Abort...\n", param.conservedquantitiesfilename);
            exit(1);
        }
        fprintf(conservedquantitiesfile, " # 1:time 2:SPH-part-total 3:SPH-part-deactivated 4:grav.point-masses 5:total-mass 6:total-kinetic-energy 7:total-inner-energy ");
        int output_cnt = 8;
#if OUTPUT_GRAV_ENERGY
        fprintf(conservedquantitiesfile, "%d:total-grav-energy ", output_cnt++);
#endif
        fprintf(conservedquantitiesfile, "%d:total-momentum ", output_cnt++);
        fprintf(conservedquantitiesfile, "%d:total-momentum[x] ", output_cnt++);
#if DIM > 1
        fprintf(conservedquantitiesfile, "%d:total-momentum[y] ", output_cnt++);
#if DIM == 3
        fprintf(conservedquantitiesfile, "%d:total-momentum[z] ", output_cnt++);
#endif
#endif
#if DIM > 1
        fprintf(conservedquantitiesfile, "%d:total-angular-mom ", output_cnt++);
        fprintf(conservedquantitiesfile, "%d:total-angular-mom[x] ", output_cnt++);
        fprintf(conservedquantitiesfile, "%d:total-angular-mom[y] ", output_cnt++);
#if DIM == 3
        fprintf(conservedquantitiesfile, "%d:total-angular-mom[z] ", output_cnt++);
#endif
#endif
        fprintf(conservedquantitiesfile, "%d:barycenter-pos[x] ", output_cnt++);
#if DIM > 1
        fprintf(conservedquantitiesfile, "%d:barycenter-pos[y] ", output_cnt++);
#if DIM == 3
        fprintf(conservedquantitiesfile, "%d:barycenter-pos[z] ", output_cnt++);
#endif
#endif
        fprintf(conservedquantitiesfile, "%d:barycenter-vel[x] ", output_cnt++);
#if DIM > 1
        fprintf(conservedquantitiesfile, "%d:barycenter-vel[y] ", output_cnt++);
#if DIM == 3
        fprintf(conservedquantitiesfile, "%d:barycenter-vel[z]", output_cnt++);
#endif
#endif
        fprintf(conservedquantitiesfile, "\n");
    }
    fclose(conservedquantitiesfile);

    /* if HDF5 output is enabled and no hdf5 input is set, write the ASCII input file to HDF5 */
    if (param.hdf5output && !param.hdf5input) {
        fprintf(stdout, "\nWriting input ASCII file to HDF5 %s.h5...\n", inputFile.name);
        int asciiflag = param.ascii_output;
        param.ascii_output = 0;
        h5time = startTime;
        write_particles_to_file(inputFile);
        param.ascii_output = asciiflag;
    }

    // run the thing
    fprintf(stdout, "\n\nStarting time integration from start time %e...\n\n", startTime);
    cudaProfilerStart();
    timeIntegration();
    cudaProfilerStop();
    fprintf(stdout, "\nTime integration finished.\n\n");

    free_memory();

    fprintf(stdout, "Resetting GPU...\n");
    cudaVerify(cudaDeviceReset());

    fprintf(stdout, "\nkthxbye.\n");

    return 0;
}
