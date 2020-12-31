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


#include "io.h"
#include "timeintegration.h"
#include "config_parameter.h"
#include "pressure.h"
#include <libconfig.h>
#include <float.h>
#include "aneos.h"

#if HDF5IO
#include <hdf5.h>
#endif


int currentDiskIO = FALSE;

extern pthread_t fileIOthread;

extern double startTime;

extern double Smin;
extern double alphamin;
extern double emin;
extern double rhomin;
extern double damagemin;
extern double betamin;
extern double alpha_epspormin;
extern double epsilon_vmin;

File inputFile;


/*! \brief
  Reading the material properties using libconfig.
  \param pointer to config file
  \return nothing
*/
void loadConfigFromFile(char *configFile)
{
    config_init(&param.config);

    if (!config_read_file(&param.config, configFile)) {
        fprintf(stderr, "Error reading config file %s.\n", configFile);
        config_destroy(&param.config);
        exit(1);
    }
}


void set_integration_parameters()
{
    FILE *f;
    char line[1024];
    char *name;
    char *value;
    char *fn = "pc_values.dat";
    const char s[] = " = ";
    int found = FALSE;


    if ( (f = fopen(fn,"r")) == NULL) {
        if (param.integrator_type == MONAGHAN_PC || param.integrator_type == EULER_PC) {
            fprintf(stderr, "Can't open file %s!\n", fn);
            exit(1);
        }
    } else {
        found = TRUE;
    }

    Smin = rhomin = emin = alphamin = betamin = damagemin = alpha_epspormin = epsilon_vmin = 1e99;

    if (found) {
        while (fgets(line, sizeof(line), f)) {
            /* comments start with # */
            if (line[0] == '#') continue;
            name = strtok(line, s);
            value = strtok(NULL, s);
            if (!strcmp(name, "Smin")) {
                Smin = atof(value);
            }
            else if (!strcmp(name, "rhomin")) {
                rhomin = atof(value);
            }
            else if (!strcmp(name, "emin")) {
                emin = atof(value);
            }
            else if (!strcmp(name, "alphamin")) {
                alphamin = atof(value);
            }
            else if (!strcmp(name, "betamin")) {
                betamin = atof(value);
            }
            else if (!strcmp(name, "damagemin")) {
                damagemin = atof(value);
            }
            else if (!strcmp(name, "alpha_epspormin")) {
                alpha_epspormin = atof(value);
            }
            else if (!strcmp(name, "epsilon_vmin")) {
                epsilon_vmin = atof(value);
            }
        }
        fclose (f);
    }
    if (param.verbose && (param.integrator_type == MONAGHAN_PC || param.integrator_type == EULER_PC)) {
        fprintf(stdout, "Using following values for the predictor corrector integrator:\n");
#if SOLID
        fprintf(stdout, "Smin:\t\t\t %e\n", Smin);
#endif
#if INTEGRATE_ENERGY
        fprintf(stdout, "emin:\t\t\t %e\n", emin);
#endif
#if INTEGRATE_DENSITY
        fprintf(stdout, "rhomin:\t\t\t %e\n", rhomin);
#endif
#if FRAGMENTATION
        fprintf(stdout, "damagemin:\t\t %e\n", damagemin);
#endif
#if PALPHA_POROSITY
        fprintf(stdout, "alphamin:\t\t %e\n", alphamin);
#endif
#if INVISCID_SPH
        fprintf(stdout, "betamin:\t\t %e\n", betamin);
#endif
#if EPSALPHA_POROSITY
        fprintf(stdout, "alpha_epspormin:\t\t %e\n", alpha_epspormin);
        fprintf(stdout, "epsilon_vmin:\t\t %e\n", epsilon_vmin);
#endif
        fprintf(stdout, "These values (if not 1e99) are taken from file <pc_values.dat>.\n");
    }
}



/* set some initial values */
void init_values(void)
{
    int i;
    int matId;

    if (param.verbose) {
        printf("initialising material constants and copying them to the gpu\n");
    }
    transferMaterialsToGPU();

    for (i = 0; i < numberOfParticles; i++) {
        matId = p_host.materialId[i];

#if MORE_OUTPUT

#if PALPHA_POROSITY
	p_host.p_max[i] = p_host.p[i];
	p_host.p_min[i] = p_host.p[i];
#else
	p_host.p_max[i] = -DBL_MAX;
	p_host.p_min[i] = DBL_MAX;
#endif
#if INTEGRATE_DENSITY
	p_host.rho_max[i] = p_host.rho[i];
	p_host.rho_min[i] = p_host.rho[i];
#else
	p_host.rho_max[i] = -DBL_MAX;
	p_host.rho_min[i] = DBL_MAX;
#endif

	p_host.e_max[i] = p_host.e[i];
	p_host.e_min[i] = p_host.e[i];

	p_host.cs_max[i] = -DBL_MAX;
	p_host.cs_min[i] = DBL_MAX;

#endif

#if PALPHA_POROSITY
        p_host.cs[i] = cs_porous[matId];
#else
        p_host.cs[i] = sqrt(bulk_modulus[matId]/till_rho_0[matId]);
#endif

#if !READ_INITIAL_SML_FROM_PARTICLE_FILE
        if (!(p_host.h[i] > 0)) {
            p_host.h[i] = sml[matId];
        }
#endif
        p_host.h0[i] = p_host.h[i];
    }
}



// read in particles from start file
void read_particles_from_file(File inputFile)
{
    int my_anop;
    int i;
    int d;
    int c;
#if SOLID || NAVIER_STOKES
    int e;
#endif
    char h5filename[256];
    char h5massfilename[256];
    char massfilename[256];
    FILE *massfile;
    double h5time;
    double *x;
    int *ix;

#if HDF5IO
    hid_t file_id;
    hid_t x_id, v_id, m_id, mtype_id;
# if INTEGRATE_DENSITY
    hid_t rho_id;
# endif
# if INTEGRATE_ENERGY
    hid_t e_id;
# endif
    hid_t time_id;
# if VARIABLE_SML || READ_INITIAL_SML_FROM_PARTICLE_FILE
    hid_t sml_id;
# endif
    hid_t dspace;
# if FRAGMENTATION
    hid_t noaf_id, damage_id;
    hid_t activation_thresholds_id;
    hid_t maxnof_id;
    int nofi;
    int maxnof;
    double *ax;
# endif
# if GRAVITATING_POINT_MASSES
    hid_t rmin_id;
    hid_t rmax_id;
    hid_t flag_id;
# endif
# if JC_PLASTICITY
    hid_t ep_id, T_id;
# endif
# if SOLID
    hid_t S_id;
# endif
# if NAVIER_STOKES
    hid_t Tshear_id;
# endif

# if PALPHA_POROSITY
    hid_t p_id;
    hid_t alpha_id;
# if SOLID
# if FRAGMENTATION
    hid_t damage_porjutzi_id;
# endif
# endif
# endif

# if SIRONO_POROSITY
    hid_t K_id;
    hid_t rho_0prime_id;
    hid_t rho_c_plus_id;
    hid_t rho_c_minus_id;
    hid_t compressive_strength_id;
    hid_t tensile_strength_id;
    hid_t flag_rho_0prime_id;
    hid_t flag_plastic_id;
    hid_t shear_strength_id;
# endif
# if EPSALPHA_POROSITY
    hid_t alpha_epspor_id;
    hid_t epsilon_v_id;
# endif

    herr_t status;

    /* filename extension is .h5 */
    strcpy(massfilename, inputFile.name);
    strcat(massfilename, ".mass");
    strcpy(h5filename, inputFile.name);
    strcpy(h5massfilename, inputFile.name);
    strcat(h5filename, ".h5");
    strcat(h5massfilename, ".mass.h5");
#endif // HDF5IO

    // set start timestep from input filename
    const char* ext;
    ext = strrchr(inputFile.name, '.');
    if (!ext) {
        printf("could not get start timestep from filename. make sure to name the file *.1234 or something like this");
        exit(1);
    } else {
        sscanf(ext+1, "%04d", &startTimestep);
        printf("set start timestep to %d\n", startTimestep);
    }

    // START READING HDF5 INPUT FILE...
#if HDF5IO
    if (param.hdf5input) {
        printf("reading particle data from hdf5 file: %s.h5\n", inputFile.name);
# if GRAVITATING_POINT_MASSES
        printf("reading pointmass data from hdf5 file: %s.mass.h5\n", inputFile.name);
# endif
        file_id = H5Fopen (h5filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) {
            fprintf(stderr, "********************** Error opening file %s\n", h5filename);
            exit(1);
        } else {
            fprintf(stdout, "Using hdf5 input file %s\n", h5filename);
        }

        /* open the dataset for the positions */
        x_id = H5Dopen(file_id, "/x", H5P_DEFAULT);
        if (x_id < 0) {
            fprintf(stderr, "Could not find locations in hdf5 file.  Exiting.\n");
        }

        /* determine number of particles stored in hdf5 file */
        dspace = H5Dget_space(x_id);
        const int ndims = H5Sget_simple_extent_ndims(dspace);
        hsize_t dims[ndims];
        H5Sget_simple_extent_dims(dspace, dims, NULL);
        my_anop = dims[0];

        fprintf(stdout, "Reading data for %d particles.\n", my_anop);

        /* allocate space for my_anop particles */
        x = (double *) malloc(sizeof(double) * my_anop * DIM);

        /* read positions */
        status = H5Dread(x_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(x_id);
        for (i = 0, d = 0; i < my_anop; i++, d += DIM) {
            p_host.x[i] = x[d];
# if DIM > 1
            p_host.y[i] = x[d+1];
# if DIM == 3
            p_host.z[i] = x[d+2];
# endif
# endif
        }

        /* read velocities */
        v_id = H5Dopen(file_id, "/v", H5P_DEFAULT);
        if (v_id < 0) {
            fprintf(stderr, "Could not find velocities in hdf5 file.  Exiting.\n");
            exit(1);
        }
        status = H5Dread(v_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(v_id);
        for (i = 0, d = 0; i < my_anop; i++, d += DIM) {
            p_host.vx[i] = x[d];
# if DIM > 1
            p_host.vy[i] = x[d+1];
# if DIM == 3
            p_host.vz[i] = x[d+2];
# endif
# endif
        }

        /* read accreted velocities */
        v_id = H5Dopen(file_id, "/v_accreted", H5P_DEFAULT);
        if (v_id < 0) {
            fprintf(stdout, "Could not find accreted velocities in hdf5 file.\n");
        }
        else {
            fprintf(stdout, "Found velocities of accreted particles and reading them.\n");
            status = H5Dread(v_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
            status = H5Dclose(v_id);
            for (i = 0, d = 0; i < my_anop; i++, d += DIM) {
                p_host.vx0[i] = x[d];
# if DIM > 1
                p_host.vy0[i] = x[d+1];
# if DIM == 3
                p_host.vz0[i] = x[d+2];
# endif
# endif
            }
        }
        free(x);

        /* read simulation time */
        time_id = H5Dopen(file_id, "/time",  H5P_DEFAULT);
        if (time_id < 0) {
            fprintf(stderr, "Could not find time in hdf5 file.  Exiting.\n");
            exit(1);
        }
        status = H5Dread(time_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &h5time);
        status = H5Dclose(time_id);
        fprintf(stdout, "Current time: %g\n", h5time);
        startTime = h5time;

        /* read masses */
        dims[0] = my_anop;
        dims[1] = 1;
        x = (double * ) malloc(sizeof(double) * my_anop);
        m_id = H5Dopen(file_id, "/m", H5P_DEFAULT);
        if (m_id < 0) {
            fprintf(stderr, "Could not find mass information in hdf5 file.  Exiting\n");
            exit(1);
        }
        status = H5Dread(m_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(m_id);
        for (i = 0; i < my_anop; i++) {
            p_host.m[i] = x[i];
        }
        free(x);

# if PALPHA_POROSITY
        /* read alpha_jutzi */
        dims[0] = my_anop;
        dims[1] = 1;
        alpha_id = H5Dopen(file_id, "/alpha_jutzi", H5P_DEFAULT);
        if (alpha_id < 0) {
            fprintf(stderr, "Could not find alpha_jutzi information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(alpha_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(alpha_id);

        for (i = 0; i < my_anop; i++) {
            p_host.alpha_jutzi[i] = x[i];
        }
        free(x);

        /* read pressures */
        p_id = H5Dopen(file_id, "/p", H5P_DEFAULT);
        if (p_id < 0) {
            fprintf(stderr, "Could not find pressure information in hdf5 file.  Exiting\n");
            exit(1);
        } else {
            fprintf(stdout, "Reading actual pressure data to pressure_old on the device.\n");
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(p_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(p_id);

        for (i = 0; i < my_anop; i++) {
            p_host.pold[i] = x[i];
        }
        free(x);

# if FRAGMENTATION
        /* read damage_porjutzi */
        damage_porjutzi_id = H5Dopen(file_id, "/DIM_root_of_damage_porjutzi", H5P_DEFAULT);
        if (damage_porjutzi_id < 0) {
            fprintf(stderr, "Could not find damage_porjutzi information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(damage_porjutzi_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(damage_porjutzi_id);

        for (i = 0; i < my_anop; i++) {
            p_host.damage_porjutzi[i] = x[i];
        }
        free(x);
# endif
# endif

# if SIRONO_POROSITY
        /* read rho_c_plus */
        rho_c_plus_id = H5Dopen(file_id, "/rho_c_plus", H5P_DEFAULT);
        if (rho_c_plus_id < 0) {
            fprintf(stderr, "Could not find rho_c_plus information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(rho_c_plus_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rho_c_plus_id);
        for (i = 0; i < my_anop; i++) {
            p_host.rho_c_plus[i] = x[i];
        }
        free(x);

        /* read rho_c_minus */
        rho_c_minus_id = H5Dopen(file_id, "/rho_c_minus", H5P_DEFAULT);
        if (rho_c_minus_id < 0) {
            fprintf(stderr, "Could not find rho_c_minus information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(rho_c_minus_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rho_c_minus_id);

        for (i = 0; i < my_anop; i++) {
            p_host.rho_c_minus[i] = x[i];
        }
        free(x);

        /* read bulk modulus */
        K_id = H5Dopen(file_id, "/K", H5P_DEFAULT);
        if (K_id < 0) {
            fprintf(stderr, "Could not find bulk modulus information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(K_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(K_id);

        for (i = 0; i < my_anop; i++) {
            p_host.K[i] = x[i];
        }
        free(x);

        /* read rho_0prime */
        rho_0prime_id = H5Dopen(file_id, "/rho_0prime", H5P_DEFAULT);
        if (rho_0prime_id < 0) {
            fprintf(stderr, "Could not find rho_0prime information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(rho_0prime_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rho_0prime_id);

        for (i = 0; i < my_anop; i++) {
            p_host.rho_0prime[i] = x[i];
        }
        free(x);

        /* read compressive_strength */
        compressive_strength_id = H5Dopen(file_id, "/compressive_strength", H5P_DEFAULT);
        if (compressive_strength_id < 0) {
            fprintf(stderr, "Could not find compressive_strength information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(compressive_strength_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(compressive_strength_id);

        for (i = 0; i < my_anop; i++) {
            p_host.compressive_strength[i] = x[i];
        }
        free(x);

        /* read tensile_strength */
        tensile_strength_id = H5Dopen(file_id, "/tensile_strength", H5P_DEFAULT);
        if (tensile_strength_id < 0) {
            fprintf(stderr, "Could not find tensile_strength information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(tensile_strength_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(tensile_strength_id);

        for (i = 0; i < my_anop; i++) {
            p_host.tensile_strength[i] = x[i];
        }
        free(x);

        /* read shear_strength */
        shear_strength_id = H5Dopen(file_id, "/shear_strength", H5P_DEFAULT);
        if (shear_strength_id < 0) {
            fprintf(stderr, "Could not find shear_strength information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(shear_strength_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(shear_strength_id);

        for (i = 0; i < my_anop; i++) {
            p_host.shear_strength[i] = x[i];
        }
        free(x);

        /* read flag_rho_0prime */
        flag_rho_0prime_id = H5Dopen(file_id, "/flag_rho_0prime", H5P_DEFAULT);
        if (flag_rho_0prime_id < 0) {
            fprintf(stderr, "Could not flag_rho_0prime information in hdf5 file.  Exiting\n");
            exit(1);
        }

        ix = (int *) malloc(sizeof(int) * my_anop);
        if (!(ix)) {
            fprintf(stderr, "Cannot allocate enough memory.\n");
            exit(1);
        }
        status = H5Dread(flag_rho_0prime_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ix);
        status = H5Dclose(flag_rho_0prime_id);

        for (i = 0; i < my_anop; i++) {
            p_host.flag_rho_0prime[i] = ix[i];
        }
        free(ix);

        /* read flag_plastic */
        flag_plastic_id = H5Dopen(file_id, "/flag_plastic", H5P_DEFAULT);
        if (flag_plastic_id < 0) {
            fprintf(stderr, "Could not flag_plastic information in hdf5 file.  Exiting\n");
            exit(1);
        }
        ix = (int *) malloc(sizeof(int) * my_anop);
        status = H5Dread(flag_plastic_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ix);
        status = H5Dclose(flag_plastic_id);

        for (i = 0; i < my_anop; i++) {
            p_host.flag_plastic[i] = ix[i];
        }
        free(ix);
# endif

# if EPSALPHA_POROSITY
        /* read alpha_epspor */
        alpha_epspor_id = H5Dopen(file_id, "/alpha_epspor", H5P_DEFAULT);
        if (alpha_epspor_id < 0) {
            fprintf(stderr, "Could not find alpha_epspor information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(alpha_epspor_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(alpha_epspor_id);

        for (i = 0; i < my_anop; i++) {
            p_host.alpha_epspor[i] = x[i];
        }
        free(x);

        /* read epsilon_v */
        epsilon_v_id = H5Dopen(file_id, "/epsilon_v", H5P_DEFAULT);
        if (epsilon_v_id < 0) {
            fprintf(stderr, "Could not find epsilon_v information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(epsilon_v_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(epsilon_v_id);

        for (i = 0; i < my_anop; i++) {
            p_host.epsilon_v[i] = x[i];
        }
        free(x);
# endif

# if VARIABLE_SML || READ_INITIAL_SML_FROM_PARTICLE_FILE
        /* read sml */
        sml_id =  H5Dopen(file_id, "/sml", H5P_DEFAULT);
        if (sml_id < 0) {
            fprintf(stderr, "Could not find smoothing length information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(sml_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(sml_id);

        for (i = 0; i < my_anop; i++) {
            p_host.h[i] = x[i];
        }
        free(x);
# endif

# if READ_INITIAL_SML_FROM_PARTICLE_FILE
        /* read sml0 */
        sml_id =  H5Dopen(file_id, "/sml_initial", H5P_DEFAULT);
        if (sml_id < 0) {
            fprintf(stderr, "Could not find initial smoothing length information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(sml_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(sml_id);

        for (i = 0; i < my_anop; i++) {
            p_host.h0[i] = x[i];
        }
        free(x);
# endif

# if INTEGRATE_DENSITY
        /* read densities */
        rho_id = H5Dopen(file_id, "/rho", H5P_DEFAULT);
        if (rho_id < 0) {
            fprintf(stderr, "Could not find density information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(rho_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rho_id);

        for (i = 0; i < my_anop; i++) {
            p_host.rho[i] = x[i];
        }
        free(x);
# endif

# if INTEGRATE_ENERGY
        /* read internal energies */
        e_id = H5Dopen(file_id, "/e", H5P_DEFAULT);
        if (e_id < 0) {
            fprintf(stderr, "Could not find energy information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(e_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(e_id);

        for (i = 0; i < my_anop; i++) {
            p_host.e[i] = x[i];
        }
        free(x);
# endif

        /* read material types */
        mtype_id = H5Dopen(file_id, "/material_type", H5P_DEFAULT);
        if (mtype_id < 0) {
            fprintf(stderr, "Could not material type information in hdf5 file.  Exiting\n");
            exit(1);
        }
        ix = (int *) malloc(sizeof(int) * my_anop);
        if (!(ix)) {
            fprintf(stderr, "Cannot allocate enough memory.\n");
            exit(1);
        }
        status = H5Dread(mtype_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ix);
        status = H5Dclose(mtype_id);

        for (i = 0; i < my_anop; i++) {
            p_host.materialId[i] = ix[i];
        }
        free(ix);

# if JC_PLASTICITY
        /* read plastic strains */
        ep_id = H5Dopen(file_id, "/ep", H5P_DEFAULT);
        if (ep_id < 0) {
            fprintf(stderr, "Could not find plastic strain information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(ep_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(ep_id);

        for (i = 0; i < my_anop; i++) {
            p_host.ep[i] = x[i];
        }
        free(x);

        /* read temperatures */
        T_id = H5Dopen(file_id, "/T", H5P_DEFAULT);
        if (T_id < 0) {
            fprintf(stderr, "Could not find temperature information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(T_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(T_id);

        for (i = 0; i < my_anop; i++) {
            p_host.T[i] = x[i];
        }
        free(x);
# endif

# if FRAGMENTATION
        /* read number of activated flaws */
        noaf_id = H5Dopen(file_id, "/number_of_activated_flaws", H5P_DEFAULT);
        if (noaf_id < 0) {
            fprintf(stderr, "Could not find number of activated flaws information in hdf5 file.  Exiting\n");
            exit(1);
        }
        ix = (int *) malloc(sizeof(int) * my_anop);
        status = H5Dread(noaf_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ix);
        status = H5Dclose(noaf_id);

        for (i = 0; i < my_anop; i++) {
            p_host.numActiveFlaws[i] = ix[i];
        }
        free(ix);

        /* read damage */
        damage_id = H5Dopen(file_id, "/DIM_root_of_damage_tensile", H5P_DEFAULT);
        if (damage_id < 0) {
            fprintf(stderr, "Could not find tensile damage information in hdf5 file.  Exiting\n");
            exit(1);
        }
        x = (double * ) malloc(sizeof(double) * my_anop);
        status = H5Dread(damage_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(damage_id);

        for (i = 0; i < my_anop; i++) {
            p_host.d[i] = x[i];
        }
        free(x);

        /* read max number of activation thresholds */
        maxnof_id = H5Dopen(file_id, "/maximum_number_of_flaws", H5P_DEFAULT);
        if (maxnof_id < 0) {
            fprintf(stderr, "Could not find maximum number of flaws in hdf5 file.  Exiting.\n");
            exit(1);
        }
        status = H5Dread(maxnof_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &maxnof);
        status = H5Dclose(maxnof_id);
        fprintf(stdout, "Maximum number of activation thresholds for a particle in the data is %d\n", maxnof);

        /* read the activation thresholds (and set number-of-flaws accordingly) */
        dims[0] = my_anop;
        dims[1] = maxnof;
        x = (double *) malloc(sizeof(double) * my_anop * maxnof);
        if (!x) {
            fprintf(stderr, "Cannot allocate enough memory.\n");
            exit(1);
        }
        activation_thresholds_id = H5Dopen(file_id, "/activation_thresholds", H5P_DEFAULT);
        if (activation_thresholds_id < 0) {
            fprintf(stderr, "Could not find activation thresholds in hdf5 file.  Exiting.\n");
            exit(1);
        }
        status = H5Dread(activation_thresholds_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(activation_thresholds_id);

        ax = (double *) malloc(sizeof(double) * maxnof);
        for (i = 0; i < my_anop; i++) {
            nofi = 0;
            while (x[i*maxnof + nofi] > 0 && nofi < maxnof) {
                ax[nofi] = x[i*maxnof + nofi];
                nofi++;
            }
            p_host.numFlaws[i] = nofi;
            for (d = 0; d < nofi; d++) {
                p_host.flaws[i*MAX_NUM_FLAWS+d] = ax[d];
            }
        }
        free(ax);
        free(x);
# endif

# if NAVIER_STOKES
        /* read deviatoric stresses */
        x = (double *) malloc(sizeof(double) * my_anop * DIM * DIM);
        dims[0] = my_anop;
        dims[1] = DIM*DIM;

        Tshear_id = H5Dopen(file_id, "/viscous_shear_stress", H5P_DEFAULT);
        if (Tshear_id < 0) {
            fprintf(stderr, "Could not find viscous_shear_stress information in hdf5 file.  Exiting\n");
            exit(1);
        }
        status = H5Dread(Tshear_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(Tshear_id);
        for (i = 0; i < my_anop; i++) {
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    p_host.Tshear[i*DIM*DIM+d*DIM+e] = x[i*DIM*DIM + d*DIM + e];
                }
            }
        }
        free(x);
# endif

# if SOLID
        /* read deviatoric stresses */
        x = (double *) malloc(sizeof(double) * my_anop * DIM * DIM);
        dims[0] = my_anop;
        dims[1] = DIM*DIM;

        S_id = H5Dopen(file_id, "/deviatoric_stress", H5P_DEFAULT);
        if (S_id < 0) {
            fprintf(stderr, "Could not find stress information in hdf5 file.  Exiting\n");
            exit(1);
        }
        status = H5Dread(S_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(S_id);
        for (i = 0; i < my_anop; i++) {
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    p_host.S[i*DIM*DIM+d*DIM+e] = x[i*DIM*DIM + d*DIM + e];
                }
            }
        }
        free(x);
# endif
        H5Fclose(file_id);


        // START READING POINTMASSES INPUT FILE...
# if GRAVITATING_POINT_MASSES
        file_id = H5Fopen (h5massfilename, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) {
            fprintf(stderr, "********************** Error opening file %s\n", h5massfilename);
            exit(1);
        } else {
            fprintf(stdout, "Using hdf5 input file %s\n", h5massfilename);
        }

        /* open the dataset for the positions */
        x_id = H5Dopen(file_id, "/x", H5P_DEFAULT);
        if (x_id < 0) {
            fprintf(stderr, "Could not find locations in hdf5 file.  Exiting.\n");
        }

        /* determine number of particles stored in hdf5 file */
        dspace = H5Dget_space(x_id);
        const int mndims = H5Sget_simple_extent_ndims(dspace);
        hsize_t mdims[mndims];
        H5Sget_simple_extent_dims(dspace, mdims, NULL);
        my_anop = mdims[0];
        fprintf(stdout, "Reading data for %d pointmasses.\n", my_anop);

        /* allocate space for my_anop particles */
        x = (double *) malloc(sizeof(double) * my_anop * DIM);

        /* read positions */
        status = H5Dread(x_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(x_id);
        for (i = 0, d = 0; i < my_anop; i++, d += DIM) {
            pointmass_host.x[i] = x[d];
# if DIM > 1
            pointmass_host.y[i] = x[d+1];
# if DIM == 3
            pointmass_host.z[i] = x[d+2];
# endif
# endif
        }

        /* read velocities */
        v_id = H5Dopen(file_id, "/v", H5P_DEFAULT);
        if (v_id < 0) {
            fprintf(stderr, "Could not find velocities in hdf5 file.  Exiting.\n");
            exit(1);
        }
        status = H5Dread(v_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(v_id);
        for (i = 0, d = 0; i < my_anop; i++, d += DIM) {
            pointmass_host.vx[i] = x[d];
# if DIM > 1
            pointmass_host.vy[i] = x[d+1];
# if DIM == 3
            pointmass_host.vz[i] = x[d+2];
# endif
# endif
        }

        /* read masses */
        dims[0] = my_anop;
        dims[1] = 1;
        free(x);
        x = (double * ) malloc(sizeof(double) * my_anop);
        m_id = H5Dopen(file_id, "/m", H5P_DEFAULT);
        if (m_id < 0) {
            fprintf(stderr, "Could not find mass information in hdf5 file.  Exiting\n");
            exit(1);
        }
        status = H5Dread(m_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(m_id);

        for (i = 0; i < my_anop; i++) {
            pointmass_host.m[i] = x[i];
        }

        rmin_id = H5Dopen(file_id, "/rmin", H5P_DEFAULT);
        if (rmin_id < 0) {
            fprintf(stderr, "Could not find rmin information in hdf5 file.  Exiting\n");
            exit(1);
        }
        status = H5Dread(rmin_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rmin_id);

        for (i = 0; i < my_anop; i++) {
            pointmass_host.rmin[i] = x[i];
        }

        rmax_id = H5Dopen(file_id, "/rmax", H5P_DEFAULT);
        if (rmax_id < 0) {
            fprintf(stderr, "Could not find rmax information in hdf5 file.  Exiting\n");
            exit(1);
        }
        status = H5Dread(rmax_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rmax_id);

        for (i = 0; i < my_anop; i++) {
            pointmass_host.rmax[i] = x[i];
        }
        free(x);

        // read feels_particles flag
        ix = (int *) malloc(sizeof(int) * my_anop);
        flag_id = H5Dopen(file_id, "/feels_particles", H5P_DEFAULT);
        if (flag_id < 0) {
            fprintf(stderr, "Could not find feels_particles flag information in hdf5 file.  Exiting\n");
            exit(1);
        }
        status = H5Dread(flag_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ix);
        status = H5Dclose(flag_id);

        for (i = 0; i < my_anop; i++) {
            pointmass_host.feels_particles[i] = ix[i];
        }
        free(ix);

        H5Fclose(file_id);
# endif // GRAVITATING_POINT_MASSES

        if (param.verbose)
            fprintf(stdout, "%d\n", status);
    }
#endif // HDF5IO


    // START READING ASCII INPUT FILE...
#if FRAGMENTATION
    maxNumFlaws_host = MAX_NUM_FLAWS;
#endif
    int columns;
    int pcnt = 0;
    char iotmp[256];

    if (!param.hdf5input) {
        for (i = 0; i < numberOfParticles; i++) {
            // read in coordinates
            columns = 0;
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read x-position from input file\n");
            p_host.x[i] = atof(iotmp);
            columns++;
#if DIM > 1
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read y-position from input file\n");
            p_host.y[i] = atof(iotmp);
            columns++;
#if DIM == 3
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read z-position from input file\n");
            p_host.z[i] = atof(iotmp);
            columns++;
#endif
#endif
            // read in velocity
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read x-velocity from input file\n");
            p_host.vx[i] = atof(iotmp);
            columns++;
#if DIM > 1
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read y-velocity from input file\n");
            p_host.vy[i] = atof(iotmp);
            columns++;
#if DIM == 3
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read z-velocity from input file\n");
            p_host.vz[i] = atof(iotmp);
            columns++;
#endif
#endif
            // read in mass
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read mass from input file\n");
            p_host.m[i] = atof(iotmp);
            columns++;
#if INTEGRATE_DENSITY
            // read in density
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read density from input file\n");
            p_host.rho[i] = atof(iotmp);
            columns++;
#endif

#if INTEGRATE_ENERGY
            // read in energy
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read energy from input file\n");
            p_host.e[i] = atof(iotmp);
            columns++;
#endif

#if READ_INITIAL_SML_FROM_PARTICLE_FILE
            // read in smoothing length
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read smoothing length from input file\n");
            p_host.h[i] = atof(iotmp);
            // remember the input value
            p_host.h0[i] = p_host.h[i];
            columns++;
#else
            if (param.restart) {
                // dummy read in of sml
                fscanf(inputFile.data, "%s", &iotmp);
                // dummy read in of number of interaction partners
                fscanf(inputFile.data, "%s", &iotmp);
            }
#endif

            // read in material ID
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read material ID from particle %d input file\n", i);
            p_host.materialId[i] = atoi(iotmp);
            columns++;

#if JC_PLASTICITY
            // read in strain
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read strain from particle %d input file\n", i);
            p_host.ep[i] = atof(iotmp);
            columns++;

            // read in temperature
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read temperature from particle %d input file\n", i);
            p_host.T[i] = atof(iotmp);
            columns++;
#endif

#if FRAGMENTATION
            // read in number of flaws
            if (!fscanf(inputFile.data, "%s", &iotmp)) {
                fprintf(stderr, "ERROR. Could not read number of flaws from input file...\n");
                exit(1);
            }
            p_host.numFlaws[i] = atoi(iotmp);
            columns++;

            if (p_host.numFlaws[i] > maxNumFlaws_host) {
                fprintf(stderr, "ERROR. Found particle with %d flaws in input file. Set max number of flaws higher.\n", p_host.numFlaws[i]);
                exit(1);
            }

            if (param.restart) {
                // read in number of activated flaws
                if (!fscanf(inputFile.data, "%s", &iotmp)) {
                    fprintf(stderr, "ERROR. Could not read number of activated flaws from input file...\n");
                    exit(1);
                }
                p_host.numActiveFlaws[i] = atoi(iotmp);
                columns++;
                if (p_host.numActiveFlaws[i] > p_host.numFlaws[i]) {
                    fprintf(stderr, "ERROR. Found particle with more activated flaws than actual flaws in input file...\n");
                    exit(1);
                }

                // dummy read in of local strain
                fscanf(inputFile.data, "%s", &iotmp);
            }

            // read in damage
            if (!fscanf(inputFile.data, "%s", &iotmp)) {
                fprintf(stderr, "ERROR. Could not read damage from input file.\n");
                exit(1);
            }
            p_host.d[i] = atof(iotmp);
            columns++;

            if (!param.restart) {
                // calculate number of activated flaws for consistent initial conditions
                p_host.numActiveFlaws[i] = ceil( (double)p_host.numFlaws[i] * pow(p_host.d[i],DIM) );
                if (p_host.numActiveFlaws[i] > p_host.numFlaws[i])
                    p_host.numActiveFlaws[i] = p_host.numFlaws[i];
                if (p_host.numActiveFlaws[i] < 0) {
                    fprintf(stderr, "ERROR. Found particle with negative number of activated flaws while reading input file...\n");
                    exit(1);
                }
            }
#endif

            if (param.restart) {
                // dummy read in of pressure
                fscanf(inputFile.data, "%s", &iotmp);
            }

#if SOLID
            // read in deviator stress tensor
            int j, k;
            for (j = 0; j < DIM; j++) {
                for (k = 0; k < DIM; k++) {
                    if (!fscanf(inputFile.data, "%s", &iotmp))
                        fprintf(stderr, "could not read stress tensor from input file\n");
                    p_host.S[i*DIM*DIM+j*DIM+k] = atof(iotmp);
                    columns++;
                }
            }
#endif

#if SIRONO_POROSITY
            // read in rho_0prime
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read rho_0prime from input file\n");
            p_host.rho_0prime[i] = atof(iotmp);
            columns++;

            // read in rho_c_plus
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read rho_c_plus from input file\n");
            p_host.rho_c_plus[i] = atof(iotmp);
            columns++;

            // read in rho_c_minus
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read rho_c_minus from input file\n");
            p_host.rho_c_minus[i] = atof(iotmp);
            columns++;

            // read in compressive_strength
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read compressive_strength from input file\n");
            p_host.compressive_strength[i] = atof(iotmp);
            columns++;

            // read in tensile_strength
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read tensile_strength from input file\n");
            p_host.tensile_strength[i] = atof(iotmp);
            columns++;

            // read in bulk modulus K
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read bulk modulus K from input file\n");
            p_host.K[i] = atof(iotmp);
            columns++;

            // read in flag_rho_0prime
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read flag_rho_0prime from input file\n");
            p_host.flag_rho_0prime[i] = atoi(iotmp);
            columns++;

            // read in flag_plastic
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read flag_plastic from input file\n");
            p_host.flag_plastic[i] = atoi(iotmp);
            columns++;

            // read in shear_strength
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read shear_strength from input file\n");
            p_host.shear_strength[i] = atof(iotmp);
            columns++;
#endif

#if PALPHA_POROSITY
            // read in alpha_jutzi
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read alpha_jutzi from input file\n");
            p_host.alpha_jutzi[i] = atof(iotmp);
            columns++;
            // read in initial pressure
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read pressure from input file\n");
            p_host.pold[i] = atof(iotmp);
            columns++;
#endif

#if EPSALPHA_POROSITY
            // read in alpha_epspor
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read alpha_epspor from input file\n");
            p_host.alpha_epspor[i] = atof(iotmp);
            columns++;
            // read in  epsilon_v
            if (!fscanf(inputFile.data, "%s", &iotmp))
                fprintf(stderr, "could not read epsilon_v from input file\n");
            p_host.epsilon_v[i] = atof(iotmp);
            columns++;
#endif

#if FRAGMENTATION
            int d;
            for (d = 0; d < p_host.numFlaws[i]; d++) {
                if (!fscanf(inputFile.data, "%s", &iotmp))
                    fprintf(stderr, "could not read activation threshold\n");
                p_host.flaws[i*MAX_NUM_FLAWS+d] = atof(iotmp);
                columns++;
            }
#endif

            // check for end of line
            char ch;
            char ch2;
            fscanf(inputFile.data, "%c", &ch);
            if (ch == '\n') {
#if DEBUG
# if DIM == 1
    	        fprintf(stdout, "Reading coordinates for particle no. %d (x) = %e \n", i+1, p_host.x[i]);
# endif
# if DIM == 2
	            fprintf(stdout, "Reading coordinates for particle no. %d (x,y) = %e %e\n", i+1, p_host.x[i], p_host.y[i]);
# endif
# if DIM == 3
	            fprintf(stdout, "Reading coordinates for particle no. %d (x,y,z) = %e %e %e\n", i+1, p_host.x[i], p_host.y[i], p_host.z[i]);
# endif
#endif
            } else if (ch == '\t') {
                fscanf(inputFile.data, "%c", &ch2);
                if (ch2 == '\n') {
                    fprintf(stdout, "Warning. Line ending with \\t\\n, expected only a \\n.");
#if DIM == 1
                    fprintf(stdout, "Reading coordinates for particle no. %d (x) = %e \n", i+1, p_host.x[i]);
#endif
#if DIM == 2
                    fprintf(stdout, "Reading coordinates for particle no. %d (x,y) = %e %e\n", i+1, p_host.x[i], p_host.y[i]);
#endif
#if DIM == 3
                    fprintf(stdout, "Reading coordinates for particle no. %d (x,y,z) = %e %e %e\n", i+1, p_host.x[i], p_host.y[i], p_host.z[i]);
#endif
                } else {
                    fprintf(stderr, "End of line not reached. Check your input file.\n");
                    exit(1);
                }
            }
        }

#if GRAVITATING_POINT_MASSES
        if ((massfile = fopen(massfilename, "r")) == NULL) {
        fprintf(stderr, "Error: File %s not found.\n", massfilename);
            exit(1);
        }

        for (i = 0; i < numberOfPointmasses; i++) {
            // read in coordinates
            columns = 0;
            if (!fscanf(massfile, "%s", &iotmp))
                fprintf(stderr, "could not read x-position from input file\n");
            pointmass_host.x[i] = atof(iotmp);
            columns++;
#if DIM > 1
            if (!fscanf(massfile, "%s", &iotmp))
                fprintf(stderr, "could not read y-position from input file\n");
            pointmass_host.y[i] = atof(iotmp);
            columns++;
#if DIM == 3
            if (!fscanf(massfile, "%s", &iotmp))
                fprintf(stderr, "could not read z-position from input file\n");
            pointmass_host.z[i] = atof(iotmp);
            columns++;
#endif
#endif
            // read in velocities
            if (!fscanf(massfile, "%s", &iotmp))
                fprintf(stderr, "could not read x-velocity from input file\n");
            pointmass_host.vx[i] = atof(iotmp);
            columns++;
#if DIM > 1
            if (!fscanf(massfile, "%s", &iotmp))
                fprintf(stderr, "could not read y-velocity from input file\n");
            pointmass_host.vy[i] = atof(iotmp);
            columns++;
#if DIM == 3
            if (!fscanf(massfile, "%s", &iotmp))
                fprintf(stderr, "could not read z-velocity from input file\n");
            pointmass_host.vz[i] = atof(iotmp);
            columns++;
#endif
#endif
            // read in mass
            if (!fscanf(massfile, "%s", &iotmp))
                fprintf(stderr, "could not read mass from input file\n");
            pointmass_host.m[i] = atof(iotmp);
            columns++;
            // read in rmin
            if (!fscanf(massfile, "%s", &iotmp))
                fprintf(stderr, "could not read rmin from input file\n");
            pointmass_host.rmin[i] = atof(iotmp);
            columns++;
            // read in rmax
            if (!fscanf(massfile, "%s", &iotmp))
                fprintf(stderr, "could not read rmax from input file\n");
            pointmass_host.rmax[i] = atof(iotmp);
            columns++;
            // read in feels_particles flag
            if (!fscanf(massfile, "%s", &iotmp)) {
                fprintf(stderr, "could not read feels_particles flag from input file\n");
                fprintf(stderr, "using the default value of zero\n");
                pointmass_host.feels_particles[i] = 0;
            } else {
                pointmass_host.feels_particles[i] = atoi(iotmp);
            }
            printf("Mass no %d feels particles (no 0/ yes 1) %d\n", i, pointmass_host.feels_particles[i]);
            columns++;
        }

        c = fgetc(massfile);
        if (c != '\n' && c != EOF && c != '\t') {
            fprintf(stderr, "Error in input file format. Read %d columns and did not reach end of line.\n", columns);
            fprintf(stderr, "c=%c. i=%d.\n", c, c);
            exit(1);
        }
        fclose(massfile);
#endif // GRAVITATING_POINT_MASSES
    } /* ! param.hdf5input */
}

void write_particles_to_file(File file) {

    char h5filename[256];
    char infofilename[256];
    char massfilename[256];
    char h5massfilename[256];
    double *x;
    int *ix;
    int e;
    int i, j, k;
    int d;
    FILE *infofile;
    FILE *massfile;
    FILE *conservedquantitiesfile;
    FILE *binarysystemfile;
    int numberOfParticlesToIgnore;
    double totalp, totalmass, totalkineticenergy, totalinnerenergy;
#if GRAVITATING_POINT_MASSES
    double binaryangularmomentum[DIM], diskangularmomentum[DIM], diskmass, totalDL, totalBL;
    diskmass = totalDL = totalBL = 0.0;
    for (d=0; d<DIM; d++) {
        diskangularmomentum[d] = binaryangularmomentum[d] = 0.0;
    }
#endif
#if OUTPUT_GRAV_ENERGY
    double totalgravenergy = 0.0;
    double dist;
#endif
    double totalangularmomentum[DIM], totalL;
    double totalpx, totalpy, totalpz, v;
    double barycenter_pos[DIM], barycenter_vel[DIM];

    totalp = totalmass = totalkineticenergy = totalinnerenergy = 0.0;

    strcpy(h5filename, file.name);
    strcat(h5filename, ".h5");
    strcpy(infofilename, file.name);
    strcat(infofilename, ".info");
    strcpy(massfilename, file.name);
    strcat(massfilename, ".mass");
    strcpy(h5massfilename, massfilename);
    strcat(h5massfilename, ".h5");

#if HDF5IO
    /* hdf5 related stuff */
    hid_t file_id;
    hid_t x_id, v_id, m_id, rho_id, e_id, sml_id, noi_id, mtype_id;
    hid_t a_id;
    hid_t g_a_id;
#if MORE_ANEOS_OUTPUT
    hid_t aneos_T_id, aneos_cs_id, aneos_entropy_id, aneos_phase_flag_id;
#endif
#if MORE_OUTPUT
    hid_t p_min_id, p_max_id, rho_min_id, rho_max_id, e_min_id, e_max_id, cs_min_id, cs_max_id;
#endif
    hid_t time_id;
    hid_t cs_id;
    hid_t depth_id;
#if GRAVITATING_POINT_MASSES
    hid_t rmin_id, rmax_id, flag_id;
#endif
#if FRAGMENTATION
    hid_t noaf_id, damage_id;
    hid_t dddt_id;
    hid_t activation_thresholds_id;
    hid_t maxnof_id;
    int maxnof = 0;
#if PALPHA_POROSITY
    hid_t damage_porjutzi_id;
    hid_t damage_total_id;
    hid_t ddamage_porjutzidt_id;
#endif
#endif
#if INTEGRATE_DENSITY
    hid_t drhodt_id;
#endif
#if INTEGRATE_ENERGY
    hid_t dedt_id;
#endif
#if JC_PLASTICITY
    hid_t ep_id, T_id;
#endif
    hid_t p_id;
#if NAVIER_STOKES
    hid_t Tshear_id;
#endif
#if SOLID
    hid_t S_id;
    hid_t dSdt_id;
    hid_t local_strain_id;
#endif

#if PALPHA_POROSITY
    hid_t alpha_id;
    hid_t dalphadt_id;
#endif

#if SIRONO_POROSITY
    hid_t compressive_strength_id;
    hid_t tensile_strength_id;
    hid_t K_id;
    hid_t rho_0prime_id;
    hid_t rho_c_plus_id;
    hid_t rho_c_minus_id;
    hid_t flag_rho_0prime_id;
    hid_t flag_plastic_id;
    hid_t shear_strength_id;
#endif

#if EPSALPHA_POROSITY
    hid_t alpha_epspor_id;
    hid_t epsilon_v_id;
#endif

    herr_t status;
    hsize_t dims[2];
    hid_t dataspace_id;
#endif // HDF5IO

    if (! (param.hdf5output || param.ascii_output)) {
        fprintf(stderr, "Neither HDF5 nor ASCII output wanted. That's not what you want. Re-enabling ASCII output.\n");
        param.ascii_output = TRUE;
    }

    if (param.ascii_output) {
        // open file for writing
        if ((file.data = fopen(file.name, "w")) == NULL) {
            fprintf(stderr, "Eih? Cannot write to %s.\n", file.name);
            exit(1);
        }
        // write to file
        for (i = 0; i < numberOfParticles; i++) {
            fprintf(file.data, "%+.15le\t", p_host.x[i]);
#if DIM > 1
            fprintf(file.data, "%+.15le\t", p_host.y[i]);
#if DIM == 3
            fprintf(file.data, "%+.15le\t", p_host.z[i]);
#endif
#endif
            fprintf(file.data, "%+.15le\t", p_host.vx[i]);
#if DIM > 1
            fprintf(file.data, "%+.15le\t", p_host.vy[i]);
#if DIM == 3
            fprintf(file.data, "%+.15le\t", p_host.vz[i]);
#endif
#endif
            fprintf(file.data, "%e\t", p_host.m[i]);
            fprintf(file.data, "%e\t", p_host.rho[i]);
#if INTEGRATE_ENERGY
            fprintf(file.data, "%e\t", p_host.e[i]);
#endif
            fprintf(file.data, "%.6le\t", p_host.h[i]);
            fprintf(file.data, "%d\t", p_host.noi[i]);
            fprintf(file.data, "%d\t", p_host.materialId[i]);
#if JC_PLASTICITY
            fprintf(file.data, "%+.6le\t", p_host.ep[i]);
            fprintf(file.data, "%+.6le\t", p_host.T[i]);
#endif
#if FRAGMENTATION
            fprintf(file.data, "%d\t", p_host.numFlaws[i]);
            fprintf(file.data, "%d\t", p_host.numActiveFlaws[i]);
            fprintf(file.data, "%.6le\t", p_host.d[i]);
#endif
#if !PALPHA_POROSITY
            fprintf(file.data, "%e\t", p_host.p[i]);
#endif
#if SOLID
            fprintf(file.data, "%.6le\t", p_host.local_strain[i]);
            for (j = 0; j < DIM; j++) {
                for (k = 0; k < DIM; k++) {
                    fprintf(file.data, "%.6le\t", p_host.S[i*DIM*DIM+j*DIM+k]);
                }
            }
#endif
#if NAVIER_STOKES
            for (j = 0; j < DIM; j++) {
                for (k = 0; k < DIM; k++) {
                    fprintf(file.data, "%.6le\t", p_host.Tshear[i*DIM*DIM+j*DIM+k]);
                }
            }
#endif
#if SIRONO_POROSITY
            fprintf(file.data, "%.6le\t", p_host.rho_0prime[i]);
            fprintf(file.data, "%.6le\t", p_host.rho_c_plus[i]);
            fprintf(file.data, "%.6le\t", p_host.rho_c_minus[i]);
            fprintf(file.data, "%.6le\t", p_host.compressive_strength[i]);
            fprintf(file.data, "%.6le\t", p_host.tensile_strength[i]);
            fprintf(file.data, "%.6le\t", p_host.K[i]);
            fprintf(file.data, "%.d\t", p_host.flag_rho_0prime[i]);
            fprintf(file.data, "%.d\t", p_host.flag_plastic[i]);
            fprintf(file.data, "%.6le\t", p_host.shear_strength[i]);
#endif
#if PALPHA_POROSITY
            fprintf(file.data, "%.6le\t", p_host.alpha_jutzi[i]);
            fprintf(file.data, "%.6le\t", p_host.p[i]);
#endif
#if EPSALPHA_POROSITY
            fprintf(file.data, "%.6le\t", p_host.alpha_epspor[i]);
            fprintf(file.data, "%.6le\t", p_host.epsilon_v[i]);
#endif
#if FRAGMENTATION
            for (d = 0; d < p_host.numFlaws[i]; d++) {
                fprintf(file.data, "%.6le\t", p_host.flaws[i*maxNumFlaws_host+d]);
            }
#endif
            fprintf(file.data, "\n");
        }
        fclose(file.data);
        if (param.verbose) {
            printf("wrote to %s.\n", file.name);
        }
    } /* ascii_output == TRUE */


    /* compute kin. energy, inner energy, lin. momentum, and ang. momentum (of all SPH particles that have not EOS_TYPE_IGNORE + all gravitating point masses) */
    totalpx = totalpy = totalpz = totalL = 0.0;
    for (d=0; d<DIM; d++) {
        totalangularmomentum[d] = 0.0;
    }
    numberOfParticlesToIgnore=0;    // number of particles with EOS_TYPE_IGNORE
    for (i=0; i<numberOfParticles; i++) {
        if( p_host.materialId[i] == EOS_TYPE_IGNORE ) {
            numberOfParticlesToIgnore++;
        }
        else {  // process only SPH particles that have not EOS_TYPE_IGNORE
            v = p_host.vx[i] * p_host.vx[i];
            totalpx += p_host.vx[i] * p_host.m[i];
#if DIM > 1
            v += p_host.vy[i] * p_host.vy[i];
            totalpy += p_host.vy[i] * p_host.m[i];
#if DIM == 3
            v += p_host.vz[i] * p_host.vz[i];
            totalpz += p_host.vz[i] * p_host.m[i];
#endif
#endif
            totalmass += p_host.m[i];
            totalkineticenergy += p_host.m[i] * v;
#if INTEGRATE_ENERGY
            totalinnerenergy += p_host.m[i] * p_host.e[i];
#endif
#if DIM == 2
            totalangularmomentum[0] += p_host.m[i] * (p_host.x[i]*p_host.vy[i] - p_host.y[i]*p_host.vx[i]);
#elif DIM > 2
            totalangularmomentum[0] += p_host.m[i] * (p_host.y[i]*p_host.vz[i] - p_host.z[i]*p_host.vy[i]);
            totalangularmomentum[1] += p_host.m[i] * (p_host.z[i]*p_host.vx[i] - p_host.x[i]*p_host.vz[i]);
            totalangularmomentum[2] += p_host.m[i] * (p_host.x[i]*p_host.vy[i] - p_host.y[i]*p_host.vx[i]);
#endif
        }
    }
#if GRAVITATING_POINT_MASSES
    diskmass += totalmass;
#if DIM == 2
    diskangularmomentum[0] += totalangularmomentum[0];
#elif DIM == 3
    diskangularmomentum[0] += totalangularmomentum[0];
    diskangularmomentum[1] += totalangularmomentum[1];
    diskangularmomentum[2] += totalangularmomentum[2];
#endif
    for(i=0; i<numberOfPointmasses; i++) {
        v = pointmass_host.vx[i] * pointmass_host.vx[i];
        totalpx += pointmass_host.vx[i] * pointmass_host.m[i];
#if DIM > 1
        v += pointmass_host.vy[i] * pointmass_host.vy[i];
        totalpy += pointmass_host.vy[i] * pointmass_host.m[i];
#if DIM == 3
        v += pointmass_host.vz[i] * pointmass_host.vz[i];
        totalpz += pointmass_host.vz[i] * pointmass_host.m[i];
#endif
#endif
        totalmass += pointmass_host.m[i];
        totalkineticenergy += pointmass_host.m[i] * v;
#if DIM == 2
        binaryangularmomentum[0] += pointmass_host.m[i] * (pointmass_host.x[i]*pointmass_host.vy[i] - pointmass_host.y[i]*pointmass_host.vx[i]);
        totalangularmomentum[0] += binaryangularmomentum[0];
#elif DIM > 2
        binaryangularmomentum[0] += pointmass_host.m[i] * (pointmass_host.y[i]*pointmass_host.vz[i] - pointmass_host.z[i]*pointmass_host.vy[i]);
        binaryangularmomentum[1] += pointmass_host.m[i] * (pointmass_host.z[i]*pointmass_host.vx[i] - pointmass_host.x[i]*pointmass_host.vz[i]);
        binaryangularmomentum[2] += pointmass_host.m[i] * (pointmass_host.x[i]*pointmass_host.vy[i] - pointmass_host.y[i]*pointmass_host.vx[i]);
        totalangularmomentum[0] += binaryangularmomentum[0];
        totalangularmomentum[1] += binaryangularmomentum[1];
        totalangularmomentum[2] += binaryangularmomentum[2];
#endif
    }
    for (d=0; d<DIM; d++) {
        totalDL += diskangularmomentum[d]*diskangularmomentum[d];
        totalBL += binaryangularmomentum[d]*binaryangularmomentum[d];
        }
    totalDL = sqrt(totalDL);
    totalBL = sqrt(totalBL);
#endif // GRAVITATINT_POINT_MASSES

    totalkineticenergy *= 0.5;
    totalp = totalpx*totalpx;
#if DIM > 1
    totalp += totalpy*totalpy;
#if DIM == 3
    totalp += totalpz*totalpz;
#endif
#endif
    totalp = sqrt(totalp);
    for (d=0; d<DIM; d++) {
        totalL += totalangularmomentum[d]*totalangularmomentum[d];
    }
    totalL = sqrt(totalL);

    /* compute grav. energy between all SPH particles, and also gravitating point masses (w.r.t. SPH particles and also among themselves) */
#if OUTPUT_GRAV_ENERGY
    totalgravenergy = 0.0;
    if( param.selfgravity || param.directselfgravity ) {
        // mutual grav. energy between SPH particles:
        for(i=0; i<(numberOfParticles-1); i++) {
            for (j=i+1; j<numberOfParticles; j++) {
                if( p_host.materialId[i] != EOS_TYPE_IGNORE  &&  p_host.materialId[j] != EOS_TYPE_IGNORE ) {
                    dist = pow(p_host.x[i]-p_host.x[j],2);
#if DIM > 1
                    dist += pow(p_host.y[i]-p_host.y[j],2);
#if DIM == 3
                    dist += pow(p_host.z[i]-p_host.z[j],2);
#endif
#endif
                    totalgravenergy -= C_GRAVITY * p_host.m[i] * p_host.m[j] / sqrt(dist);
                }
            }
        }
    }
    if( GRAVITATING_POINT_MASSES ) {
        // mutual grav. energy between SPH particles and gravitating point masses:
        for(i=0; i<numberOfParticles; i++) {
            if( p_host.materialId[i] != EOS_TYPE_IGNORE ) {
                for(j=0; j<numberOfPointmasses; j++) {
                    dist = pow(p_host.x[i]-pointmass_host.x[j],2);
#if DIM > 1
                    dist += pow(p_host.y[i]-pointmass_host.y[j],2);
#if DIM == 3
                    dist += pow(p_host.z[i]-pointmass_host.z[j],2);
#endif
#endif
                    totalgravenergy -= C_GRAVITY * p_host.m[i] * pointmass_host.m[j] / sqrt(dist);
                }
            }
        }
        // mutual grav. energy between gravitating point masses:
        for(i=0; i<(numberOfPointmasses-1); i++) {
            for (j=i+1; j<numberOfPointmasses; j++) {
                dist = pow(pointmass_host.x[i]-pointmass_host.x[j],2);
#if DIM > 1
                dist += pow(pointmass_host.y[i]-pointmass_host.y[j],2);
#if DIM == 3
                dist += pow(pointmass_host.z[i]-pointmass_host.z[j],2);
#endif
#endif
                totalgravenergy -= C_GRAVITY * pointmass_host.m[i] * pointmass_host.m[j] / sqrt(dist);
            }
        }
    }
#endif  // OUTPUT_GRAV_ENERGY

    /* compute position and velocity of the barycenter (of all SPH particles + gravitating point masses) */
    for(i=0; i<DIM; i++)
        barycenter_pos[i] = barycenter_vel[i] = 0.0;
    for(i=0; i<numberOfParticles; i++) {
        if( p_host.materialId[i] != EOS_TYPE_IGNORE ) {
            barycenter_pos[0] += p_host.m[i] * p_host.x[i];
            barycenter_vel[0] += p_host.m[i] * p_host.vx[i];
#if DIM > 1
            barycenter_pos[1] += p_host.m[i] * p_host.y[i];
            barycenter_vel[1] += p_host.m[i] * p_host.vy[i];
#if DIM == 3
            barycenter_pos[2] += p_host.m[i] * p_host.z[i];
            barycenter_vel[2] += p_host.m[i] * p_host.vz[i];
#endif
#endif
        }
    }
    if( GRAVITATING_POINT_MASSES ) {
        for(i=0; i<numberOfPointmasses; i++) {
            barycenter_pos[0] += pointmass_host.m[i] * pointmass_host.x[i];
            barycenter_vel[0] += pointmass_host.m[i] * pointmass_host.vx[i];
#if DIM > 1
            barycenter_pos[1] += pointmass_host.m[i] * pointmass_host.y[i];
            barycenter_vel[1] += pointmass_host.m[i] * pointmass_host.vy[i];
#if DIM == 3
            barycenter_pos[2] += pointmass_host.m[i] * pointmass_host.z[i];
            barycenter_vel[2] += pointmass_host.m[i] * pointmass_host.vz[i];
#endif
#endif
        }
    }
    for(i=0; i<DIM; i++) {
        barycenter_pos[i] /= totalmass;
        barycenter_vel[i] /= totalmass;
    }

#if GRAVITATING_POINT_MASSES
    if (param.ascii_output) {
        // write mass file
        /* write info file name */
        if (param.verbose) {
            printf("writing info file %s.mass.\n", file.name);
        }
        if ((massfile = fopen(massfilename, "w")) == NULL) {
            fprintf(stderr, "Eih? Cannot write to %s.\n", massfilename);
            exit(1);
        }

        for (i = 0; i < numberOfPointmasses; i++) {
            fprintf(massfile, "%+.15le\t", pointmass_host.x[i]);
    #if DIM > 1
            fprintf(massfile, "%+.15le\t", pointmass_host.y[i]);
    #if DIM == 3
            fprintf(massfile, "%+.15le\t", pointmass_host.z[i]);
    #endif
    #endif
            fprintf(massfile, "%+.15le\t", pointmass_host.vx[i]);
    #if DIM > 1
            fprintf(massfile, "%+.15le\t", pointmass_host.vy[i]);
    #if DIM == 3
            fprintf(massfile, "%+.15le\t", pointmass_host.vz[i]);
    #endif
    #endif
            fprintf(massfile, "%+.17le\t", pointmass_host.m[i]);
            fprintf(massfile, "%+.17le\t", pointmass_host.rmin[i]);
            fprintf(massfile, "%+.17le\t", pointmass_host.rmax[i]);
            fprintf(massfile, "%d", pointmass_host.feels_particles[i]);
            fprintf(massfile, "\n");
        }
        fclose(massfile);
    } // param.ascii_output

// calculate semi-major axis and eccentricity of binary system and write in a file
#if BINARY_INFO
    double distance, velocity, h, r_x, r_y, r_z, v_x, v_y, v_z, h_x, h_y, h_z;
    double a_binary, ecc;

#if DIM == 2
    distance = sqrt( (pointmass_host.x[0]-pointmass_host.x[1])*(pointmass_host.x[0]-pointmass_host.x[1]) + (pointmass_host.y[0]-pointmass_host.y[1])*(pointmass_host.y[0]-pointmass_host.y[1]) );
    velocity = sqrt( (pointmass_host.vx[0]-pointmass_host.vx[1])*(pointmass_host.vx[0]-pointmass_host.vx[1]) + (pointmass_host.vy[0]-pointmass_host.vy[1])*(pointmass_host.vy[0]-pointmass_host.vy[1]) );
    // angular momentum
    r_x = pointmass_host.x[0] - pointmass_host.x[1];
    r_y = pointmass_host.y[0] - pointmass_host.y[1];
    v_x = pointmass_host.vx[0] - pointmass_host.vx[1];
    v_y = pointmass_host.vy[0] - pointmass_host.vy[1];
    h_z = r_x*v_y - r_y*v_x;
    h = sqrt(h_z*h_z);
#endif // DIM == 2

#if DIM == 3
    distance = sqrt( (pointmass_host.x[0]-pointmass_host.x[1])*(pointmass_host.x[0]-pointmass_host.x[1]) + (pointmass_host.y[0]-pointmass_host.y[1])*(pointmass_host.y[0]-pointmass_host.y[1]) + (pointmass_host.z[0]-pointmass_host.z[1])*(pointmass_host.z[0]-pointmass_host.z[1]) );
    velocity = sqrt( (pointmass_host.vx[0]-pointmass_host.vx[1])*(pointmass_host.vx[0]-pointmass_host.vx[1]) + (pointmass_host.vy[0]-pointmass_host.vy[1])*(pointmass_host.vy[0]-pointmass_host.vy[1]) + (pointmass_host.vz[0]-pointmass_host.vz[1])*(pointmass_host.vz[0]-pointmass_host.vz[1]) );
    // angular momentum
	r_x = pointmass_host.x[0] - pointmass_host.x[1];
    r_y = pointmass_host.y[0] - pointmass_host.y[1];
    r_z = pointmass_host.z[0] - pointmass_host.z[1];
    v_x = pointmass_host.vx[0] - pointmass_host.vx[1];
    v_y = pointmass_host.vy[0] - pointmass_host.vy[1];
    v_z = pointmass_host.vz[0] - pointmass_host.vz[1];
    h_x = r_y*v_z - r_z*v_y;
    h_y = r_z*v_x - r_x*v_z;
    h_z = r_x*v_y - r_y*v_x;
    h = sqrt(h_x*h_x + h_y*h_y + h_z*h_z);
#endif // DIM == 3

    a_binary = 1 / ( 2 / distance - velocity*velocity / (C_GRAVITY*(pointmass_host.m[0] + pointmass_host.m[1])) );   // semi-major axis of binary system
    ecc = sqrt( 1 - h*h / (C_GRAVITY*(pointmass_host.m[0] + pointmass_host.m[1])*a_binary) );    // eccentricity of binary system

    /* write binary system file*/
    if( (binarysystemfile = fopen(param.binarysystemfilename, "a")) == NULL ) {
        fprintf(stderr, "Ohoh..Merry Xmas... Cannot open '%s' for appending. Abort...\n", param.binarysystemfilename);
        exit(1);
    }
    fprintf(binarysystemfile, "%26.10le\t%.10le\t%.10le\t%.17le\n", h5time, a_binary, ecc, totalBL);
    fclose(binarysystemfile);
#endif // BINARY_INFO
#endif // GRAVITATING_POINT_MASSES

    /* write info file */
    if (param.verbose) {
        printf("writing info file %s.info.\n", file.name);
    }
    if ((infofile = fopen(infofilename, "w")) == NULL) {
        fprintf(stderr, "Eih? Cannot write to %s.\n", infofilename);
        exit(1);
    }
    fprintf(infofile, "Information about particle data of file %s\n", file.name);
    fprintf(infofile, "Time: \t %.20e\n", h5time);
    fprintf(infofile, "Number of SPH particles (total): \t %d\n", numberOfParticles);
#if GRAVITATING_POINT_MASSES // info file dedicated to disk only. future UPDATE: columns table format instead of rows
    fprintf(infofile, "Number of deactivated SPH particles (EOS_TYPE_IGNORED): \t %d\n", numberOfParticlesToIgnore);
    fprintf(infofile, "Number of gravitating point masses: \t %d\n", numberOfPointmasses);
    fprintf(infofile, "Disk mass: \t %.20e\n", diskmass);
    fprintf(infofile, "Total (disk+binary) kinetic energy: \t %.20e\n", totalkineticenergy);
    fprintf(infofile, "Total (disk+binary) internal energy: \t %.20e\n", totalinnerenergy);
    fprintf(infofile, "Total (disk+binary) momentum: \t %.20e\n", totalp);
    fprintf(infofile, "Total momentum in each direction: \t %.20e ", totalpx);
#if DIM > 1
    fprintf(infofile, "  %.20e ", totalpy);
#if DIM == 3
    fprintf(infofile, "  %.20e ", totalpz);
#endif
#endif
    fprintf(infofile, "\n");
    fprintf(infofile, "Disk angular momentum:   ");
    for (d=0; d<DIM; d++)
        fprintf(infofile, "L[%d] = %.20e \t", d, diskangularmomentum[d]);
    fprintf(infofile, "\n");
    fprintf(infofile, "Disk angular momentum: norm(L) = %.20e\n", totalDL);
#if OUTPUT_GRAV_ENERGY
    fprintf(infofile, "\n");
    fprintf(infofile, "Total (disk+binary) grav. energy: \t %.20e\n", totalgravenergy);
#endif
#else // general info file for total quantities
    fprintf(infofile, "Total mass: \t %.20e\n", totalmass);
    fprintf(infofile, "Total kinetic energy: \t %.20e\n", totalkineticenergy);
    fprintf(infofile, "Total internal energy: \t %.20e\n", totalinnerenergy);
    fprintf(infofile, "Total  momentum: \t %.20e\n", totalp);
    fprintf(infofile, "Total momentum in each direction: \t %.20e ", totalpx);
#if DIM > 1
    fprintf(infofile, "  %.20e ", totalpy);
#if DIM == 3
    fprintf(infofile, "  %.20e ", totalpz);
#endif
#endif
    fprintf(infofile, "\n");
    fprintf(infofile, "Total angular momentum:   ");
    for (d=0; d<DIM; d++)
        fprintf(infofile, "L[%d] = %.20e \t", d, totalangularmomentum[d]);
    fprintf(infofile, "\n");
    fprintf(infofile, "Total angular momentum: norm(L) = %.20e\n", totalL);
#if OUTPUT_GRAV_ENERGY
    fprintf(infofile, "\n");
    fprintf(infofile, "Total grav. energy: \t %.20e\n", totalgravenergy);
#endif
#endif // GRAVITATING_POINT_MASSES
    fclose(infofile);

    /* append to conserved quantities logfile */
    if( (conservedquantitiesfile = fopen(param.conservedquantitiesfilename, "a")) == NULL ) {
        fprintf(stderr, "Ohoh... Cannot open '%s' for appending. Abort...\n", param.conservedquantitiesfilename);
        exit(1);
    }
    fprintf(conservedquantitiesfile, "%26.16le%18d%24d%21d%26.16le%26.16le%26.16le", h5time, numberOfParticles, numberOfParticlesToIgnore, numberOfPointmasses, totalmass, totalkineticenergy, totalinnerenergy);
#if OUTPUT_GRAV_ENERGY
    fprintf(conservedquantitiesfile, "%26.16le", totalgravenergy);
#endif
    fprintf(conservedquantitiesfile, "%26.16le%26.16le", totalp, totalpx);
#if DIM > 1
    fprintf(conservedquantitiesfile, "%26.16le", totalpy);
#if DIM == 3
    fprintf(conservedquantitiesfile, "%26.16le", totalpz);
#endif
#endif
#if DIM > 1
    fprintf(conservedquantitiesfile, "%26.16le%26.16le%26.16le", totalL, totalangularmomentum[0], totalangularmomentum[1]);
#if DIM == 3
    fprintf(conservedquantitiesfile, "%26.16le", totalangularmomentum[2]);
#endif
#endif
    fprintf(conservedquantitiesfile, "%26.16le", barycenter_pos[0]);
#if DIM > 1
    fprintf(conservedquantitiesfile, "%26.16le", barycenter_pos[1]);
#if DIM == 3
    fprintf(conservedquantitiesfile, "%26.16le", barycenter_pos[2]);
#endif
#endif
    fprintf(conservedquantitiesfile, "%26.16le", barycenter_vel[0]);
#if DIM > 1
    fprintf(conservedquantitiesfile, "%26.16le", barycenter_vel[1]);
#if DIM == 3
    fprintf(conservedquantitiesfile, "%26.16le", barycenter_vel[2]);
#endif
#endif
    fprintf(conservedquantitiesfile, "\n");
    fclose(conservedquantitiesfile);

#if HDF5IO
    if (param.hdf5output) {
        if (param.verbose) {
            printf("writing to %s.h5.\n", file.name);
        }
        file_id = H5Fcreate(h5filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        /* the positions */
        dims[0] = numberOfParticles;
        dims[1] = DIM;
        dataspace_id =  H5Screate_simple(2, dims, NULL);
        x_id = H5Dcreate2(file_id, "/x", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        v_id = H5Dcreate2(file_id, "/v", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        a_id = H5Dcreate2(file_id, "/a", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        g_a_id = H5Dcreate2(file_id, "/a_grav", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


        x = (double *) malloc(sizeof(double) * numberOfParticles * DIM);
        for (i = 0, e = 0; i < numberOfParticles; i++, e += DIM) {
            x[e] = p_host.x[i];
#if DIM > 1
            x[e+1] = p_host.y[i];
#if DIM == 3
            x[e+2] = p_host.z[i];
#endif
#endif
        }

        status = H5Dwrite(x_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(x_id);

        /* the velocities */
        for (i = 0, e = 0; i < numberOfParticles; i++, e += DIM) {
            x[e] = p_host.vx[i];
#if DIM > 1
            x[e+1] = p_host.vy[i];
#if DIM == 3
            x[e+2] = p_host.vz[i];
#endif
#endif
        }

        status = H5Dwrite(v_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(v_id);


        /* the velocities at time of possible accretion */
        for (i = 0, e = 0; i < numberOfParticles; i++, e += DIM) {
            x[e] = p_host.vx0[i];
#if DIM > 1
            x[e+1] = p_host.vy0[i];
#if DIM == 3
            x[e+2] = p_host.vz0[i];
#endif
#endif
        }
        v_id = H5Dcreate2(file_id, "/v_accreted", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        status = H5Dwrite(v_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(v_id);

        /* the accelerations */
        for (i = 0, e = 0; i < numberOfParticles; i++, e += DIM) {
            x[e] = p_host.ax[i];
#if DIM > 1
            x[e+1] = p_host.ay[i];
#if DIM == 3
            x[e+2] = p_host.az[i];
#endif
#endif
        }
        status = H5Dwrite(a_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(a_id);

        /* the accelerations due to gravity */
        for (i = 0, e = 0; i < numberOfParticles; i++, e += DIM) {
            x[e] = p_host.g_ax[i];
#if DIM > 1
            x[e+1] = p_host.g_ay[i];
#if DIM == 3
            x[e+2] = p_host.g_az[i];
#endif
#endif
        }
        status = H5Dwrite(g_a_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(g_a_id);

        free(x);
        x = (double *)  malloc(sizeof(double) * numberOfParticles);

        /* time */
        dims[0] = 1;
        dataspace_id = H5Screate_simple(1, dims, NULL);
        time_id = H5Dcreate2(file_id, "/time", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(time_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &h5time);
        status = H5Dclose(time_id);

        /* mass */
        dims[0] = numberOfParticles;
        dims[1] = 1;
        dataspace_id = H5Screate_simple(1, dims, NULL);
        m_id = H5Dcreate2(file_id, "/m", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.m[i];

        status = H5Dwrite(m_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(m_id);

        /* density */
        rho_id = H5Dcreate2(file_id, "/rho", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.rho[i];

        status = H5Dwrite(rho_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rho_id);

        /* energy */
        e_id = H5Dcreate2(file_id, "/e", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.e[i];

        status = H5Dwrite(e_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(e_id);

#if MORE_ANEOS_OUTPUT
        // compute ANEOS quantities
        int aneos_i_rho, aneos_i_e;
        double *x_aneos_T = (double *)malloc(numberOfParticles*sizeof(double));
        double *x_aneos_cs = (double *)malloc(numberOfParticles*sizeof(double));
        double *x_aneos_entropy = (double *)malloc(numberOfParticles*sizeof(double));
        int *x_aneos_phase_flag = (int *)malloc(numberOfParticles*sizeof(int));
        for(i=0; i<numberOfParticles; i++) {
            j = p_host.materialId[i];
            if( g_eos_is_aneos[j] == TRUE ) {
                aneos_i_rho = array_index_host(p_host.rho[i], g_aneos_rho[j], g_aneos_n_rho[j]);
                aneos_i_e = array_index_host(p_host.e[i], g_aneos_e[j], g_aneos_n_e[j]);
                x_aneos_T[i] = bilinear_interpolation_from_matrix(p_host.rho[i], p_host.e[i], g_aneos_T[j], g_aneos_rho[j], g_aneos_e[j], aneos_i_rho, aneos_i_e, g_aneos_n_rho[j], g_aneos_n_e[j]);
                x_aneos_cs[i] = bilinear_interpolation_from_matrix(p_host.rho[i], p_host.e[i], g_aneos_cs[j], g_aneos_rho[j], g_aneos_e[j], aneos_i_rho, aneos_i_e, g_aneos_n_rho[j], g_aneos_n_e[j]);
                x_aneos_entropy[i] = bilinear_interpolation_from_matrix(p_host.rho[i], p_host.e[i], g_aneos_entropy[j], g_aneos_rho[j], g_aneos_e[j], aneos_i_rho, aneos_i_e, g_aneos_n_rho[j], g_aneos_n_e[j]);
                x_aneos_phase_flag[i] = discrete_value_table_lookup_from_matrix(p_host.rho[i], p_host.e[i], g_aneos_phase_flag[j], g_aneos_rho[j], g_aneos_e[j], aneos_i_rho, aneos_i_e, g_aneos_n_rho[j], g_aneos_n_e[j]);
            }
            else {
                x_aneos_T[i] = -1.0;
                x_aneos_cs[i] = -1.0;
                x_aneos_entropy[i] = -1.0;
                x_aneos_phase_flag[i] = -1;
            }
        }
        // write them to the output file
        aneos_T_id = H5Dcreate2(file_id, "/aneos_T", H5T_NATIVE_DOUBLE, dataspace_id,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(aneos_T_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x_aneos_T);
        status = H5Dclose(aneos_T_id);
        aneos_cs_id = H5Dcreate2(file_id, "/aneos_cs", H5T_NATIVE_DOUBLE, dataspace_id,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(aneos_cs_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x_aneos_cs);
        status = H5Dclose(aneos_cs_id);
        aneos_entropy_id = H5Dcreate2(file_id, "/aneos_entropy", H5T_NATIVE_DOUBLE, dataspace_id,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(aneos_entropy_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x_aneos_entropy);
        status = H5Dclose(aneos_entropy_id);
        aneos_phase_flag_id = H5Dcreate2(file_id, "/aneos_phase_flag", H5T_NATIVE_INT, dataspace_id,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(aneos_phase_flag_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x_aneos_phase_flag);
        status = H5Dclose(aneos_phase_flag_id);

        // free memory
        free(x_aneos_T);
        free(x_aneos_cs);
        free(x_aneos_entropy);
        free(x_aneos_phase_flag);
#endif
#if MORE_OUTPUT
	p_max_id = H5Dcreate2(file_id, "/p_max", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.p_max[i];

        status = H5Dwrite(p_max_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(p_max_id);


	p_min_id = H5Dcreate2(file_id, "/p_min", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.p_min[i];

        status = H5Dwrite(p_min_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(p_min_id);


	rho_min_id = H5Dcreate2(file_id, "/rho_min", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.rho_min[i];

        status = H5Dwrite(rho_min_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rho_min_id);


	rho_max_id = H5Dcreate2(file_id, "/rho_max", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.rho_max[i];

        status = H5Dwrite(rho_max_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rho_max_id);

	e_min_id = H5Dcreate2(file_id, "/e_min", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.e_min[i];

        status = H5Dwrite(e_min_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(e_min_id);


	e_max_id = H5Dcreate2(file_id, "/e_max", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.e_max[i];

        status = H5Dwrite(e_max_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(e_max_id);

	cs_min_id = H5Dcreate2(file_id, "/cs_min", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.cs_min[i];

        status = H5Dwrite(cs_min_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(cs_min_id);


	cs_max_id = H5Dcreate2(file_id, "/cs_max", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.cs_max[i];

        status = H5Dwrite(cs_max_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(cs_max_id);

#endif

        /* sml */
        sml_id = H5Dcreate2(file_id, "/sml", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++) {
#if (VARIABLE_SML || INTEGRATE_SML || DEAL_WITH_TOO_MANY_INTERACTIONS || READ_INITIAL_SML_FROM_PARTICLE_FILE)
            x[i] = p_host.h[i];
#else
            x[i] = sml[p_host.materialId[i]];
#endif
        }

        status = H5Dwrite(sml_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(sml_id);
#if READ_INITIAL_SML_FROM_PARTICLE_FILE
        /* sml initial */
        sml_id = H5Dcreate2(file_id, "/sml_initial", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++) {
            x[i] = p_host.h0[i];
        }
        status = H5Dwrite(sml_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(sml_id);
#endif


#if JC_PLASTICITY
        /* plastic strain */
        ep_id = H5Dcreate2(file_id, "/ep", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.ep[i];

        status = H5Dwrite(ep_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(ep_id);

        /* temperature */
        T_id = H5Dcreate2(file_id, "/T", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.T[i];

        status = H5Dwrite(T_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(T_id);
#endif



        /* number of interactions */
        ix = (int *) malloc(sizeof(int) * numberOfParticles);

        noi_id = H5Dcreate2(file_id, "/number_of_interactions", H5T_NATIVE_INT, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        for (i = 0; i < numberOfParticles; i++)
            ix[i] = p_host.noi[i];

        status = H5Dwrite(noi_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ix);
        status = H5Dclose(noi_id);

        /* material type */
        mtype_id = H5Dcreate2(file_id, "/material_type", H5T_NATIVE_INT, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        for (i = 0; i < numberOfParticles; i++)
            ix[i] = p_host.materialId[i];

        status = H5Dwrite(mtype_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ix);
        status = H5Dclose(mtype_id);


        /* pressure */
        p_id = H5Dcreate2(file_id, "/p", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.p[i];

        status = H5Dwrite(p_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(p_id);


#if PALPHA_POROSITY
        /* alpha_jutzi */
        alpha_id = H5Dcreate2(file_id, "/alpha_jutzi", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.alpha_jutzi[i];

        status = H5Dwrite(alpha_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(alpha_id);

#endif
        /* soundspeed */
        cs_id = H5Dcreate2(file_id, "/soundspeed", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.cs[i];

        status = H5Dwrite(cs_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(cs_id);

#if SIRONO_POROSITY
        /* compressive strength */
        compressive_strength_id = H5Dcreate2(file_id, "/compressive_strength", H5T_NATIVE_DOUBLE, dataspace_id,
                                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.compressive_strength[i];

        status = H5Dwrite(compressive_strength_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(compressive_strength_id);

        /* tensile strength */
        tensile_strength_id = H5Dcreate2(file_id, "/tensile_strength", H5T_NATIVE_DOUBLE, dataspace_id,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.tensile_strength[i];

        status = H5Dwrite(tensile_strength_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(tensile_strength_id);

        /* bulk modulus */
        K_id = H5Dcreate2(file_id, "/K", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.K[i];

        status = H5Dwrite(K_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(K_id);

        /* density_0prime */
        rho_0prime_id = H5Dcreate2(file_id, "/rho_0prime", H5T_NATIVE_DOUBLE, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.rho_0prime[i];

        status = H5Dwrite(rho_0prime_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rho_0prime_id);

        /* density_c_plus */
        rho_c_plus_id = H5Dcreate2(file_id, "/rho_c_plus", H5T_NATIVE_DOUBLE, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.rho_c_plus[i];

        status = H5Dwrite(rho_c_plus_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rho_c_plus_id);

        /* density_c_minus */
        rho_c_minus_id = H5Dcreate2(file_id, "rho_c_minus", H5T_NATIVE_DOUBLE, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.rho_c_minus[i];

        status = H5Dwrite(rho_c_minus_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rho_c_minus_id);

        /* flag_rho0_prime */
        flag_rho_0prime_id = H5Dcreate2(file_id, "/flag_rho_0prime", H5T_NATIVE_INT, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            ix[i] = p_host.flag_rho_0prime[i];

        status = H5Dwrite(flag_rho_0prime_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ix);
        status = H5Dclose(flag_rho_0prime_id);

        /* flag_plastic */
        flag_plastic_id = H5Dcreate2(file_id, "/flag_plastic", H5T_NATIVE_INT, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            ix[i] = p_host.flag_plastic[i];

        status = H5Dwrite(flag_plastic_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ix);
        status = H5Dclose(flag_plastic_id);

        /* shear_strength */
        shear_strength_id = H5Dcreate2(file_id, "shear_strength", H5T_NATIVE_DOUBLE, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.shear_strength[i];

        status = H5Dwrite(shear_strength_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(shear_strength_id);
#endif

#if EPSALPHA_POROSITY
        /* alpha_epspor */
        alpha_epspor_id = H5Dcreate2(file_id, "/alpha_epspor", H5T_NATIVE_DOUBLE, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.alpha_epspor[i];

        status = H5Dwrite(alpha_epspor_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(alpha_epspor_id);

        /* epsilon_v */
        epsilon_v_id = H5Dcreate2(file_id, "/epsilon_v", H5T_NATIVE_DOUBLE, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.epsilon_v[i];

        status = H5Dwrite(epsilon_v_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(epsilon_v_id);
#endif

#if FRAGMENTATION
        /* number of activated flaws */
        noaf_id = H5Dcreate2(file_id, "/number_of_activated_flaws", H5T_NATIVE_INT, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        for (i = 0; i < numberOfParticles; i++)
            ix[i] = p_host.numActiveFlaws[i];

        status = H5Dwrite(noaf_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ix);
        status = H5Dclose(noaf_id);

        /* damage */
        damage_id = H5Dcreate2(file_id, "/DIM_root_of_damage_tensile", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.d[i];

        status = H5Dwrite(damage_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(damage_id);

#if PALPHA_POROSITY
        /* damage porjutzi */
        damage_porjutzi_id = H5Dcreate2(file_id, "/DIM_root_of_damage_porjutzi", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.damage_porjutzi[i];

        status = H5Dwrite(damage_porjutzi_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(damage_porjutzi_id);

        /* damage total */
        damage_total_id = H5Dcreate2(file_id, "/damage_total", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++) {
            x[i] = pow(p_host.damage_porjutzi[i], DIM) + pow(p_host.d[i], DIM);
            if (x[i] > 1.0)
                x[i] = 1.0;
        }

        status = H5Dwrite(damage_total_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(damage_total_id);
#endif

        /* find maximum number of flaws of all particles */
        maxnof = -1;
        for (i = 0; i < numberOfParticles; i++) {
            if (p_host.numFlaws[i] > maxnof)
                maxnof = p_host.numFlaws[i];
        }

        /* write maximum number of flaws */
        dims[0] = 1;
        dataspace_id = H5Screate_simple(1, dims, NULL);
        maxnof_id = H5Dcreate2(file_id, "/maximum_number_of_flaws", H5T_NATIVE_INT, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(maxnof_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &maxnof);
        status = H5Dclose(maxnof_id);

        /* we will use maxnof for the flaws storage data set */
        if (maxnof > 0) {
            dims[0] = numberOfParticles;
            dims[1] = maxnof;

            fprintf(stdout, "Using %d doubles for flaws in hdf5 output.\n", maxnof);
            dataspace_id = H5Screate_simple(2, dims, NULL);

            activation_thresholds_id = H5Dcreate2(file_id, "/activation_thresholds", H5T_NATIVE_DOUBLE, dataspace_id,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            free(x);
            x = (double *) malloc(sizeof(double) * numberOfParticles * maxnof);

            for (i = 0; i < numberOfParticles; i++) {
                for (d = 0; d < p_host.numFlaws[i]; d++) {
                    x[i*maxnof + d] = p_host.flaws[i*maxNumFlaws_host+d];
                }
                /* fill up with -1 */
                for (d = p_host.numFlaws[i]; d < maxnof; d++)
                    x[i*maxnof + d] = -1;
            }
            status = H5Dwrite(activation_thresholds_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
            status = H5Dclose(activation_thresholds_id);
        }
#endif
        free(ix);
        free(x);
#if SOLID
        /* the deviatoric stress tensor */
        x = (double *) malloc(sizeof(double) * numberOfParticles * DIM * DIM);
        dims[0] = numberOfParticles;
        dims[1] = DIM*DIM;

        dataspace_id = H5Screate_simple(2, dims, NULL);
        S_id = H5Dcreate2(file_id, "/deviatoric_stress", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++) {
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    x[i*DIM*DIM + d*DIM + e] = p_host.S[i*DIM*DIM+d*DIM+e];
                }
            }
        }
        status = H5Dwrite(S_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(S_id);
        free(x);
#endif

#if NAVIER_STOKES
        /* the deviatoric stress tensor */
        x = (double *) malloc(sizeof(double) * numberOfParticles * DIM * DIM);
        dims[0] = numberOfParticles;
        dims[1] = DIM*DIM;

        dataspace_id = H5Screate_simple(2, dims, NULL);
        Tshear_id = H5Dcreate2(file_id, "/viscous_shear_stress", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++) {
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    x[i*DIM*DIM + d*DIM + e] = p_host.Tshear[i*DIM*DIM+d*DIM+e];
                }
            }
        }
        status = H5Dwrite(Tshear_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(Tshear_id);
        free(x);
#endif


        /* write even more stuff: drhodt, dedt, dSdt, cs, depth, accels */

#if SOLID
        /* the change of the deviatoric stress tensor */
        x = (double *) malloc(sizeof(double) * numberOfParticles * DIM * DIM);
        dims[0] = numberOfParticles;
        dims[1] = DIM*DIM;

        dataspace_id = H5Screate_simple(2, dims, NULL);
        dSdt_id = H5Dcreate2(file_id, "/ddeviatoric_stress_dt", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++) {
            for (d = 0; d < DIM; d++) {
                for (e = 0; e < DIM; e++) {
                    x[i*DIM*DIM + d*DIM + e] = p_host.dSdt[i*DIM*DIM+d*DIM+e];
                }
            }
        }
        status = H5Dwrite(dSdt_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(dSdt_id);
        free(x);
#endif
        dims[0] = numberOfParticles;
        dims[1] = 1;
        dataspace_id = H5Screate_simple(1, dims, NULL);
        /* depth in tree */
        ix = (int *) malloc(sizeof(int) * numberOfParticles);
        depth_id = H5Dcreate2(file_id, "/tree_depth", H5T_NATIVE_INT, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        for (i = 0; i < numberOfParticles; i++)
            ix[i] = p_host.depth[i];

        status = H5Dwrite(depth_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ix);
        status = H5Dclose(depth_id);


        free(ix);

#if INTEGRATE_DENSITY
        /* change of density */
        x = (double *) malloc(sizeof(double) * numberOfParticles);
        drhodt_id = H5Dcreate2(file_id, "/drhodt", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.drhodt[i];

        status = H5Dwrite(drhodt_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(drhodt_id);
        free(x);
#endif
#if SOLID
        /* local strain */
        x = (double *) malloc(sizeof(double) * numberOfParticles);
        local_strain_id = H5Dcreate2(file_id, "/local_strain", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.local_strain[i];

        status = H5Dwrite(local_strain_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(local_strain_id);
        free(x);
#endif

#if INTEGRATE_ENERGY
        /* change of energy */
        x = (double *) malloc(sizeof(double) * numberOfParticles);
        dedt_id = H5Dcreate2(file_id, "/dedt", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.dedt[i];

        status = H5Dwrite(dedt_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(dedt_id);
        free(x);
#endif

#if FRAGMENTATION
        /* change of damage */
        x = (double *) malloc(sizeof(double) * numberOfParticles);
        dddt_id = H5Dcreate2(file_id, "/dddt", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.dddt[i];

        status = H5Dwrite(dddt_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(dddt_id);
        free(x);
#if PALPHA_POROSITY
        /* change of damage porjutzi */
        x = (double *) malloc(sizeof(double) * numberOfParticles);
        ddamage_porjutzidt_id = H5Dcreate2(file_id, "/ddamage_porjutzidt", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.ddamage_porjutzidt[i];

        status = H5Dwrite(ddamage_porjutzidt_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(ddamage_porjutzidt_id);
        free(x);
#endif
#endif

#if PALPHA_POROSITY
        /* change of alpha_jutzi */
        x = (double *) malloc(sizeof(double) * numberOfParticles);
        dalphadt_id = H5Dcreate2(file_id, "/dalphadt", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfParticles; i++)
            x[i] = p_host.dalphadt[i];

        status = H5Dwrite(dalphadt_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(dalphadt_id);
        free(x);
#endif

        status = H5Fclose(file_id);


#if GRAVITATING_POINT_MASSES

        if (param.verbose) {
            printf("writing to %s.mass.h5.\n", file.name);
        }
        file_id = H5Fcreate(h5massfilename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        /* the positions */
        dims[0] = numberOfPointmasses;
        dims[1] = DIM;
        dataspace_id =  H5Screate_simple(2, dims, NULL);
        x_id = H5Dcreate2(file_id, "/x", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        v_id = H5Dcreate2(file_id, "/v", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        a_id = H5Dcreate2(file_id, "/a", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        x = (double *) malloc(sizeof(double) * numberOfPointmasses * DIM);
        for (i = 0, e = 0; i < numberOfPointmasses; i++, e += DIM) {
            x[e] = pointmass_host.x[i];
#if DIM > 1
            x[e+1] = pointmass_host.y[i];
#if DIM == 3
            x[e+2] = pointmass_host.z[i];
#endif
#endif
        }

        status = H5Dwrite(x_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(x_id);

        /* the velocities */
        for (i = 0, e = 0; i < numberOfPointmasses; i++, e += DIM) {
            x[e] = pointmass_host.vx[i];
#if DIM > 1
            x[e+1] = pointmass_host.vy[i];
#if DIM == 3
            x[e+2] = pointmass_host.vz[i];
#endif
#endif
        }

        status = H5Dwrite(v_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(v_id);


        /* the accelerations */
        for (i = 0, e = 0; i < numberOfPointmasses; i++, e += DIM) {
            x[e] = pointmass_host.ax[i];
#if DIM > 1
            x[e+1] = pointmass_host.ay[i];
#if DIM == 3
            x[e+2] = pointmass_host.az[i];
#endif
#endif
        }
        status = H5Dwrite(a_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(a_id);


        free(x);
        x = (double *)  malloc(sizeof(double) * numberOfPointmasses);

        /* time */
        dims[0] = 1;
        dataspace_id = H5Screate_simple(1, dims, NULL);
        time_id = H5Dcreate2(file_id, "/time", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(time_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &h5time);
        status = H5Dclose(time_id);

        /* mass */
        dims[0] = numberOfPointmasses;
        dims[1] = 1;
        dataspace_id = H5Screate_simple(1, dims, NULL);
        m_id = H5Dcreate2(file_id, "/m", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        for (i = 0; i < numberOfPointmasses; i++)
            x[i] = pointmass_host.m[i];

        status = H5Dwrite(m_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(m_id);

        /* rmin */
        rmin_id = H5Dcreate2(file_id, "/rmin", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfPointmasses; i++)
            x[i] = pointmass_host.rmin[i];

        status = H5Dwrite(rmin_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rmin_id);

        /* rmax */
        rmax_id = H5Dcreate2(file_id, "/rmax", H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfPointmasses; i++)
            x[i] = pointmass_host.rmax[i];

        status = H5Dwrite(rmax_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
        status = H5Dclose(rmax_id);

        free(x);

        ix = (int *) malloc(sizeof(int) * numberOfPointmasses);
        flag_id = H5Dcreate2(file_id, "/feels_particles", H5T_NATIVE_INT, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        for (i = 0; i < numberOfPointmasses; i++)
            ix[i] = pointmass_host.feels_particles[i];

        status = H5Dwrite(flag_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, ix);
        status = H5Dclose(flag_id);
        free(ix);



        status = H5Fclose(file_id);
#endif // GRAVITATING_POINT_MASSES

        if (param.verbose) {
            fprintf(stdout, "%d\n", status);
        }
    }

#endif // HDF5IO

}

void *write_timestep(void *argument)
{
    int timestep = *((int *) argument);
    if (timestep < 0) {
        fprintf(stderr, "special write because SIGTERM received...\n");
    }
    fprintf(stderr, "start printing %d \n", timestep+1);
    // last occurrence of .
    char *pch;
    pch = strrchr(inputFile.name, '.');
    // copy everything except the digits to the output filename
    File outputFile;
    strncpy(outputFile.name, inputFile.name, pch-inputFile.name+1);
    outputFile.name[pch-inputFile.name+1] = 0;
    // what was the start number of the input file?
    char inputFileNumber[256];
    memcpy(inputFileNumber, pch, strlen(pch)+1);
    int startNumber = atoi(inputFileNumber);
    // set output filename
    char outputFileEnding[256];
    sprintf(outputFileEnding, "%04d", startNumber+timestep+1);
    strcat(outputFile.name, outputFileEnding);
    // write
    write_particles_to_file(outputFile);

#if TREEDEBUG
    // write tree
    File treeFile;
    strcpy(treeFile.name, outputFile.name);
    strcat(treeFile.name, ".tree");
    write_tree_to_file(treeFile);
#endif
    if (timestep < 0) exit(0);

    //pthread_exit((void*) argument);
    fprintf(stderr, "end printing %d \n", timestep+1);
    currentDiskIO = FALSE;
    free(argument);
    return NULL;
}

void get_performance_file(File &file) {
    sprintf(file.name, "%010d.performance", numberOfParticles);
}

void clear_performance_file() {
    File file;
    get_performance_file(file);
    // open file for writing
    if ((file.data = fopen(file.name, "w")) == NULL) {
        fprintf(stderr, "Eih? Cannot write to %s.\n", file.name);
        exit(1);
    }
    fclose(file.data);
}

void write_performance(float *time) {
    File file;
    get_performance_file(file);
    // open file for writing
    if ((file.data = fopen(file.name, "a")) == NULL) {
        fprintf(stderr, "Eih? Cannot write to %s.\n", file.name);
        exit(1);
    }
    // write to file
    int i;
    float totalTime = 0;
    for (i = 0; i < 13; i++) {
        fprintf(file.data, "%f\t", time[i]);
        totalTime += time[i];
    }
    fprintf(file.data, "%f\n", totalTime);
    fclose(file.data);
}

void write_tree_to_file(File file) {
    // open file for writing
    if ((file.data = fopen(file.name, "w")) == NULL) {
        fprintf(stderr, "Eih? Cannot write to %s.\n", file.name);
        exit(1);
    }
    // write to file
    int i;
    for (i = numberOfParticles; i <= maxNodeIndex_host; i++) {
        fprintf(file.data, "%le\t", p_host.x[i]);
#if DIM > 1
        fprintf(file.data, "%le\t", p_host.y[i]);
#if DIM == 3
        fprintf(file.data, "%le\t", p_host.z[i]);
#endif
#endif
        fprintf(file.data, "%le\t", p_host.m[i]);
        fprintf(file.data, "\n");
    }
    fclose(file.data);
}

void write_fragments_file() {
    File file;
    sprintf(file.name, "fragments.input");
    // write to file
    int i;
    for (i = 0; i < numberOfParticles; i++) {
        fprintf(file.data, "%le\t", p_host.x[i]);
#if DIM > 1
        fprintf(file.data, "%le\t", p_host.y[i]);
#if DIM == 3
        fprintf(file.data, "%le\t", p_host.z[i]);
#endif
#endif
        fprintf(file.data, "%le\t", p_host.vx[i]);
#if DIM > 1
        fprintf(file.data, "%le\t", p_host.vy[i]);
#if DIM == 3
        fprintf(file.data, "%le\t", p_host.vz[i]);
#endif
#endif
        fprintf(file.data, "%le\t", p_host.m[i]);
        fprintf(file.data, "%le\t", sml[p_host.materialId[i]]);
        fprintf(file.data, "%le\t", p_host.rho[i]);
        fprintf(file.data, "\n");
    }
    fclose(file.data);
}

void copyToHostAndWriteToFile(int timestep, int lastTimestep)
{
    cudaVerify(cudaDeviceSynchronize());

    int rc;


    if (currentDiskIO) {
        if (param.verbose) fprintf(stderr, "waiting for i/o thread ...\n");
        rc = pthread_join(fileIOthread, NULL);
        assert(0 == rc);
    }

    // calling additional functions to get correct values at timestep
    if (param.verbose) {
        printf("calling pressure for i/o\n");
    }
    cudaVerify(cudaDeviceSynchronize());
	cudaVerifyKernel((calculatePressure<<<numberOfMultiprocessors * 4, NUM_THREADS_PRESSURE>>>()));


    // copy particle data back to host
    if (param.verbose) printf("copying data to host...\n");

#if GRAVITATING_POINT_MASSES
    cudaVerify(cudaMemcpy(pointmass_host.x, pointmass_device.x, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
#if DIM > 1
    cudaVerify(cudaMemcpy(pointmass_host.y, pointmass_device.y, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
#endif
    cudaVerify(cudaMemcpy(pointmass_host.vx, pointmass_device.vx, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
#if DIM > 1
    cudaVerify(cudaMemcpy(pointmass_host.vy, pointmass_device.vy, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
#endif
    cudaVerify(cudaMemcpy(pointmass_host.ax, pointmass_device.ax, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
#if DIM > 1
    cudaVerify(cudaMemcpy(pointmass_host.ay, pointmass_device.ay, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
#endif
#if DIM == 3
    cudaVerify(cudaMemcpy(pointmass_host.z, pointmass_device.z, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(pointmass_host.vz, pointmass_device.vz, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(pointmass_host.az, pointmass_device.az, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
#endif
    cudaVerify(cudaMemcpy(pointmass_host.m, pointmass_device.m, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(pointmass_host.rmin, pointmass_device.rmin, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(pointmass_host.rmax, pointmass_device.rmax, memorySizeForPointmasses, cudaMemcpyDeviceToHost));
#endif // GRAVITATING_POINT_MASSES

    cudaVerify(cudaMemcpy(p_host.x, p_device.x, memorySizeForTree, cudaMemcpyDeviceToHost));
#if DIM > 1
    cudaVerify(cudaMemcpy(p_host.y, p_device.y, memorySizeForTree, cudaMemcpyDeviceToHost));
#endif
    cudaVerify(cudaMemcpy(p_host.vx, p_device.vx, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.vx0, p_device.vx0, memorySizeForParticles, cudaMemcpyDeviceToHost));
#if DIM > 1
    cudaVerify(cudaMemcpy(p_host.vy, p_device.vy, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.vy0, p_device.vy0, memorySizeForParticles, cudaMemcpyDeviceToHost));
#endif
    cudaVerify(cudaMemcpy(p_host.ax, p_device.ax, memorySizeForParticles, cudaMemcpyDeviceToHost));
#if DIM > 1
    cudaVerify(cudaMemcpy(p_host.ay, p_device.ay, memorySizeForParticles, cudaMemcpyDeviceToHost));
#endif
    cudaVerify(cudaMemcpy(p_host.g_ax, p_device.g_ax, memorySizeForParticles, cudaMemcpyDeviceToHost));
#if DIM > 1
    cudaVerify(cudaMemcpy(p_host.g_ay, p_device.g_ay, memorySizeForParticles, cudaMemcpyDeviceToHost));
#endif
#if DIM == 3
    cudaVerify(cudaMemcpy(p_host.z, p_device.z, memorySizeForTree, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.vz, p_device.vz, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.vz0, p_device.vz0, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.az, p_device.az, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.g_az, p_device.g_az, memorySizeForParticles, cudaMemcpyDeviceToHost));
#endif
    cudaVerify(cudaMemcpy(p_host.m, p_device.m, memorySizeForTree, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.depth, p_device.depth, memorySizeForInteractions, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.rho, p_device.rho, memorySizeForParticles, cudaMemcpyDeviceToHost));
#if INTEGRATE_DENSITY
    cudaVerify(cudaMemcpy(p_host.drhodt, p_device.drhodt, memorySizeForParticles, cudaMemcpyDeviceToHost));
#endif
    cudaVerify(cudaMemcpy(p_host.h, p_device.h, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.materialId, p_device.materialId, memorySizeForInteractions, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.p, p_device.p, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.cs, p_device.cs, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.noi, p_device.noi, memorySizeForInteractions, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(interactions_host, interactions, memorySizeForInteractions*MAX_NUM_INTERACTIONS, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(childList_host, (void * )childListd, memorySizeForChildren, cudaMemcpyDeviceToHost));
#if MORE_OUTPUT
    cudaVerify(cudaMemcpy(p_host.p_min, p_device.p_min, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.p_max, p_device.p_max, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.rho_min, p_device.rho_min, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.rho_max, p_device.rho_max, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.e_min, p_device.e_min, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.e_max, p_device.e_max, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.cs_min, p_device.cs_min, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.cs_max, p_device.cs_max, memorySizeForParticles, cudaMemcpyDeviceToHost));
#endif
#if PALPHA_POROSITY
    cudaVerify(cudaMemcpy(p_host.pold, p_device.pold, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.alpha_jutzi, p_device.alpha_jutzi, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.dalphadt, p_device.dalphadt, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.alpha_jutzi_old, p_device.alpha_jutzi_old, memorySizeForParticles, cudaMemcpyDeviceToHost));
#endif
#if SIRONO_POROSITY
    cudaVerify(cudaMemcpy(p_host.compressive_strength, p_device.compressive_strength, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.tensile_strength, p_device.tensile_strength, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.K, p_device.K, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.rho_0prime, p_device.rho_0prime, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.rho_c_plus, p_device.rho_c_plus, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.rho_c_minus, p_device.rho_c_minus, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.shear_strength, p_device.shear_strength, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.flag_rho_0prime, p_device.flag_rho_0prime, memorySizeForInteractions, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.flag_plastic, p_device.flag_plastic, memorySizeForInteractions, cudaMemcpyDeviceToHost));
#endif

#if EPSALPHA_POROSITY
    cudaVerify(cudaMemcpy(p_host.alpha_epspor, p_device.alpha_epspor, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.epsilon_v, p_device.epsilon_v, memorySizeForParticles, cudaMemcpyDeviceToHost));
#endif

#if INTEGRATE_ENERGY
    cudaVerify(cudaMemcpy(p_host.e, p_device.e, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.dedt, p_device.dedt, memorySizeForParticles, cudaMemcpyDeviceToHost));
#endif
#if JC_PLASTICITY
    cudaVerify(cudaMemcpy(p_host.ep, p_device.ep, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.T, p_device.T, memorySizeForParticles, cudaMemcpyDeviceToHost));
#endif
#if NAVIER_STOKES
    cudaVerify(cudaMemcpy(p_host.Tshear, p_device.Tshear, memorySizeForStress, cudaMemcpyDeviceToHost));
#endif
#if SOLID
    cudaVerify(cudaMemcpy(p_host.S, p_device.S, memorySizeForStress, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.dSdt, p_device.dSdt, memorySizeForStress, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.local_strain, p_device.local_strain, memorySizeForParticles, cudaMemcpyDeviceToHost));
#endif
#if FRAGMENTATION
    cudaVerify(cudaMemcpy(p_host.d, p_device.d, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.dddt, p_device.dddt, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.numActiveFlaws, p_device.numActiveFlaws, memorySizeForInteractions, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.flaws, p_device.flaws, memorySizeForActivationThreshold, cudaMemcpyDeviceToHost));
#if PALPHA_POROSITY
    cudaVerify(cudaMemcpy(p_host.damage_porjutzi, p_device.damage_porjutzi, memorySizeForParticles, cudaMemcpyDeviceToHost));
    cudaVerify(cudaMemcpy(p_host.ddamage_porjutzidt, p_device.ddamage_porjutzidt, memorySizeForParticles, cudaMemcpyDeviceToHost));
#endif
#endif

    cudaVerify(cudaDeviceSynchronize());


    // write data to file
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    int *t = (int*)malloc(sizeof(int));
    *t = timestep;
    currentDiskIO = TRUE;
    h5time = currentTime;
    rc = pthread_create(&fileIOthread, &attr, write_timestep, (void*)t);
    pthread_attr_destroy(&attr);
    assert(0 == rc);
}
