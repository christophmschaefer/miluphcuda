/* Tool for identifying fragments (particles connected by up to a smoothing length) in a miluphcuda output file.
 * Both, ASCII and HDF5 miluphcuda output files are supported, but currently only a constant smoothing length,
 * which is read directly from the miluphcuda output file.
 * 
 * All units are SI
 * 
 * last updated: 26/Nov/2020
 * 
 * Christoph Burger
 * christoph.burger@uni-tuebingen.de
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <hdf5.h>

#define TREE_ALLOC_FACT 100.0 // specifies the amount of allocated memory for tree node data, in units of total particle number
#define MAX_TREE_DEPTH 100    // used only for assigning lengths to arrays storing path in tree walk
#define MAX_NUM_INTERACTIONS 250

#define DIM 3
#define PATHLENGTH 256
#define TRUE 1
#define FALSE 0
#define EPS6 1.0e-6

#define MIN(x,y) ((x)<(y) ? (x):(y))
#define ERRORTEXT(x) {fprintf(stderr,x); exit(1);}
#define ERRORVAR(x,y) {fprintf(stderr,x,y); exit(1);}

//#define DEBUG_MODE
//#define FRAG_DEBUG
//#define SORT_DOUBLE_DEBUG
//#define SORT_INT_DEBUG

typedef struct dataOnParticles
{
    double* x;
    double* y;
    double* z;
    double* vx;
    double* vy;
    double* vz;
    double* m;
    double* rho;
    double* e;
    int* mat_types;
    int* treeIndices;
    double* rootNodeLength;
    int* interactionPartner;    // (one-dimensional) list of interaction partner
    int* noip;  // number of interaction partner
} particlesData;

typedef struct dataOnFragment
{
    double x[DIM];
    double v[DIM];
    double m;
    double rel_m;   //fragment mass relative to total mass
    double* mat_frac;
    int n_members; // number of particles in fragment
    int* members; // list of particles in fragment
} fragmentData;

void buildTree(particlesData p, int N, int M, int verbose);
void findInteractionPartner(particlesData p, int N, int M, double sml, int verbose);
void mergeSortDouble(double* array, int* indices, int noets);
void mergeSortInt(int* array, int* indices, int noets);


void usage(char* programname)
{
    fprintf(stdout, "\nTool for identifying fragments (particles connected by up to a smoothing length) in a miluphcuda output file.\n"
                    "Both, ASCII and HDF5 miluphcuda output files are supported, but currently only a constant smoothing length,\n"
                    "which is read directly from the miluphcuda output file.\n");
    fprintf(stdout, "\n  Usage: %s [Options]\n", programname);
    fprintf(stdout, "\n  Options:\n");
    fprintf(stdout, "    -?               display this message and exit\n");
    fprintf(stdout, "    -v               be verbose, default is false\n");
    fprintf(stdout, "    -H               read from a miluphcuda HDF5 output file, otherwise an ASCII output file is assumed\n");
    fprintf(stdout, "                     NOTE: fairly often the limited precision in ASCII output files results in particles with identical coordinates, and the tree building fails.\n");
    fprintf(stdout, "    -i inputfile     specify miluphcuda outputfile to read from\n");
//    fprintf(stdout, "    -s               set this flag if the miluphcuda file to read is from a solid run, otherwise it is assumed to be from a hydro run\n");
    fprintf(stdout, "    -o outputfile    specify file to write to\n");
    fprintf(stdout, "    -m n_materials   specify number of different materials (without the material to ignore - if '-I' is used)\n");
    fprintf(stdout, "    -I ignore-mat    specify single material ID which is simply ignored for fragment search\n");
    fprintf(stdout, "    -l filename      optional; set flag and filename to write some metadata, especially the particleindices in the inputfile (linenumbers)\n");
    fprintf(stdout, "                     of particles in all fragments to file (sorted); this doesn't work if '-I' is used!\n");
    fprintf(stdout, "    -L               legacy-flag to read from old HDF5 files which label the smoothing length as \"hsml\" (instead of \"sml\")\n");
    fprintf(stdout, "\n");
}


int main(int argc, char* argv[])
{
    int i,j,k,l,m,mIndex;
    double a;
    FILE *ifl, *ofl, *lfl;
    hid_t ifl_id, x_id, v_id, m_id, rho_id, e_id, sml_id, mattypes_id, mattypes_pre_id;
    int verbose = FALSE;
    int hdf5Flag = FALSE;
//    int solidFlag = FALSE;
    int ignoreFlag = FALSE;
    int ignoreMat;  // material ID to ignore
    int* mat_types_pre; // array for storing all mattypes - used if some material is to be ignored
    int linesFlag = FALSE;
    int legacyFlag = FALSE;
    char infile[PATHLENGTH];
    char outfile[PATHLENGTH];
    char linesfile[PATHLENGTH];
    int N_pre = 0;  // preliminary particle number ( = final 'N' if no mattype is to be ignored)
    int N = 0;  // final particle number to consider for fragment search
    particlesData p;
    int particleArrayLength = -1;
    double sml = -1.0; // only constant smoothing length
    fragmentData* fragments = NULL; // initialize to avoid problems with realloc
    int* whichFragment; // stores the fragment each particle is in, and -1 otherwise
    int nof;    // number of fragments
    int n_mat = -1;
    int currentDepth = -1;  // current depth during fragment search starting from one particle (which has depth 0)
    int* currentPath; // current path during fragment search in terms of particle indices
    int* currentPartner;  // for storing current interaction partner (from 0 to noip-1) for all levels of the currentPath
    int currentParticle;    // index of currently treated particle
    double M_tot;
    int max_depth;  // max "depth" during "tree walk" when searching for connected fragments
    hsize_t dims[2];    // for storing dimensions of hdf5 (position) dataset
    double* hdf5_buf;  // buffer for reading data (positions and velocities) from hdf5 file
    double *buf1, *buf2, *buf3; // buffer
    herr_t hdf5_status;
    hid_t dataspace_id;
    
    
// process command line options:
    while ( (i = getopt(argc, argv, "?vHi:o:m:I:l:L")) != -1 )
        switch((char)i)
        {
            case '?':
                usage(argv[0]);
                exit(1);
            case 'v':
                verbose = TRUE;
                break;
            case 'H':
                hdf5Flag = TRUE;
                break;
            case 'i':
                strncpy(infile,optarg,PATHLENGTH);
                break;
//            case 's':
//                solidFlag = TRUE;
//                break;
            case 'o':
                strncpy(outfile,optarg,PATHLENGTH);
                break;
            case 'm':
                n_mat = atoi(optarg);
                break;
            case 'I':
                ignoreFlag = TRUE;
                ignoreMat = atoi(optarg);
                if( linesFlag )
                    ERRORTEXT("WARNING - writing metadata (with flag '-l') is not possible while simultaneously ignoring a material (with flag '-I')! Aborting ...\n")
                break;
            case 'l':
                linesFlag = TRUE;
                strncpy(linesfile,optarg,PATHLENGTH);
                if( ignoreFlag )
                    ERRORTEXT("WARNING - writing metadata (with flag '-l') is not possible while simultaneously ignoring a material (with flag '-I')! Aborting ...\n")
                break;
            case 'L':
                legacyFlag = TRUE;
                break;
            default:
                usage(argv[0]);
                exit(1);
        }
    
// open files and check for errors:
    if( hdf5Flag )
        if ( (ifl_id = H5Fopen(infile, H5F_ACC_RDONLY, H5P_DEFAULT)) < 0 )
            ERRORVAR("FILE ERROR! Cannot open %s for reading.\n",infile)
    if( !hdf5Flag )
        if ( (ifl = fopen(infile, "r")) == NULL )
            ERRORVAR("FILE ERROR! Cannot open %s for reading.\n",infile)
    if ( (ofl = fopen(outfile, "w")) == NULL )
        ERRORVAR("FILE ERROR! Cannot open %s for writing.\n",outfile)
    if( linesFlag )
        if ( (lfl = fopen(linesfile, "w")) == NULL )
            ERRORVAR("FILE ERROR! Cannot open %s for writing.\n",linesfile)
    
// read input file:
    if(verbose)
        fprintf(stdout, "\n----------------------------------------------------------------\nReading input file \"%s\" ... ", infile);
    
    if( hdf5Flag )  // determine total particle number 'N_pre' from hdf5 file
    {
        x_id = H5Dopen(ifl_id, "/x", H5P_DEFAULT);  // open positions dataset
        if( x_id < 0 )
            ERRORVAR("ERROR. Cannot find dataset for positions in input-file %s.\n",infile)
        dataspace_id = H5Dget_space(x_id);
        if( dataspace_id < 0 )
            ERRORVAR("ERROR when copying dataspace in file %s!\n", infile)
        H5Sget_simple_extent_dims(dataspace_id, dims, NULL);    // get extent of dataset
        N_pre = dims[0];
    }
    if( !hdf5Flag ) // determine total particle number 'N_pre' for ascii file
    {
        while ( fscanf(ifl, "%le%*[^\n]\n", &a) == 1 )    // just save first value to dummy variable, * means don't save to var, [^\n] is regular expression - "not \n"
            N_pre++;
        rewind(ifl);
    }
    
    // determine particle number to consider 'N' ( = 'N_pre' if no mattype to ignore):
    if( ignoreFlag )
    {
        // read mattypes of all particles in input file:
        if ( (mat_types_pre = (int*)malloc(N_pre*sizeof(int))) == NULL )
            ERRORTEXT("ERROR during memory allocation for 'mat_types_pre'!\n")
        if( hdf5Flag )
        {
            mattypes_pre_id = H5Dopen(ifl_id, "/material_type", H5P_DEFAULT);  // open mattypes dataset
            if( mattypes_pre_id < 0 )
                ERRORVAR("ERROR. Cannot find dataset for mattypes in input-file %s.\n",infile)
            hdf5_status = H5Dread(mattypes_pre_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat_types_pre);
            if( hdf5_status < 0 )
                ERRORVAR("ERROR when reading mattypes data from file %s!\n", infile)
            hdf5_status = H5Dclose(mattypes_pre_id);
        }
        if( !hdf5Flag )
        {
            for(i=0; i<N_pre; i++)
                if ( fscanf(ifl, "%*le %*le %*le %*le %*le %*le %*le %*le %*le %*le %*d %d%*[^\n]\n", &(mat_types_pre[i]) ) != 1 )
                    ERRORVAR("ERROR when reading mattypes from input file %s!\n", infile)
            rewind(ifl);
        }
        
        // determine 'N':
        N = 0;
        for(i=0; i<N_pre; i++)
            if( mat_types_pre[i] != ignoreMat )
                N++;
    }
    else
        N = N_pre;
    
    // allocate memory:
    max_depth = N;  // has turned out to be a good choice ...
    if ( (currentPath = (int*)malloc(max_depth*sizeof(int))) == NULL )
        ERRORTEXT("ERROR during memory allocation for fragment data!\n")
    if ( (currentPartner = (int*)malloc(max_depth*sizeof(int))) == NULL )
        ERRORTEXT("ERROR during memory allocation for fragment data!\n")
    particleArrayLength = N + (int)ceil(TREE_ALLOC_FACT*N);
    if ( (p.x = (double*)malloc(particleArrayLength*sizeof(double))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (p.y = (double*)malloc(particleArrayLength*sizeof(double))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (p.z = (double*)malloc(particleArrayLength*sizeof(double))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (p.vx = (double*)malloc(N*sizeof(double))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (p.vy = (double*)malloc(N*sizeof(double))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (p.vz = (double*)malloc(N*sizeof(double))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (p.m = (double*)malloc(particleArrayLength*sizeof(double))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (p.rho = (double*)malloc(N*sizeof(double))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (p.e = (double*)malloc(N*sizeof(double))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (p.mat_types = (int*)malloc(N*sizeof(int))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (p.treeIndices = (int*)malloc(particleArrayLength*sizeof(int))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (p.rootNodeLength = (double*)malloc(sizeof(double))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (p.interactionPartner = (int*)malloc(N*MAX_NUM_INTERACTIONS*sizeof(int))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (p.noip = (int*)malloc(N*sizeof(int))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    if ( (whichFragment = (int*)malloc(N*sizeof(int))) == NULL )
        ERRORTEXT("ERROR during memory allocation for particle data!\n")
    for(i=0; i<N; i++)
        whichFragment[i] = -1;
    
    //read all data from infile:
    if( hdf5Flag )
    {
        if( ignoreFlag )
        {
            if( (buf1 = (double*)malloc(N_pre*sizeof(double))) == NULL )
                ERRORTEXT("ERROR during memory allocation for buffer!\n")
            if( (buf2 = (double*)malloc(N_pre*sizeof(double))) == NULL )
                ERRORTEXT("ERROR during memory allocation for buffer!\n")
            if( (buf3 = (double*)malloc(N_pre*sizeof(double))) == NULL )
                ERRORTEXT("ERROR during memory allocation for buffer!\n")
        }
        if( (hdf5_buf = (double*)malloc(N_pre*DIM*sizeof(double))) == NULL )
            ERRORTEXT("ERROR during memory allocation for hdf5 buffer!\n")
        
        // read positions dataset and close it:
        hdf5_status = H5Dread(x_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, hdf5_buf);
        if( hdf5_status < 0 )
            ERRORVAR("ERROR when reading position data from file %s!\n", infile)
        hdf5_status = H5Dclose(x_id);
        if( ignoreFlag )
        {
            for(i=0, j=0; i<N_pre; i++, j+=DIM)
            {
                buf1[i] = hdf5_buf[j];
                buf2[i] = hdf5_buf[j+1];
                buf3[i] = hdf5_buf[j+2];
            }
            j=0;
            for(i=0; i<N_pre; i++)
                if( mat_types_pre[i] != ignoreMat )
                {
                    p.x[j] = buf1[i];
                    p.y[j] = buf2[i];
                    p.z[j] = buf3[i];
                    j++;
                }
            if( j != N )
                ERRORTEXT("ERROR! Mismatch in number of read particles in mat-type-ignore-mode!\n")
        }
        else
            for(i=0, j=0; i<N; i++, j+=DIM)
            {
                p.x[i] = hdf5_buf[j];
                p.y[i] = hdf5_buf[j+1];
                p.z[i] = hdf5_buf[j+2];
            }
        
        // read velocities:
        v_id = H5Dopen(ifl_id, "/v", H5P_DEFAULT);  // open velocities dataset
        if( v_id < 0 )
            ERRORVAR("ERROR. Cannot find dataset for velocities in input-file %s.\n",infile)
        hdf5_status = H5Dread(v_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, hdf5_buf);
        if( hdf5_status < 0 )
            ERRORVAR("ERROR when reading velocities data from file %s!\n", infile)
        hdf5_status = H5Dclose(v_id);
        if( ignoreFlag )
        {
            for(i=0, j=0; i<N_pre; i++, j+=DIM)
            {
                buf1[i] = hdf5_buf[j];
                buf2[i] = hdf5_buf[j+1];
                buf3[i] = hdf5_buf[j+2];
            }
            j=0;
            for(i=0; i<N_pre; i++)
                if( mat_types_pre[i] != ignoreMat )
                {
                    p.vx[j] = buf1[i];
                    p.vy[j] = buf2[i];
                    p.vz[j] = buf3[i];
                    j++;
                }
            if( j != N )
                ERRORTEXT("ERROR! Mismatch in number of read particles in mat-type-ignore-mode!\n")
        }
        else
            for(i=0, j=0; i<N; i++, j+=DIM)
            {
                p.vx[i] = hdf5_buf[j];
                p.vy[i] = hdf5_buf[j+1];
                p.vz[i] = hdf5_buf[j+2];
            }
        free(hdf5_buf);
        
        // read masses:
        m_id = H5Dopen(ifl_id, "/m", H5P_DEFAULT);  // open masses dataset
        if( m_id < 0 )
            ERRORVAR("ERROR. Cannot find dataset for masses in input-file %s.\n",infile)
        if( ignoreFlag )
        {
            hdf5_status = H5Dread(m_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf1);
            j=0;
            for(i=0; i<N_pre; i++)
                if( mat_types_pre[i] != ignoreMat )
                {
                    p.m[j] = buf1[i];
                    j++;
                }
            if( j != N )
                ERRORTEXT("ERROR! Mismatch in number of read particles in mat-type-ignore-mode!\n")
        }
        else
            hdf5_status = H5Dread(m_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, p.m);
        if( hdf5_status < 0 )
            ERRORVAR("ERROR when reading masses data from file %s!\n", infile)
        hdf5_status = H5Dclose(m_id);
        
        // read densities:
        rho_id = H5Dopen(ifl_id, "/rho", H5P_DEFAULT);  // open densities dataset
        if( rho_id < 0 )
            ERRORVAR("ERROR. Cannot find dataset for densities in input-file %s.\n",infile)
        if( ignoreFlag )
        {
            hdf5_status = H5Dread(rho_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf1);
            j=0;
            for(i=0; i<N_pre; i++)
                if( mat_types_pre[i] != ignoreMat )
                {
                    p.rho[j] = buf1[i];
                    j++;
                }
            if( j != N )
                ERRORTEXT("ERROR! Mismatch in number of read particles in mat-type-ignore-mode!\n")
        }
        else
            hdf5_status = H5Dread(rho_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, p.rho);
        if( hdf5_status < 0 )
            ERRORVAR("ERROR when reading densities data from file %s!\n", infile)
        hdf5_status = H5Dclose(rho_id);
        
        // read energies:
        e_id = H5Dopen(ifl_id, "/e", H5P_DEFAULT);  // open energies dataset
        if( e_id < 0 )
            ERRORVAR("ERROR. Cannot find dataset for energies in input-file %s.\n",infile)
        if( ignoreFlag )
        {
            hdf5_status = H5Dread(e_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf1);
            j=0;
            for(i=0; i<N_pre; i++)
                if( mat_types_pre[i] != ignoreMat )
                {
                    p.e[j] = buf1[i];
                    j++;
                }
            if( j != N )
                ERRORTEXT("ERROR! Mismatch in number of read particles in mat-type-ignore-mode!\n")
        }
        else
            hdf5_status = H5Dread(e_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, p.e);
        if( hdf5_status < 0 )
            ERRORVAR("ERROR when reading energies data from file %s!", infile)
        hdf5_status = H5Dclose(e_id);
        
        // read sml:
        if( (hdf5_buf = (double*)malloc(N_pre*sizeof(double))) == NULL )
            ERRORTEXT("ERROR during memory allocation for hdf5 buffer!\n")
        if(legacyFlag)
            sml_id = H5Dopen(ifl_id, "/hsml", H5P_DEFAULT);  // open hsml dataset
        else
            sml_id = H5Dopen(ifl_id, "/sml", H5P_DEFAULT);  // open sml dataset
        if( sml_id < 0 )
            ERRORVAR("ERROR. Cannot find dataset for sml in input-file %s.\n",infile)
        hdf5_status = H5Dread(sml_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, hdf5_buf);
        if( hdf5_status < 0 )
            ERRORVAR("ERROR when reading sml data from file %s!\n", infile)
        hdf5_status = H5Dclose(sml_id);
        if( ignoreFlag )    // use the sml of the first particle not from the mattype to ignore
        {
            for(i=0; i<N_pre; i++)
                if( mat_types_pre[i] != ignoreMat )
                {
                    sml = hdf5_buf[i];
                    break;
                }
        }
        else
            sml = hdf5_buf[0];
        free(hdf5_buf);
        
        // read mattypes:
        if( ignoreFlag )
        {
            j=0;
            for(i=0; i<N_pre; i++)
                if( mat_types_pre[i] != ignoreMat )
                {
                    p.mat_types[j] = mat_types_pre[i];
                    j++;
                }
            if( j != N )
                ERRORTEXT("ERROR! Mismatch in number of read particles in mat-type-ignore-mode!\n")
        }
        else
        {
            mattypes_id = H5Dopen(ifl_id, "/material_type", H5P_DEFAULT);  // open mattypes dataset
            if( mattypes_id < 0 )
                ERRORVAR("ERROR. Cannot find dataset for mattypes in input-file %s.\n",infile)
            hdf5_status = H5Dread(mattypes_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, p.mat_types);
            if( hdf5_status < 0 )
                ERRORVAR("ERROR when reading mattypes data from file %s!\n", infile)
            hdf5_status = H5Dclose(mattypes_id);
        }
        
        H5Fclose(ifl_id);
        
        if(ignoreFlag )
        {
            free(buf1);
            free(buf2);
            free(buf3);
        }
    }   //end if( hdf5Flag )
    
    if( !hdf5Flag )
    {
        if( ignoreFlag )
        {
            j=0;
            for(i=0; i<N_pre; i++)
            {
                if( mat_types_pre[i] != ignoreMat ) // read line
                {
                    if ( fscanf(ifl, "%le %le %le %le %le %le %le %le %le %le %*d %d%*[^\n]\n", &(p.x[j]), &(p.y[j]), &(p.z[j]), &(p.vx[j]), &(p.vy[j]), &(p.vz[j]), &(p.m[j]), &(p.rho[j]), &(p.e[j]), &sml, &(p.mat_types[j]) ) != 11 )    //* means don't save to var, [^\n] is regular expression - "not \n"
                        ERRORVAR("ERROR when reading data from input file %s!\n", infile)
                    j++;
                }
                else    // skip line
                    fscanf(ifl, "%*[^\n]\n");
            }
            if( j != N )
                ERRORTEXT("ERROR! Mismatch in number of read particles in mat-type-ignore-mode!\n")
        }
        else
        {
            for(i=0; i<N; i++)
                if ( fscanf(ifl, "%le %le %le %le %le %le %le %le %le %le %*d %d%*[^\n]\n", &(p.x[i]), &(p.y[i]), &(p.z[i]), &(p.vx[i]), &(p.vy[i]), &(p.vz[i]), &(p.m[i]), &(p.rho[i]), &(p.e[i]), &sml, &(p.mat_types[i]) ) != 11 )    //* means don't save to var, [^\n] is regular expression - "not \n"
                    ERRORVAR("ERROR when reading data from input file %s!\n", infile)
        }
        fclose(ifl);
    }
    
    if(verbose)
    {
        if( ignoreFlag )
            fprintf(stdout, "Done.\nFound %d particles alltogether in file, but read only %d and ignored %d (because they have mattype %d).\n", N_pre, N, N_pre-N, ignoreMat);
        else
            fprintf(stdout, "Done.\nFound %d particles.\n", N);
    }
    
// build the tree and find the interaction partner:
    buildTree(p, N, particleArrayLength, verbose);
    findInteractionPartner(p, N, particleArrayLength, sml, verbose);
    
// find fragments:
    if(verbose)
        fprintf(stdout, "Start identifying fragments ... ");
    nof = 0;
    for(i=0; i<N; i++)  // loop over all particles
    {
#ifdef FRAG_DEBUG
        fprintf(stdout, "processing particle %d ... ", i);
#endif
        if( whichFragment[i] == -1 )    // particle i not yet part of any fragment, i.e. it is certainly part of a new fragment and not part of any existing one
        {
#ifdef FRAG_DEBUG
            fprintf(stdout, "which is not part of any fragment yet, therefore build fragment (with no. %d) starting with it ... ", nof);
#endif
            nof++;
            if ( (fragments = realloc( fragments, nof*sizeof(fragmentData) )) == NULL ) // expand list of fragments
                ERRORTEXT("ERROR during memory allocation for fragments!\n")
            if ( (fragments[nof-1].mat_frac = (double*)malloc(n_mat*sizeof(double))) == NULL )  // allocate memory for mat fractions of the new fragment
                ERRORTEXT("ERROR during memory allocation for fragments data!\n")
            // write particle i to member list of new fragment:
            if ( (fragments[nof-1].members = (int*)malloc(sizeof(int))) == NULL )
                ERRORTEXT("ERROR during memory allocation for fragments data!\n")
            fragments[nof-1].members[0] = i;
            fragments[nof-1].n_members = 1;
            whichFragment[i] = nof-1;
#ifdef FRAG_DEBUG
            if( p.noip[i] == 0 )
                fprintf(stdout, "but particle has no interaction partner ...\n");
#endif
            if( p.noip[i] != 0 )    // particle has at least one interaction partner
            {
#ifdef FRAG_DEBUG
                fprintf(stdout, "and processing its %d interaction partner ...\n", p.noip[i]);
#endif
                // start with all interaction partner of particle i:
                currentPath[0] = i;
                currentDepth = 1;
                currentPartner[1] = 0;
                currentPath[1] = p.interactionPartner[i*MAX_NUM_INTERACTIONS];
#ifdef FRAG_DEBUG
                fprintf(stdout, "    starting at depth %d with particle no. %d ...\n", currentDepth, currentPath[1]);
#endif
                while(TRUE)
                {
                    while( currentPartner[currentDepth] < p.noip[currentPath[currentDepth-1]] ) // loop over all interaction partner
                    {
                        currentParticle = p.interactionPartner[currentPath[currentDepth-1]*MAX_NUM_INTERACTIONS+currentPartner[currentDepth]];    // index of currently treated particle
#ifdef FRAG_DEBUG
                        fprintf(stdout, "    processing interaction partner no. %d/%d, which is particle %d ... ", currentPartner[currentDepth]+1, p.noip[currentPath[currentDepth-1]], currentParticle);
#endif
                        if( whichFragment[currentParticle] == -1 )   // currentParticle not yet part of any fragment -> will be added to current one
                        {
                            fragments[nof-1].n_members++;
#ifdef FRAG_DEBUG
                            fprintf(stdout, "not part of any fragment so far, thus included in this one (%d), increasing n_members to %d ... ", nof-1, fragments[nof-1].n_members);
#endif
                            if ( (fragments[nof-1].members = realloc( fragments[nof-1].members, (fragments[nof-1].n_members)*sizeof(int) )) == NULL )
                                ERRORTEXT("ERROR during memory allocation for fragments!\n")
                            fragments[nof-1].members[(fragments[nof-1].n_members)-1] = currentParticle;
                            whichFragment[currentParticle] = nof-1;
                            // go one level down to interaction partner of currentParticle and start with first one (there should be at least one, the one particle we are coming from):
                            currentDepth++;

                            if( currentDepth > max_depth )
                                ERRORVAR("ERROR! Current depth has become larger than max_depth = %d!\n",max_depth)
                            currentPartner[currentDepth] = 0;
                            currentPath[currentDepth] = p.interactionPartner[currentPath[currentDepth-1]*MAX_NUM_INTERACTIONS];
#ifdef FRAG_DEBUG
                            fprintf(stdout, "and descend one level down (to depth %d) to its interaction partner, and continue with its first one, particle no. %d ...\n", currentDepth, currentPath[currentDepth]);
#endif
                        }
                        else if( whichFragment[currentParticle] == (nof-1) )    // currentParticle is already part of current fragment
                        {
                            currentPartner[currentDepth]++;
                            if( currentPartner[currentDepth] < p.noip[currentPath[currentDepth-1]] )
                                currentPath[currentDepth] = p.interactionPartner[currentPath[currentDepth-1]*MAX_NUM_INTERACTIONS+currentPartner[currentDepth]];
#ifdef FRAG_DEBUG
                            fprintf(stdout, "which is already part of the current fragment ...\n");
#endif
                        }
                        else    // currentParticle not part of current fragment (but of other one) -> error
                            ERRORVAR("ERROR! Particle is already part of other fragment with number %d.\n",whichFragment[currentParticle])
                    }   // end loop over all interaction partner
                    currentDepth--;
                    if( currentDepth <= 0 )
                        break;
                    currentPartner[currentDepth]++;
                    if( currentPartner[currentDepth] < p.noip[currentPath[currentDepth-1]] )
                        currentPath[currentDepth] = p.interactionPartner[currentPath[currentDepth-1]*MAX_NUM_INTERACTIONS+currentPartner[currentDepth]];
#ifdef FRAG_DEBUG
                    if( currentPartner[currentDepth] < p.noip[currentPath[currentDepth-1]] )
                        fprintf(stdout, "    ascend one level (to depth %d) and continue there with interaction partner no. %d (particle no. %d) ...\n", currentDepth, currentPartner[currentDepth], currentPath[currentDepth]);
                    else
                        fprintf(stdout, "    ascend one level (to depth %d) ... and see that we have already covered all interaction partner there ...\n", currentDepth);
#endif
                }   // end infinite loop
            }   // end if( p.noip[i] != 0 )
        }   // end if( whichFragment[i] == -1 )
#ifdef FRAG_DEBUG
        else    // particle i already part of some fragment
            fprintf(stdout, "which is already part of fragment no. %d ...\n", whichFragment[i]);
#endif
    }   // end loop over all particles
    if(verbose)
        fprintf(stdout, "Done.\nFound %d fragments.\n", nof);
    
// compute all fragment properties:
    M_tot = 0.0;
    for(i=0; i<nof; i++)
    {
        // compute fragment mass:
        fragments[i].m = 0.0;
        for(j=0; j<fragments[i].n_members; j++)
            fragments[i].m += p.m[fragments[i].members[j]];
        M_tot += fragments[i].m;
        for(j=0; j<DIM; j++)
            fragments[i].x[j] = fragments[i].v[j] = 0.0;
        for(j=0; j<fragments[i].n_members; j++)
        {
            k = fragments[i].members[j];
            fragments[i].x[0] += p.m[k] * p.x[k];
            fragments[i].x[1] += p.m[k] * p.y[k];
            fragments[i].x[2] += p.m[k] * p.z[k];
            fragments[i].v[0] += p.m[k] * p.vx[k];
            fragments[i].v[1] += p.m[k] * p.vy[k];
            fragments[i].v[2] += p.m[k] * p.vz[k];
        }
        for(j=0; j<DIM; j++)
        {
            fragments[i].x[j] /= fragments[i].m;
            fragments[i].v[j] /= fragments[i].m;
        }
        for(j=0; j<n_mat; j++)
            fragments[i].mat_frac[j] = 0.0;
        for(j=0; j<fragments[i].n_members; j++)
        {
            k = fragments[i].members[j];
            fragments[i].mat_frac[p.mat_types[k]] += p.m[k];
        }
        for(j=0; j<n_mat; j++)
            fragments[i].mat_frac[j] /= fragments[i].m;
    }
    for(i=0; i<nof; i++)
        fragments[i].rel_m = fragments[i].m/M_tot;

// write all fragments to outfile, sorted by mass, and write linesfile if desired:
    if(verbose)
    {
        fprintf(stdout, "Sort fragments by mass and write them to outfile ... ");
        fflush(stdout);
    }
    fprintf(ofl, "#   x1                        x2                        x3                        v1                        v2                        v3                        mass                      rel_mass ");
    for(i=0; i<n_mat; i++)
        fprintf(ofl, "                 mat%d_frac", i);
    fprintf(ofl, "\n");
    
    if( linesFlag )
        fprintf(lfl, "# Number of particles: %d\n# Number of found fragments: %d\n#\n#  1: fragment no. (sorted by mass)    2: no. particles    3: particle indices ...\n#\n", N, nof);
    
    // sort fragments by mass (in increasing order):
    double* sortMasses; // used as intermediate storage of all masses (to pass them to the sort function)
    int* sortIndices;   // used for returning the sorted indices
    fragmentData* sortedFragments;
    
    if ( (sortMasses = (double*)malloc(nof*sizeof(double))) == NULL )
        ERRORTEXT("ERROR during memory allocation for data for sorting fragments!\n")
    if ( (sortIndices = (int*)malloc(nof*sizeof(int))) == NULL )
        ERRORTEXT("ERROR during memory allocation for data for sorting fragments!\n")
    if ( (sortedFragments = malloc(nof*sizeof(fragmentData))) == NULL )
        ERRORTEXT("ERROR during memory allocation for data on sorted fragments!\n")
    
    for(i=0; i<nof; i++)
        sortMasses[i] = fragments[i].m;
    mergeSortDouble(sortMasses, sortIndices, nof);
    
    // build new fragments array 'sortedFragments' containing all fragment data sorted by mass (increasing):
    for(i=0; i<nof; i++)
    {
        j = sortIndices[nof-1-i];   // index in 'fragments' whose data shall be copied to 'sortedFragments[i]'
        
        if ( (sortedFragments[i].mat_frac = (double*)malloc(n_mat*sizeof(double))) == NULL )
            ERRORTEXT("ERROR during memory allocation for data on sorted fragments!\n")
        if ( (sortedFragments[i].members = (int*)malloc(fragments[j].n_members*sizeof(int))) == NULL )
            ERRORTEXT("ERROR during memory allocation for data on sorted fragments!\n")
        
        memcpy(sortedFragments[i].x, fragments[j].x, DIM*sizeof(double));
        memcpy(sortedFragments[i].v, fragments[j].v, DIM*sizeof(double));
        sortedFragments[i].m = fragments[j].m;
        sortedFragments[i].rel_m = fragments[j].rel_m;
        memcpy(sortedFragments[i].mat_frac, fragments[j].mat_frac, n_mat*sizeof(double));
        sortedFragments[i].n_members = fragments[j].n_members;
        memcpy(sortedFragments[i].members, fragments[j].members, fragments[j].n_members*sizeof(int));
    }
    
    // write fragments to outfile:
    for(k=0; k<nof; k++)
    {
        fprintf(ofl, "%26.16le%26.16le%26.16le%26.16le%26.16le%26.16le%26.16le%26.16le", sortedFragments[k].x[0], sortedFragments[k].x[1], sortedFragments[k].x[2],
                sortedFragments[k].v[0], sortedFragments[k].v[1], sortedFragments[k].v[2], sortedFragments[k].m, sortedFragments[k].rel_m);
        for(j=0; j<n_mat; j++)
            fprintf(ofl, "%26.16le", sortedFragments[k].mat_frac[j]);
        fprintf(ofl, "\n");
    }
    
    // write linesfile if desired:
    int* sortLineIndices; // used for returning the sorted indices
    int* tempMembers; // used for intermediate storage of sorted members
    if( linesFlag )
    {
        if ( (sortLineIndices = (int*)malloc(N*sizeof(int))) == NULL )
            ERRORTEXT("ERROR during memory allocation for data on sorted fragment-members!\n")
        if ( (tempMembers = (int*)malloc(N*sizeof(int))) == NULL )
            ERRORTEXT("ERROR during memory allocation for data on sorted fragment-members!\n")
        
        for(k=0; k<nof; k++)
        {
#ifdef SORT_INT_DEBUG
            fprintf(stdout, "sorting members of fragment k = %d ...\n", k);
#endif
            fprintf(lfl, "%d %d", k, sortedFragments[k].n_members);
            mergeSortInt(sortedFragments[k].members, sortLineIndices, sortedFragments[k].n_members);
            
            // store the list of sorted members for 'sortedFragments[k]' in 'tempMembers':
            for(i=0; i<sortedFragments[k].n_members; i++)
                tempMembers[i] = sortedFragments[k].members[sortLineIndices[i]];
            
            // copy the sorted list of members from 'tempMembers' to 'sortedFragments[k].members':
            for(i=0; i<sortedFragments[k].n_members; i++)
                sortedFragments[k].members[i] = tempMembers[i];
            
            // write sorted list of members to file:
            for(i=0; i<sortedFragments[k].n_members; i++)
                fprintf(lfl, " %d", sortedFragments[k].members[i]);
            fprintf(lfl, "\n");
        }
    }
    
    fclose(ofl);
    if( linesFlag )
        fclose(lfl);
    if(verbose)
        fprintf(stdout, "Done.\n");
    
// free memory and clean up:
    free(currentPath);
    free(currentPartner);
    free(p.x);
    free(p.y);
    free(p.z);
    free(p.vx);
    free(p.vy);
    free(p.vz);
    free(p.m);
    free(p.rho);
    free(p.e);
    free(p.mat_types);
    if( ignoreFlag )
        free(mat_types_pre);
    free(p.treeIndices);
    free(p.rootNodeLength);
    free(p.interactionPartner);
    free(p.noip);
    free(whichFragment);
    for(i=0; i<nof; i++)
    {
        free(fragments[i].members);
        free(fragments[i].mat_frac);
        free(sortedFragments[i].members);
        free(sortedFragments[i].mat_frac);
    }
    free(fragments);
    free(sortedFragments);
    free(sortMasses);
    free(sortIndices);
    if( linesFlag )
    {
        free(sortLineIndices);
        free(tempMembers);
    }
    return(0);
}


void buildTree(particlesData p, int N, int M, int verbose)
// Builds the Barnes-Hut tree, following "Burtcher (2011) - An efficient CUDA implementation of the tree-based Barnes-Hut n-body algorithm".
// N is the overall particle number, M is the length of arrays including particle data as well as tree data.
{
    int i;
    double limits[DIM][2];
    double L;   // length of the root node
    double currentNodeLength;   // "current" always refers to the new particle to be sorted into the tree
    double x,y,z;
    int currentTreeDepth;   // "current" always refers to the new particle to be sorted into the tree
    int maxTreeDepth = 0;
    int relChildIndex;  // between 0 and 7
    int currentNodeIndex;   // "current" always refers to the new particle to be sorted into the tree
    int prevNodeIndex;
    int minNodeIndex;   // lowest index in the indexarray occupied by a node (the nodes are filled "from the right")
    int residentialParticleIndex;   // "residential" always refers to a particle already present in a node where the new particle should be inserted
    int residentialCurrentNodeIndex;
    
// Define root node dimensions and position:
    for(i=0; i<DIM; i++)
    {
        limits[i][0] = 1.0e30;
        limits[i][1] = -1.0e30;
    }
    for(i=0; i<N; i++)  // find the limits in x,y, and z direction
    {
            if( p.x[i] < limits[0][0] )
                limits[0][0] = p.x[i];
            if( p.x[i] > limits[0][1] )
                limits[0][1] = p.x[i];
            if( p.y[i] < limits[1][0] )
                limits[1][0] = p.y[i];
            if( p.y[i] > limits[1][1] )
                limits[1][1] = p.y[i];
            if( p.z[i] < limits[2][0] )
                limits[2][0] = p.z[i];
            if( p.z[i] > limits[2][1] )
                limits[2][1] = p.z[i];
    }
    if(verbose)
        fprintf(stdout, "Computational domain: [%g,%g] x [%g,%g] x [%g,%g]\n", limits[0][0], limits[0][1], limits[1][0], limits[1][1], limits[2][0], limits[2][1] );
    p.x[M-1] = (limits[0][0]+limits[0][1])/2.0; //set root node x-position
    p.y[M-1] = (limits[1][0]+limits[1][1])/2.0;
    p.z[M-1] = (limits[2][0]+limits[2][1])/2.0;
    *(p.rootNodeLength) = L = (1.0+EPS6)*fmax( fmax( limits[0][1]-limits[0][0], limits[1][1]-limits[1][0] ), limits[2][1]-limits[2][0] );  // include some safety margin
    if(verbose)
        fprintf(stdout, "Root node:            [%g,%g] x [%g,%g] x [%g,%g]\n", p.x[M-1]-L/2.0, p.x[M-1]+L/2.0, p.y[M-1]-L/2.0, p.y[M-1]+L/2.0, p.z[M-1]-L/2.0, p.z[M-1]+L/2.0 );
    
// Build tree:
    p.treeIndices[M-1] = 0;  // the first particle is assigned to the root node ...
    p.treeIndices[0] = M-1; // ... and the root node to the first particle
    for(i=N; i<M-1; i++)
        p.treeIndices[i] = -1;  // all other node fields are initialized with -1, meaning "no particle and no child"
    minNodeIndex = M-1;
    
    for(i=1; i<N; i++)  // loop over all particles starting with the second one
    {
#ifdef DEBUG_MODE
        fprintf(stdout, "Processing particle with index %d into tree:\n", i);
#endif
        x = p.x[i]; //cache particle coordinates
        y = p.y[i];
        z = p.z[i];
        currentNodeIndex = M-1; // start at root node
        currentNodeLength = L;  // assign root node length
        currentTreeDepth = 0;
        
        while(TRUE) // loop until current particle is inserted in a (perhaps temporary) leaf of the tree
        {
            if( p.treeIndices[currentNodeIndex] == -1 ) // node is currently empty and can be filled with particle i
            {
                p.treeIndices[currentNodeIndex] = i;    // node is filled with particle i
                p.treeIndices[i] = currentNodeIndex;    // a "pointer" to this newly filled node is assigned to particle i
                p.x[currentNodeIndex] = p.x[prevNodeIndex] - currentNodeLength*0.5 + (relChildIndex & 1)*currentNodeLength; //compute and store position of this newly filled node
                p.y[currentNodeIndex] = p.y[prevNodeIndex] - currentNodeLength*0.5 + ((relChildIndex >> 1) & 1)*currentNodeLength;
                p.z[currentNodeIndex] = p.z[prevNodeIndex] - currentNodeLength*0.5 + ((relChildIndex >> 2) & 1)*currentNodeLength;
#ifdef DEBUG_MODE
                fprintf(stdout, "    Node (with index %d) is currently empty and is filled with particle (with index %d) ...\n", currentNodeIndex, i);
#endif
                break;
            }
            if( p.treeIndices[currentNodeIndex] >= N )  // node already has children, thus we descend one level into them
            {               
                // find child node to descend to:
                relChildIndex = 0;
                if( x > p.x[currentNodeIndex] )   relChildIndex = 1;
                if( y > p.y[currentNodeIndex] )   relChildIndex += 2;
                if( z > p.z[currentNodeIndex] )   relChildIndex += 4;
                prevNodeIndex = currentNodeIndex;   // intermediate storage for later access of that node's coordinates
                currentNodeIndex = p.treeIndices[currentNodeIndex] - relChildIndex;
                currentTreeDepth++;
                currentNodeLength = L/pow(2.0,currentTreeDepth);
                //currentNodeLength *= 0.5;
#ifdef DEBUG_MODE
                fprintf(stdout, "    Node (with index %d) already has children, thus moving one level down (to node with index %d, and tree-depth %d)  ...\n", prevNodeIndex, currentNodeIndex, currentTreeDepth);
#endif 
                continue;
            }
            // coming until here means that the current node already contains a particle, therefore we create a child and move (only!) this 
            // particle (the "residential") there. The currently treated particle is then further processed in the next iteration of the loop ...
            if( minNodeIndex-8 >= N )
            {
                residentialParticleIndex = p.treeIndices[currentNodeIndex]; // intermediate storage of the "residential" particle's index
                p.treeIndices[currentNodeIndex] = minNodeIndex-1;   // create new node at first free position
                minNodeIndex -= 8;
                
                // Now move the "residential" particle down to the new node:
                relChildIndex = 0;
                if( p.x[residentialParticleIndex] > p.x[currentNodeIndex] )   relChildIndex = 1;
                if( p.y[residentialParticleIndex] > p.y[currentNodeIndex] )   relChildIndex += 2;
                if( p.z[residentialParticleIndex] > p.z[currentNodeIndex] )   relChildIndex += 4;
                residentialCurrentNodeIndex = p.treeIndices[currentNodeIndex] - relChildIndex;
                p.treeIndices[residentialCurrentNodeIndex] = residentialParticleIndex;  // assign "residential" particle index to respective node
                p.treeIndices[residentialParticleIndex] = residentialCurrentNodeIndex;  // assign respective node index to "residential" particle
                p.x[residentialCurrentNodeIndex] = p.x[currentNodeIndex] - currentNodeLength*0.25 + (relChildIndex & 1)*currentNodeLength*0.5; //compute and store position of this newly filled node
                p.y[residentialCurrentNodeIndex] = p.y[currentNodeIndex] - currentNodeLength*0.25 + ((relChildIndex >> 1) & 1)*currentNodeLength*0.5;
                p.z[residentialCurrentNodeIndex] = p.z[currentNodeIndex] - currentNodeLength*0.25 + ((relChildIndex >> 2) & 1)*currentNodeLength*0.5;
#ifdef DEBUG_MODE
                fprintf(stdout, "    Current node (index %d) already contains particle (with index %d) - create children (from index %d downwards) and move residential particle one level down (to node with index %d) ...\n", currentNodeIndex, residentialParticleIndex, p.treeIndices[currentNodeIndex], residentialCurrentNodeIndex);
#endif
            }
            else
                ERRORTEXT("ERROR! Not enough nodes for building the tree! Probable cause is either too little allocated memory, or that 2 particles have practically identical coordinates (possible especially with too low floating point precision, as in ascii output-files!)\n");
        }
        if( currentTreeDepth > maxTreeDepth )
            maxTreeDepth = currentTreeDepth;
        
    }   // end loop over all particles
    if( maxTreeDepth > MAX_TREE_DEPTH )
        ERRORVAR("ERROR! Tree-depth (%d) is larger than maximum value allowed!\n", maxTreeDepth)
    if(verbose)
        fprintf(stdout, "Finished building tree with max. depth of %d.\n", maxTreeDepth);
#ifdef DEBUG_MODE
    for(i=0; i<M; i++)
        fprintf(stdout, "%d\t%d\n", i, p.treeIndices[i]);
#endif
}   // end buildTree



void findInteractionPartner(particlesData p, int N, int M, double sml, int verbose)
// Finds interaction partner for all particles by traversing the (already set up) tree, and stores them in 1D array "interactionPartner".
{
    int i;
    double x,y,z;
    int currentNodeIndex;
    double currentNodeLength;
    int currentTreeDepth;   // corresponds also to index in currentTreePath and relNodeIndices arrays
    int currentTreePath[MAX_TREE_DEPTH];    // stores the path taken to the current node (in terms of node indices), starting at the root node
    int relNodeIndices[MAX_TREE_DEPTH];    // stores the relative node indices (from "right to left", between 0 and 7) for the current path
    double limitingDist;
    double sqrDist; // squared distance between two particles
    int numInteractionPartner;
    
    currentTreePath[0] = M-1;
    
    for(i=0; i<N*MAX_NUM_INTERACTIONS; i++)
        p.interactionPartner[i] = -1;   // set all fields in interactionPartner array to -1
    
    for(i=0; i<N; i++)  // loop over all particles
    {
#ifdef DEBUG_MODE
        fprintf(stdout, "Search for interaction partner of particle with index %d:\n", i);
#endif
        x = p.x[i]; //cache particle coordinates and smoothing length
        y = p.y[i];
        z = p.z[i];
        relNodeIndices[1] = 0; // start at node 0 of root's children
        currentNodeLength = *(p.rootNodeLength)/2.0;
        currentTreeDepth = 1;
        numInteractionPartner = 0;
        
        do
        {
            while( relNodeIndices[currentTreeDepth] < 8 )  // loops over a node's children (from "right to left" in the treeIndices array)
            {
                currentTreePath[currentTreeDepth] = currentNodeIndex = p.treeIndices[currentTreePath[currentTreeDepth-1]] - relNodeIndices[currentTreeDepth];
                limitingDist = currentNodeLength*0.5 + sml;
                if( p.treeIndices[currentNodeIndex]<0 || fabs(x-p.x[currentNodeIndex])>limitingDist || fabs(y-p.y[currentNodeIndex])>limitingDist || fabs(z-p.z[currentNodeIndex])>limitingDist )
                {   // current node is empty (i.e. contains neither a particle nor has children), or is far enough away to be ignored
                    relNodeIndices[currentTreeDepth]++;
#ifdef DEBUG_MODE
                    if( p.treeIndices[currentNodeIndex] < 0 )
                        fprintf(stdout, "    Current node (index %d) is empty (contains -1), therefore going to next one (with rel. index %d) on same tree-depth (%d) ...\n", currentNodeIndex, relNodeIndices[currentTreeDepth], currentTreeDepth);
                    else
                        fprintf(stdout, "    Current node (index %d) is far enough away to be ignored, therefore going to next one (with rel. index %d) on same tree-depth (%d) ...\n", currentNodeIndex, relNodeIndices[currentTreeDepth], currentTreeDepth);
#endif
                }
                else if( p.treeIndices[currentNodeIndex] >= N )  // current node has children ... descend one level down to them
                {
                    relNodeIndices[currentTreeDepth]++;
                    currentTreeDepth++;
                    relNodeIndices[currentTreeDepth] = 0;
                    currentNodeLength *= 0.5;
#ifdef DEBUG_MODE
                    fprintf(stdout, "    Current node (index %d) has children, thus descend one level down to them (to tree-depth %d) ...\n", currentNodeIndex, currentTreeDepth);
#endif
                }
                else    // current node contains a particle ... compute its distance and if close enough add it to the list of interaction partner
                {
#ifdef DEBUG_MODE
                    fprintf(stdout, "    Current node (index %d) contains a particle (with index %d) ... ", currentNodeIndex, p.treeIndices[currentNodeIndex]);
#endif
                    sqrDist = (x-p.x[p.treeIndices[currentNodeIndex]])*(x-p.x[p.treeIndices[currentNodeIndex]]) + (y-p.y[p.treeIndices[currentNodeIndex]])*(y-p.y[p.treeIndices[currentNodeIndex]]) + (z-p.z[p.treeIndices[currentNodeIndex]])*(z-p.z[p.treeIndices[currentNodeIndex]]);
                    if( sqrDist < sml*sml && p.treeIndices[currentNodeIndex] != i ) // also don't add it if the found particle is the particle of interest itself
                    {
                        if( (numInteractionPartner++) > MAX_NUM_INTERACTIONS )
                            ERRORVAR("ERROR! Maximum number of interaction partner (%d) exceeded!\n",MAX_NUM_INTERACTIONS)
                        p.interactionPartner[i*MAX_NUM_INTERACTIONS+numInteractionPartner-1] = p.treeIndices[currentNodeIndex];
#ifdef DEBUG_MODE
                        fprintf(stdout, "with a distance < sml, thus adding it to list of interaction partner, increasing the numInteractionPartner to %d ...\n", numInteractionPartner);
#endif
                    }
#ifdef DEBUG_MODE
                    else
                        fprintf(stdout, "with a too large distance to interact (or being the particle of interest itself) ...\n");
#endif
                    relNodeIndices[currentTreeDepth]++;
                }
            }   // we have covered all nodes on the current level once this loop is finished, so let's get up one level:
            currentTreeDepth--;
            currentNodeLength *= 2.0;
#ifdef DEBUG_MODE
            fprintf(stdout, "    All nodes on current level covered, thus getting up one level (to tree-depth %d) ...\n", currentTreeDepth);
#endif
        } while( currentTreeDepth > 0 );
        
        p.noip[i] = numInteractionPartner;
    }   // end loop over all particles
    
#ifdef DEBUG_MODE
    for(i=0; i<N*MAX_NUM_INTERACTIONS; i++)
        fprintf(stdout, "%d\t%d\t%d\n", i, p.interactionPartner[i], p.noip[(int)(i/MAX_NUM_INTERACTIONS)]);
#endif
}   // end findInteractionPartner


void mergeSortDouble(double* array, int* indices, int noets)
// Merge-sort algorithm to sort (double) elements in 'array' (in increasing order). However, 'array' is not changed but instead 
// 'indices' returns a list of indices corresponding to the sorted 'array'. 'noets' is the length of array ("number of elements to sort").
{
    int i,j;
    int ILeft, IRight, IEnd;    // beginning of left and right pair of sublists (inclusive) and end of right one (exclusive)
    int iLeft, iRight;  // beginning of left and right head of a pair of sublists (i.e. these indices change (increase) as the sublists are processed)
    int noeis;  // current number of elements in a sublist
    int indices2[noets]; // same as 'indices' but used for intermediate storage

#ifdef SORT_DOUBLE_DEBUG
    fprintf(stdout, "\n\nstart sorting list of %d elements:\n", noets);
    for(i=0; i<noets; i++)
        fprintf(stdout, "%e\n", array[i]);
#endif
    
    // start with indices ordering according to the non-sorted array:
    for(i=0; i<noets; i++)
        indices[i] = i;

    for(noeis=1; noeis<noets; noeis*=2)   // repeat until the length of a sublist is >= noets
    {
#ifdef SORT_DOUBLE_DEBUG
        fprintf(stdout, "-------------------------------------------------\n");
        fprintf(stdout, "number-of-elements-in-sublist: %d\n", noeis);
        fprintf(stdout, "[1] current-indices-list    [2] elements-to-sort (original order)\n");
        for(i=0; i<noets; i++)
            fprintf(stdout, "%d\t%e\n", indices[i], array[i]);
#endif
        for(i=0; i<noets; i+=(2*noeis))   // each iteration corresponds to the merging of one pair of sublists
        {
            ILeft = iLeft = i;
            IRight = iRight = MIN(i+noeis, noets);
            IEnd = MIN(i+2*noeis, noets);
#ifdef SORT_DOUBLE_DEBUG
            fprintf(stdout, "  merge sublists with ILeft = %d, IRight = %d and IEnd = %d ...\n", ILeft, IRight, IEnd);
#endif
            // merge sublists and write results to indices2:
            for(j=ILeft; j<IEnd; j++)
            {
                if( iLeft < IRight && (iRight >= IEnd || array[indices[iLeft]] <= array[indices[iRight]]) )
                {
                    indices2[j] = indices[iLeft];
                    iLeft++;
                }
                else
                {
                    indices2[j] = indices[iRight];
                    iRight++;
                }
            }
        }
        // copy indices2 to indices:
        for(i=0; i<noets; i++)
            indices[i] = indices2[i];
    }
}


void mergeSortInt(int* array, int* indices, int noets)
// Merge-sort algorithm to sort (int) elements in 'array' (in increasing order). However, 'array' is not changed but instead 
// 'indices' returns a list of indices corresponding to the sorted 'array'. 'noets' is the length of array ("number of elements to sort").
{
    int i,j;
    int ILeft, IRight, IEnd;    // beginning of left and right pair of sublists (inclusive) and end of right one (exclusive)
    int iLeft, iRight;  // beginning of left and right head of a pair of sublists (i.e. these indices change (increase) as the sublists are processed)
    int noeis;  // current number of elements in a sublist
    int indices2[noets]; // same as 'indices' but used for intermediate storage
    
#ifdef SORT_INT_DEBUG
    fprintf(stdout, "\n\nstart sorting list of %d elements:\n", noets);
    for(i=0; i<noets; i++)
        fprintf(stdout, "%d\n", array[i]);
#endif
    
    // start with indices ordering according to the non-sorted array:
    for(i=0; i<noets; i++)
        indices[i] = i;
    
    for(noeis=1; noeis<noets; noeis*=2)   // repeat until the length of a sublist is >= noets
    {
#ifdef SORT_INT_DEBUG
        fprintf(stdout, "-------------------------------------------------\n");
        fprintf(stdout, "number-of-elements-in-sublist: %d\n", noeis);
        fprintf(stdout, "[1] current-indices-list    [2] elements-to-sort (original order)\n");
        for(i=0; i<noets; i++)
            fprintf(stdout, "%d\t%d\n", indices[i], array[i]);
#endif
        for(i=0; i<noets; i+=(2*noeis))   // each iteration corresponds to the merging of one pair of sublists
        {
            ILeft = iLeft = i;
            IRight = iRight = MIN(i+noeis, noets);
            IEnd = MIN(i+2*noeis, noets);
#ifdef SORT_INT_DEBUG
            fprintf(stdout, "  merge sublists with ILeft = %d, IRight = %d and IEnd = %d ...\n", ILeft, IRight, IEnd);
#endif
            // merge sublists and write results to indices2:
            for(j=ILeft; j<IEnd; j++)
            {
                if( iLeft < IRight && (iRight >= IEnd || array[indices[iLeft]] <= array[indices[iRight]]) )
                {
                    indices2[j] = indices[iLeft];
                    iLeft++;
                }
                else
                {
                    indices2[j] = indices[iRight];
                    iRight++;
                }
            }
        }
        // copy indices2 to indices:
        for(i=0; i<noets; i++)
            indices[i] = indices2[i];
    }
}
