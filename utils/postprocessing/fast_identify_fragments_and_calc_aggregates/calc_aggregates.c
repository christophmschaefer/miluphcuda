/* Based on the output of 'fast_identify_fragments' (*.frag file), this tool computes either:
 *   (1) whether the two largest fragments are mutually bound
 *   (2) the final mass, composition and kinetics (up to four different materials) of the largest aggregate(s) by computing gravitationally bound fragments
 *
 * All units are SI
 *
 * last updated: 24/Feb/2021
 * 
 * Christoph Burger, Áron Suli
 * christoph.burger@uni-tuebingen.de
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#define TRUE 1
#define FALSE 0
#define DIM 3
#define PATHLENGTH 256

#define ERRORTEXT(x) {fprintf(stderr,x); exit(1);}
#define ERRORVAR(x,y) {fprintf(stderr,x,y); exit(1);}
#define ERRORVAR2(x,y,z) {fprintf(stderr,x,y,z); exit(1);}

typedef struct __fragment   // fragment data as read from the *.frag file
{
    double x[DIM];
    double v[DIM];
    double mass;
    double rel_mass;
    double mat0_frac;
    double mat1_frac;
    double mat2_frac;
    double mat3_frac;
} _fragment;


int is_bound(_fragment* frag1, _fragment* frag2, int verbose);
void compute_aggregate(_fragment* frags, int n_frags, int* bound_frags, _fragment* aggregate);
void print(_fragment** agg, int n_agg, int n_materials, FILE* fout);
void clear(_fragment* agg);


void usage(char* programname)
{
    fprintf(stdout, "\nTool to compute various properties of a distribution of fragments, based on the output of 'fast_identify_fragments'.\n");
    fprintf(stdout, "\n  Usage: %s [Options]\n", programname);
    fprintf(stdout, "\n  Options:\n");
    fprintf(stdout, "    -?               display this message and exit\n");
    fprintf(stdout, "    -f frag-file     specify file containing the 'fast_identify_fragments' output (usually *.frag file)\n");
    fprintf(stdout, "    -n number-mat    specify number of material types in frag file (2,3 or 4)\n");
    fprintf(stdout, "    -b               set this to determine whether the two largest fragments are gravitationally bound (mutually)\n");
    fprintf(stdout, "    -g               set this to determine mass, composition and kinetics of the largest aggregate by computing all fragments gravitationally bound to it, by\n");
    fprintf(stdout, "                     starting with all directly bound fragments and then iteratively checking only those anymore (method 1)\n");
    fprintf(stdout, "    -m               set this to determine mass, composition and kinetics of the largest aggregate by computing all fragments gravitationally bound to it, by\n");
    fprintf(stdout, "                     starting with all directly bound fragments and then iteratively checking (and possibly adding) all remaining ones (method 2)\n");
    fprintf(stdout, "    -a               do both methods 1 and 2\n");
    fprintf(stdout, "    -t               do both methods 1 and 2, first for the largest aggregate, and subsequently also starting with the largest still unbound fragment to\n");
    fprintf(stdout, "                     determine the 2nd largest aggregate (fragments already included in the largest aggregate are always left there)\n");
    fprintf(stdout, "    -o output-file   optional, in addition to output to stdout: specify output file containing the barycentric coordinates and properties of the computed aggregates\n");
    fprintf(stdout, "    -v               be verbose\n");
    fprintf(stdout, "\n");
}


int main(int argc, char* argv[])
{
    int i,j,k;
    const double eps = 1.0e-6;
    const double G = 6.6741e-11;    // gravitational constant
    char fragfile[PATHLENGTH];
    char aggfile[PATHLENGTH];       // output file for the results
    FILE *ffl;
    int b_flag = FALSE;
    int g_flag = FALSE;
    int m_flag = FALSE;
    int a_flag = FALSE;
    int t_flag = FALSE;
    int verbose = FALSE;
    int n_materials = -1;   // number of different materials in the frag file
    int nrf = 0;    // number of read fragments
    _fragment* frags;
    int* bound_frags;   // array as long as the number of fragments where the i-th index = TRUE if the i-th fragment is bound (and FALSE otherwise)
    int* bound_frags_2nd;
    int* frags_in_rest;
    int repeat; // TRUE if a further iteration is necessary (i.e. the number of bound fragments has changed), otherwise FALSE
    _fragment aggregate;
    _fragment aggregate_2nd;
    _fragment aggregate_rest;
    int seed_2nd;   // index of seed fragment for accumulating 2nd-largest aggregate
    _fragment aggregate_all;  // to store some information on all matter combined
    double dist, v_rel, v_esc;    // distance, relative velocity and escape velocity between aggregates
    int aggregate_2nd_exists;
    int aggregate_rest_exists;


// initialize
    clear(&aggregate);
    clear(&aggregate_2nd);
    clear(&aggregate_rest);
    clear(&aggregate_all);
    
    fragfile[0] = '\0';
    aggfile[0] = '\0';


// process command line options
    while ( ( i = getopt(argc, argv, "?f:o:n:bgmatv") ) != -1 )	// int-representations of command line options are successively saved in i
        switch((char)i)
        {
            case '?':
                usage(*argv);
                exit(0);
            case 'f':
                strncpy(fragfile,optarg,PATHLENGTH);
                break;
            case 'o':
				strncpy(aggfile, optarg, PATHLENGTH);
				break;
            case 'n':
                n_materials = atoi(optarg);
                break;
            case 'b':
                b_flag = TRUE;
                break;
            case 'g':
                g_flag = TRUE;
                break;
            case 'm':
                m_flag = TRUE;
                break;
            case 'a':
                a_flag = TRUE;
                break;
            case 't':
                t_flag = TRUE;
                break;
            case 'v':
                verbose = TRUE;
                break;
            default:
                usage(*argv);
                exit(1);
        }
    
    if( n_materials != 2 && n_materials != 3 && n_materials != 4 )
        ERRORTEXT("ERROR! The number of materials in the fragments file has to be 2,3 or 4!\n")
    
    
// read frag file
    if ( (ffl = fopen(fragfile,"r")) == NULL )
        ERRORVAR("FILE ERROR! Cannot open %s for reading!\n",fragfile)
    fscanf(ffl, "%*[^\n]");   // '%*[^\n]' represents the whole first line, '*' means don't save to var, '[^\n]' is regular expression
    if( (frags = (_fragment*)malloc(sizeof(_fragment))) == NULL )
        ERRORTEXT("ERROR during memory allocation!\n")
    
    if( n_materials == 2 )
        while( fscanf(ffl, "\n%le %le %le %le %le %le %le %le %le %le%*[^\n]", &frags[nrf].x[0], &frags[nrf].x[1], &frags[nrf].x[2], &frags[nrf].v[0], &frags[nrf].v[1], &frags[nrf].v[2], 
            &frags[nrf].mass, &frags[nrf].rel_mass, &frags[nrf].mat0_frac, &frags[nrf].mat1_frac) == 10 )
        {
            nrf++;
            if( (frags = realloc(frags, sizeof(_fragment)*(nrf+1))) == NULL )
                ERRORTEXT("ERROR during memory allocation!\n")
        }
    if( n_materials == 3 )
        while( fscanf(ffl, "\n%le %le %le %le %le %le %le %le %le %le %le%*[^\n]", &frags[nrf].x[0], &frags[nrf].x[1], &frags[nrf].x[2], &frags[nrf].v[0], &frags[nrf].v[1], &frags[nrf].v[2], 
            &frags[nrf].mass, &frags[nrf].rel_mass, &frags[nrf].mat0_frac, &frags[nrf].mat1_frac, &frags[nrf].mat2_frac) == 11 )
        {
            nrf++;
            if( (frags = realloc(frags, sizeof(_fragment)*(nrf+1))) == NULL )
                ERRORTEXT("ERROR during memory allocation!\n")
        }
    if( n_materials == 4 )
        while( fscanf(ffl, "\n%le %le %le %le %le %le %le %le %le %le %le %le%*[^\n]", &frags[nrf].x[0], &frags[nrf].x[1], &frags[nrf].x[2], &frags[nrf].v[0], &frags[nrf].v[1], &frags[nrf].v[2], 
            &frags[nrf].mass, &frags[nrf].rel_mass, &frags[nrf].mat0_frac, &frags[nrf].mat1_frac, &frags[nrf].mat2_frac, &frags[nrf].mat3_frac) == 12 )
        {
            nrf++;
            if( (frags = realloc(frags, sizeof(_fragment)*(nrf+1))) == NULL )
                ERRORTEXT("ERROR during memory allocation!\n")
        }
    fclose(ffl);


// compute whether the two largest fragments are gravitationally bound
    if( b_flag )
    {
        if( g_flag || m_flag || a_flag || t_flag )
            ERRORTEXT("ERROR! Something's wrong with your command line flags!\n")
        if( nrf < 2 )
            ERRORTEXT("ERROR! The number of fragments in the chosen *.frag file is only 1 ...!\n")
        if( verbose )
            fprintf(stdout, "\nMODE: Determine whether the 2 largest fragments are gravitationally bound (mutually)\n");
        if( is_bound(frags, frags+1, verbose) == TRUE )
            fprintf(stdout, "BOUND!\n\n");
        else
            fprintf(stdout, "NOT BOUND!\n\n");
    }
    
    
// compute largest aggregate by computing (all) gravitationally bound fragments (all methods)
    if( g_flag || m_flag || a_flag || t_flag )
    {
        if( b_flag )
            ERRORTEXT("ERROR! Something's wrong with your command line flags!\n")
        if( verbose && g_flag )
            fprintf(stdout, "\nMODE: Compute the final mass, composition and kinetics of the largest aggregate by computing (all) gravitationally bound fragments, by starting with all directly bound fragments and then iteratively checking only those anymore\n");
        if( verbose && m_flag )
            fprintf(stdout, "\nMODE: Compute the final mass, composition and kinetics of the largest aggregate by computing (all) gravitationally bound fragments, by starting with all directly bound fragments and then iteratively checking (and possibly adding) all remaining ones\n");
        if( verbose && a_flag )
            fprintf(stdout, "\nMODE: Compute the final mass, composition and kinetics of the largest aggregate by computing (all) gravitationally bound fragments, by starting with all directly bound fragments and then iteratively checking (and possibly adding or removing) all fragments (bound and unbound)\n");
        if( verbose && t_flag )
            fprintf(stdout, "\nMODE: Compute the final mass, composition and kinetics of the two largest aggregates by computing (all) gravitationally bound fragments, by (1) starting with all fragments directly bound to the largest and then iteratively checking (and possibly adding or removing) all fragments (bound and unbound), and (2) repeating this procedure for finding the 2nd-largest aggregate starting from the largest remaining unbound fragment\n");
        if( (bound_frags = (int*)malloc(nrf*sizeof(int))) == NULL )
            ERRORTEXT("ERROR during memory allocation!\n")
        if( verbose && t_flag )
            fprintf(stdout, "Start computing largest aggregate ...\n");
        
        // set flags in bound_frags by checking all fragments whether they are bound to the largest
        if( verbose )
            fprintf(stdout, "Initial computation for all fragments whether they are bound to the largest (m = %g kg) ...\n", frags[0].mass);
        for(i=0; i<nrf; i++)
            bound_frags[i] = FALSE;
        bound_frags[0] = TRUE;
        for(i=1; i<nrf; i++)
            if( is_bound(&frags[0], &frags[i], FALSE) )
                bound_frags[i] = TRUE;
        
        // start iteration
        repeat = TRUE;
        k = 0;
        while( repeat )
        {
            k++;
            if( verbose )
                fprintf(stdout, "Start iteration no. %d ...\n", k);
            repeat = FALSE;
            // compute aggregated mass, and the position and velocity of the center-of-mass according to information in bound_frags
            compute_aggregate(frags, nrf, bound_frags, &aggregate);
            if( verbose )
                fprintf(stdout, "The (current) aggregate has m = %g kg\n", aggregate.mass);
            // update bound_frags w.r.t. this aggregated mass
            if( g_flag )
            {
                for(i=0; i<nrf; i++)
                    if( bound_frags[i] == TRUE )
                        if( !is_bound(&aggregate, &frags[i], FALSE) )
                        {
                            bound_frags[i] = FALSE;
                            if( verbose )
                                fprintf(stdout, "The fragment no. %d, m = %g kg, is not bound to the (current) aggregate, m = %g kg, and left the list of bound fragments.\n", i, frags[i].mass, aggregate.mass);
                            repeat = TRUE;
                        }
            }
            if( m_flag )
            {
                for(i=0; i<nrf; i++)
                    if( bound_frags[i] == FALSE )
                        if( is_bound(&aggregate, &frags[i], FALSE) )
                        {
                            bound_frags[i] = TRUE;
                            if( verbose )
                                fprintf(stdout, "The fragment no. %d, m = %g kg, is actually bound to the (current) aggregate, m = %g kg, and was added to the list of bound fragments.\n", i, frags[i].mass, aggregate.mass);
                            repeat = TRUE;
                        }
            }
            if( a_flag || t_flag )
            {
                for(i=0; i<nrf; i++)
                    if( is_bound(&aggregate, &frags[i], FALSE) )
                    {
                        if( bound_frags[i] == FALSE )
                        {
                            bound_frags[i] = TRUE;
                            if( verbose )
                                fprintf(stdout, "The fragment no. %d, m = %g kg, is actually bound to the (current) aggregate, m = %g kg, and was added to the list of bound fragments.\n", i, frags[i].mass, aggregate.mass);
                            repeat = TRUE;
                        }
                    }
                    else
                    {
                        if( bound_frags[i] == TRUE )
                        {
                            bound_frags[i] = FALSE;
                            if( verbose )
                                fprintf(stdout, "The fragment no. %d, m = %g kg, is not bound to the (current) aggregate, m = %g kg, and left the list of bound fragments.\n", i, frags[i].mass, aggregate.mass);
                            repeat = TRUE;
                        }
                    }
            }
        }
        
        // compute and print properties of the aggregate (the mass, position and velocity from the last iteration above is still valid)
        if( verbose )
            fprintf(stdout, "Iteration finished. Properties of the largest aggregate:\n");
        aggregate.mat0_frac = aggregate.mat1_frac = aggregate.mat2_frac = aggregate.mat3_frac = 0.0;
        for(i=0; i<nrf; i++)
            if( bound_frags[i] == TRUE )
            {
                aggregate.mat0_frac += frags[i].mat0_frac * frags[i].mass;  // used as intermediate storage
                aggregate.mat1_frac += frags[i].mat1_frac * frags[i].mass;
                if( n_materials == 3 || n_materials == 4 )
                    aggregate.mat2_frac += frags[i].mat2_frac * frags[i].mass;
                if( n_materials == 4 )
                    aggregate.mat3_frac += frags[i].mat3_frac * frags[i].mass;
            }
        aggregate.mat0_frac /= aggregate.mass;
        aggregate.mat1_frac /= aggregate.mass;
        aggregate.mat2_frac /= aggregate.mass;
        aggregate.mat3_frac /= aggregate.mass;
        
        fprintf(stdout, "# largest aggregate:\n#    mass");
        for(i=0; i<n_materials; i++)
            fprintf(stdout, "    fraction_mat%d", i);
        fprintf(stdout, "\n%.16le\t%.16le\t%.16le", aggregate.mass, aggregate.mat0_frac, aggregate.mat1_frac);
        if( n_materials == 3 || n_materials == 4 )
            fprintf(stdout, "\t%.16le", aggregate.mat2_frac);
        if( n_materials == 4 )
            fprintf(stdout, "\t%.16le", aggregate.mat3_frac);
        fprintf(stdout, "\n");
    }
    
    
// compute 2nd-largest aggregate in addition
    if( t_flag )
    {
        aggregate_2nd_exists = TRUE;    // set true until proven otherwise
        if( verbose )
            fprintf(stdout, "--------------------------------\nStart computing 2nd-largest aggregate ...\n");
        if( (bound_frags_2nd = (int*)malloc(nrf*sizeof(int))) == NULL )
            ERRORTEXT("ERROR during memory allocation!\n")
        // find largest remaining unbound fragment (result will be its index, stored in seed_2nd)
        seed_2nd = 0;
        while( bound_frags[seed_2nd] != FALSE )
        {
            seed_2nd++;
            if( seed_2nd >= nrf )   // there is no 2nd largest aggregate; all fragments are in the largest
            {
                if(verbose)
                    fprintf(stdout, "WARNING: No 2nd-largest aggregate. All mass seems to be in most massive one.\n");
                aggregate_2nd.mass = aggregate_2nd.mat0_frac = aggregate_2nd.mat1_frac = aggregate_2nd.mat2_frac = aggregate_2nd.mat3_frac = -1.0;
                aggregate_2nd_exists = FALSE;
                break;
            }
        }
        if( aggregate_2nd_exists )
        {
            if( verbose )
                fprintf(stdout, "Fragment no. %d, m = %g kg was identified as the largest remaining unbound fragment.\n", seed_2nd, frags[seed_2nd].mass);
            // set flags in bound_frags_2nd by checking all fragments whether they are bound to the largest remaining unbound one
            if( verbose )
                fprintf(stdout, "Initial computation for all fragments whether they are bound to this largest remaining unbound one ...\n");
            for(i=0; i<nrf; i++)
                bound_frags_2nd[i] = FALSE;
            bound_frags_2nd[seed_2nd] = TRUE;
            for(i=0; i<nrf; i++)
                if( i != seed_2nd )
                    if( is_bound(&frags[seed_2nd], &frags[i], FALSE) )
                    {
                        if( bound_frags[i] == FALSE )    // if fragment i is not already bound to most massive aggregate
                            bound_frags_2nd[i] = TRUE;
                        else if( verbose )
                            fprintf(stdout, "WARNING: Fragment no. %d, m = %g kg is bound to the largest remaining unbound one, but already part of the largest aggregate - therefore discarded!\n", i, frags[i].mass);
                    }
            // start iteration
            repeat = TRUE;
            k = 0;
            while( repeat )
            {
                k++;
                if( verbose )
                    fprintf(stdout, "Start iteration no. %d ...\n", k);
                repeat = FALSE;
                // compute aggregated mass, and the position and velocity of the center-of-mass according to information in bound_frags_2nd
                compute_aggregate(frags, nrf, bound_frags_2nd, &aggregate_2nd);
                if( verbose )
                    fprintf(stdout, "The (current) aggregate_2nd has m = %g kg\n", aggregate_2nd.mass);
                // update bound_frags_2nd w.r.t. this aggregated mass
                for(i=0; i<nrf; i++)
                    if( is_bound(&aggregate_2nd, &frags[i], FALSE) )
                    {
                        if( bound_frags_2nd[i] == FALSE )
                        {
                            if( bound_frags[i] == FALSE )    // if fragment i is not already bound to most massive aggregate
                            {
                                bound_frags_2nd[i] = TRUE;
                                if( verbose )
                                    fprintf(stdout, "The fragment no. %d, m = %g kg, is actually bound to the (current) aggregate_2nd, m = %g kg, and was added to the list of bound fragments 2nd.\n", i, frags[i].mass, aggregate_2nd.mass);
                                repeat = TRUE;
                            }
                            else if( verbose )
                                fprintf(stdout, "WARNING: Fragment no. %d, m = %g kg is bound to the 2nd-largest aggregate, but already part of the largest aggregate - therefore discarded!\n", i, frags[i].mass);
                        }
                    }
                    else
                    {
                        if( bound_frags_2nd[i] == TRUE )
                        {
                            bound_frags_2nd[i] = FALSE;
                            if( verbose )
                                fprintf(stdout, "The fragment no. %d, m = %g kg, is not bound to the (current) aggregate_2nd, m = %g kg, and left the list of bound fragments 2nd.\n", i, frags[i].mass, aggregate_2nd.mass);
                            repeat = TRUE;
                        }
                    }
            } // end while( repeat )
            
            // compute and print properties of the aggregate_2nd (the mass, position and velocity from the last iteration above is still valid)
            if( verbose )
                fprintf(stdout, "Iteration finished. Properties of the 2nd-largest aggregate:\n");
            aggregate_2nd.mat0_frac = aggregate_2nd.mat1_frac = aggregate_2nd.mat2_frac = aggregate_2nd.mat3_frac = 0.0;
            for(i=0; i<nrf; i++)
                if( bound_frags_2nd[i] == TRUE )
                {
                    aggregate_2nd.mat0_frac += frags[i].mat0_frac * frags[i].mass;  //used as intermediate storage
                    aggregate_2nd.mat1_frac += frags[i].mat1_frac * frags[i].mass;
                    if( n_materials == 3 || n_materials == 4 )
                        aggregate_2nd.mat2_frac += frags[i].mat2_frac * frags[i].mass;
                    if( n_materials == 4 )
                        aggregate_2nd.mat3_frac += frags[i].mat3_frac * frags[i].mass;
                }
            aggregate_2nd.mat0_frac /= aggregate_2nd.mass;
            aggregate_2nd.mat1_frac /= aggregate_2nd.mass;
            aggregate_2nd.mat2_frac /= aggregate_2nd.mass;
            aggregate_2nd.mat3_frac /= aggregate_2nd.mass;
        }   // end if( aggregate_2nd_exists )
        
        fprintf(stdout, "#\n# 2nd-largest aggregate:\n#    mass");
        for(i=0; i<n_materials; i++)
            fprintf(stdout, "    fraction_mat%d", i);
        fprintf(stdout, "\n%.16le\t%.16le\t%.16le", aggregate_2nd.mass, aggregate_2nd.mat0_frac, aggregate_2nd.mat1_frac);
        if( n_materials == 3 || n_materials == 4 )
            fprintf(stdout, "\t%.16le", aggregate_2nd.mat2_frac);
        if( n_materials == 4 )
            fprintf(stdout, "\t%.16le", aggregate_2nd.mat3_frac);
        fprintf(stdout, "\n");
        if( verbose )
            fprintf(stdout, "--------------------------------\n");
    }   // end if( t_flag )
    
    
// compute rest
    if( g_flag || m_flag || a_flag || t_flag )
    {
        if( verbose )
            fprintf(stdout, "--------------------------------\nStart computing rest of material ...\n");
        aggregate_rest.mass = aggregate_rest.mat0_frac = aggregate_rest.mat1_frac = aggregate_rest.mat2_frac = aggregate_rest.mat3_frac = -1.0;
        aggregate_rest_exists = FALSE;
        
        // compute 'frags_in_rest'
        if( (frags_in_rest = (int*)malloc(nrf*sizeof(int))) == NULL )
            ERRORTEXT("ERROR during memory allocation!\n")
        for(i=0; i<nrf; i++)
            frags_in_rest[i] = FALSE;
        if( g_flag || m_flag || a_flag )
        {
            for(i=0; i<nrf; i++)
                if( bound_frags[i] == FALSE )
                {
                    frags_in_rest[i] = TRUE;
                    aggregate_rest_exists = TRUE;
                }
        }
        else if( t_flag )
        {
            for(i=0; i<nrf; i++)
                if( bound_frags[i] == FALSE  &&  bound_frags_2nd[i] == FALSE )
                {
                    frags_in_rest[i] = TRUE;
                    aggregate_rest_exists = TRUE;
                }
        }
        
        if( aggregate_rest_exists )
        {
            // compute aggregated mass in the rest of material, and the position and velocity of its center-of-mass according to information in 'frags_in_rest'
            compute_aggregate(frags, nrf, frags_in_rest, &aggregate_rest);
            
            // compute mat-fractions of the rest of material
            aggregate_rest.mat0_frac = aggregate_rest.mat1_frac = aggregate_rest.mat2_frac = aggregate_rest.mat3_frac = 0.0;
            for(i=0; i<nrf; i++)
                if( frags_in_rest[i] == TRUE )
                {
                    aggregate_rest.mat0_frac += frags[i].mat0_frac * frags[i].mass;  //used as intermediate storage
                    aggregate_rest.mat1_frac += frags[i].mat1_frac * frags[i].mass;
                    if( n_materials == 3 || n_materials == 4 )
                        aggregate_rest.mat2_frac += frags[i].mat2_frac * frags[i].mass;
                    if( n_materials == 4 )
                        aggregate_rest.mat3_frac += frags[i].mat3_frac * frags[i].mass;
                }
            aggregate_rest.mat0_frac /= aggregate_rest.mass;
            aggregate_rest.mat1_frac /= aggregate_rest.mass;
            aggregate_rest.mat2_frac /= aggregate_rest.mass;
            aggregate_rest.mat3_frac /= aggregate_rest.mass;
        }
        
        fprintf(stdout, "#\n# rest of material:\n#    mass");
        for(i=0; i<n_materials; i++)
            fprintf(stdout, "    fraction_mat%d", i);
        fprintf(stdout, "\n%.16le\t%.16le\t%.16le", aggregate_rest.mass, aggregate_rest.mat0_frac, aggregate_rest.mat1_frac);
        if( n_materials == 3 || n_materials == 4 )
            fprintf(stdout, "\t%.16le", aggregate_rest.mat2_frac);
        if( n_materials == 4 )
            fprintf(stdout, "\t%.16le", aggregate_rest.mat3_frac);
        fprintf(stdout, "\n");
        if( verbose )
            fprintf(stdout, "--------------------------------\n");
    }   // end 'compute rest'
    
    
// compute overall scenario data
    if( g_flag || m_flag || a_flag || t_flag )
    {
        if( verbose )
            fprintf(stdout, "--------------------------------\nStart computing data on all material combined ...\n");
        
        // compute total mass and mat-fractions
        aggregate_all.mass = aggregate_all.mat0_frac = aggregate_all.mat1_frac = aggregate_all.mat2_frac = aggregate_all.mat3_frac = 0.0;
        for(i=0; i<nrf; i++)
        {
            aggregate_all.mass += frags[i].mass;
            aggregate_all.mat0_frac += frags[i].mass * frags[i].mat0_frac;  //used as intermediate storage
            aggregate_all.mat1_frac += frags[i].mass * frags[i].mat1_frac;
            if( n_materials == 3 || n_materials == 4 )
                aggregate_all.mat2_frac += frags[i].mass * frags[i].mat2_frac;
            if( n_materials == 4 )
                aggregate_all.mat3_frac += frags[i].mass * frags[i].mat3_frac;
        }
        aggregate_all.mat0_frac /= aggregate_all.mass;
        aggregate_all.mat1_frac /= aggregate_all.mass;
        aggregate_all.mat2_frac /= aggregate_all.mass;
        aggregate_all.mat3_frac /= aggregate_all.mass;
        
        // compute pos. and vel. of center-of-mass
        for(i=0; i<DIM; i++)
            aggregate_all.x[i] = aggregate_all.v[i] = 0.0;
        for(i=0; i<nrf; i++)
        {
            for(j=0; j<DIM; j++)
            {
                aggregate_all.x[j] += frags[i].mass * frags[i].x[j];
                aggregate_all.v[j] += frags[i].mass * frags[i].v[j];
            }
        }
        for(i=0; i<DIM; i++)
        {
            aggregate_all.x[i] /= aggregate_all.mass;
            aggregate_all.v[i] /= aggregate_all.mass;
        }
        
        fprintf(stdout, "#\n# overall scenario data:\n#    mass");
        for(i=0; i<n_materials; i++)
            fprintf(stdout, "    fraction_mat%d", i);
        fprintf(stdout, "\n%.16le\t%.16le\t%.16le", aggregate_all.mass, aggregate_all.mat0_frac, aggregate_all.mat1_frac);
        if( n_materials == 3 || n_materials == 4 )
            fprintf(stdout, "\t%.16le", aggregate_all.mat2_frac);
        if( n_materials == 4 )
            fprintf(stdout, "\t%.16le", aggregate_all.mat3_frac);
        fprintf(stdout, "\n");
    }
    
    
// print positions and velocities to stdout
    if( g_flag || m_flag || a_flag || t_flag )
    {
        if( g_flag || m_flag || a_flag )
            fprintf(stdout, "#\n# pos and vel of (1) largest, (2) rest of material, and (3) all scenario material:\n#    x1  x2  x3  v1  v2  v3\n");
        if( t_flag )
            fprintf(stdout, "#\n# pos and vel of (1) largest, (2) 2nd largest, (3) rest of material, and (4) all scenario material:\n#    x1  x2  x3  v1  v2  v3\n");
        
        fprintf(stdout, "%.16le\t%.16le\t%.16le\t%.16le\t%.16le\t%.16le\n", aggregate.x[0], aggregate.x[1], aggregate.x[2], aggregate.v[0], aggregate.v[1], aggregate.v[2]);
        
        if( t_flag )
        {
            if( aggregate_2nd_exists ) {
                fprintf(stdout, "%.16le\t%.16le\t%.16le\t%.16le\t%.16le\t%.16le\n", aggregate_2nd.x[0], aggregate_2nd.x[1], aggregate_2nd.x[2], aggregate_2nd.v[0], aggregate_2nd.v[1], aggregate_2nd.v[2]);
            }
            else {
                fprintf(stdout, "-1.0\t-1.0\t-1.0\t-1.0\t-1.0\t-1.0\n");
            }
        }
        
        if( aggregate_rest_exists ) {
            fprintf(stdout, "%.16le\t%.16le\t%.16le\t%.16le\t%.16le\t%.16le\n", aggregate_rest.x[0], aggregate_rest.x[1], aggregate_rest.x[2], aggregate_rest.v[0], aggregate_rest.v[1], aggregate_rest.v[2]);
        }
        else {
            fprintf(stdout, "-1.0\t-1.0\t-1.0\t-1.0\t-1.0\t-1.0\n");
        }
        
        fprintf(stdout, "%.16le\t%.16le\t%.16le\t%.16le\t%.16le\t%.16le\n", aggregate_all.x[0], aggregate_all.x[1], aggregate_all.x[2], aggregate_all.v[0], aggregate_all.v[1], aggregate_all.v[2]);
    }


// write results to file (in addition to output to stdout)
    if (0 < strlen(aggfile))
    {
        int n_agg = 2; // largest aggregate + all aggregates
        if (aggregate_2nd_exists) n_agg++;
        if (aggregate_rest_exists) n_agg++;

        _fragment** agg = (_fragment**)malloc(n_agg * sizeof(_fragment*));
        if (NULL == agg)
        {
            printf("Error [ln: %d]: memory allocation.\n", __LINE__);
            exit(EXIT_FAILURE);
        }
        int k = 0;
        agg[k] = &aggregate;
        if (aggregate_2nd_exists)
        {
            k++;
            agg[k] = &aggregate_2nd;
        }
        if (aggregate_rest_exists) 
        {
            k++;
            agg[k] = &aggregate_rest;
        }
        k++;
        agg[k] = &aggregate_all;

        FILE* fagg = fopen(aggfile, "wt");
        if (NULL == fagg)
        {
            printf("Error [ln: %d]: could not open file: %s.\n", __LINE__, aggfile);
            exit(EXIT_FAILURE);
        }
        print(agg, n_agg, n_materials, fagg);

        fclose(fagg);
        free(agg);
    }


// compute and print information on kinetics to stdout
    if( g_flag || m_flag || a_flag || t_flag )
    {
        if( g_flag || m_flag || a_flag )
            fprintf(stdout, "#\n# kinetics of largest + rest:\n#    distance    v_rel    v_rel/v_esc\n");
        if( t_flag )
            fprintf(stdout, "#\n# kinetics of (1) largest + 2nd-largest, (2) largest + rest, (3) 2nd-largest + rest:\n#    distance    v_rel    v_rel/v_esc\n");
        
        // largest + 2nd-largest
        if( t_flag )
        {
            if( aggregate_2nd_exists )
            {
                dist = v_rel = 0.0;
                for(i=0; i<DIM; i++)
                    dist += pow( aggregate_2nd.x[i] - aggregate.x[i], 2 );
                dist = sqrt(dist);
                for(i=0; i<DIM; i++)
                    v_rel += pow( aggregate_2nd.v[i] - aggregate.v[i], 2 );
                v_rel = sqrt(v_rel);
                v_esc = sqrt( 2.0*G*(aggregate.mass+aggregate_2nd.mass)/dist );
                fprintf(stdout, "%.16le\t%.16le\t%.16le\n", dist, v_rel, v_rel/v_esc);
            }
            else {
                fprintf(stdout, "-1.0\t-1.0\t-1.0\n");
            }
        }
        
        // largest + rest
        if( aggregate_rest_exists )
        {
            dist = v_rel = 0.0;
            for(i=0; i<DIM; i++)
                dist += pow( aggregate_rest.x[i] - aggregate.x[i], 2 );
            dist = sqrt(dist);
            for(i=0; i<DIM; i++)
                v_rel += pow( aggregate_rest.v[i] - aggregate.v[i], 2 );
            v_rel = sqrt(v_rel);
            v_esc = sqrt( 2.0*G*(aggregate.mass+aggregate_rest.mass)/dist );
            fprintf(stdout, "%.16le\t%.16le\t%.16le\n", dist, v_rel, v_rel/v_esc);
        }
        else {
            fprintf(stdout, "-1.0\t-1.0\t-1.0\n");
        }
        
        // 2nd-largest + rest
        if( t_flag )
        {
            if( aggregate_2nd_exists && aggregate_rest_exists )
            {
                dist = v_rel = 0.0;
                for(i=0; i<DIM; i++)
                    dist += pow( aggregate_rest.x[i] - aggregate_2nd.x[i], 2 );
                dist = sqrt(dist);
                for(i=0; i<DIM; i++)
                    v_rel += pow( aggregate_rest.v[i] - aggregate_2nd.v[i], 2 );
                v_rel = sqrt(v_rel);
                v_esc = sqrt( 2.0*G*(aggregate_2nd.mass+aggregate_rest.mass)/dist );
                fprintf(stdout, "%.16le\t%.16le\t%.16le\n", dist, v_rel, v_rel/v_esc);
            }
            else {
                fprintf(stdout, "-1.0\t-1.0\t-1.0\n");
            }
        }
    }


// run some consistency checks
    if( g_flag || m_flag || a_flag || t_flag )
    {
        double tmp_sum;
        double tmp_x[DIM];
        double tmp_v[DIM];
        
        // check sum of masses
        tmp_sum = aggregate.mass;
        if( t_flag && aggregate_2nd_exists )
            tmp_sum += aggregate_2nd.mass;
        if( aggregate_rest_exists )
            tmp_sum += aggregate_rest.mass;
        if( fabs(aggregate_all.mass - tmp_sum) / aggregate_all.mass > eps )
            ERRORVAR2("ERROR. Consistency check for masses failed. 'aggregate_all.mass' = %.16le, 'tmp_sum' = %.16le ...\n", aggregate_all.mass, tmp_sum)
        
        // check merged aggregates
        for(i=0; i<DIM; i++) {
            tmp_x[i] = aggregate.mass * aggregate.x[i];
            tmp_v[i] = aggregate.mass * aggregate.v[i];
        }
        if( t_flag && aggregate_2nd_exists )
            for(i=0; i<DIM; i++) {
                tmp_x[i] += aggregate_2nd.mass * aggregate_2nd.x[i];
                tmp_v[i] += aggregate_2nd.mass * aggregate_2nd.v[i];
            }
        if( aggregate_rest_exists )
            for(i=0; i<DIM; i++) {
                tmp_x[i] += aggregate_rest.mass * aggregate_rest.x[i];
                tmp_v[i] += aggregate_rest.mass * aggregate_rest.v[i];
            }
        for(i=0; i<DIM; i++) {
            tmp_x[i] /= tmp_sum;
            tmp_v[i] /= tmp_sum;
        }
        for(i=0; i<DIM; i++) {
            if(  (fabs(aggregate_all.x[i] - tmp_x[i]) / fabs(aggregate_all.x[i]) > eps)  ||  (fabs(aggregate_all.v[i] - tmp_v[i]) / fabs(aggregate_all.v[i]) > eps)  )
            {
                fprintf(stderr, "ERROR. Consistency check for pos. and vel. failed.  (1) 'aggregate_all.x'  (2) 'tmp_x'  (3) 'aggregate_all.v'  (4) 'tmp_v':\n");
                for(j=0; j<DIM; j++)
                    fprintf(stderr, "    %.16le\t%.16le\t%.16le\t%.16le\n", aggregate_all.x[j], tmp_x[j], aggregate_all.v[j], tmp_v[j]);
                exit(1);
            }
        }
    }

// clean up
    free(frags);
    if( g_flag || m_flag || a_flag || t_flag)
    {
        free(bound_frags);
        free(frags_in_rest);
    }
    if( t_flag )
        free(bound_frags_2nd);
    return(0);
}



int is_bound(_fragment* frag1, _fragment* frag2, int verbose)
// checks whether 2 fragments, passed in 'frag1' and 'frag2', are mutually gravitationally bound. Returns TRUE if bound, and FALSE if not.
{
    const double G = 6.6741e-11;    // gravitational constant
    int i;
    double x[DIM];  // relative position vector
    double v[DIM];  // relative velocity vector
    double r = 0.0;   // mutual distance
    double v_rel = 0.0;   // mutual speed
    double v_esc = 0.0;
    double mu = 0.0;  // gravitational parameter = G*(m1+m2)
    
    // calculate the relative position vector and velocity vector
    for(i=0; i<DIM; i++)
        x[i] = frag2->x[i] - frag1->x[i];
    for(i=0; i<DIM; i++)
        v[i] = frag2->v[i] - frag1->v[i];
    // calculate distance and speed as norm of these vectors
    for(i=0; i<DIM; i++)
        r += x[i] * x[i];
    r = sqrt(r);
    for(i=0; i<DIM; i++)
        v_rel += v[i] * v[i];
    v_rel = sqrt(v_rel);
    
    mu = G*( frag1->mass + frag2->mass );
    v_esc = sqrt( 2.0*mu/r );
    
    if( verbose )
        fprintf(stdout, "The fragments' (%g kg and %g kg) mutual escape velocity (at r = %g m) is %g m/s. The relative speed (%g m/s) is %g times this value.\n", frag1->mass, frag2->mass, r, v_esc, v_rel, v_rel/v_esc);
    
    if( v_rel/v_esc < 1.0 )
        return(TRUE);
    else
        return(FALSE);
}



void compute_aggregate(_fragment* frags, int n_frags, int* bound_frags, _fragment* aggregate)
// Computes aggregated mass, and the position and velocity of the center-of-mass according to information in 
// 'bound_frags' (and stores it all in aggregate). 'frags' is an array of length 'n_frags' and contains information on all fragments.
{
    int i,j;
    
    aggregate->mass = 0.0;
    for(i=0; i<DIM; i++)
        aggregate->x[i] = aggregate->v[i] = 0.0;
    for(i=0; i<n_frags; i++)
        if( bound_frags[i] == TRUE )
        {
            aggregate->mass += frags[i].mass;
            for(j=0; j<DIM; j++)
            {
                aggregate->x[j] += frags[i].mass * frags[i].x[j];
                aggregate->v[j] += frags[i].mass * frags[i].v[j];
            }
        }
    for( i=0; i<DIM; i++)
    {
        aggregate->x[i] /= aggregate->mass;
        aggregate->v[i] /= aggregate->mass;
    }
}



void print(_fragment** agg, int n_agg, int n_materials, FILE* fout)
{
    for(int i = 0; i < n_agg; i++)
    {
        for (int j = 0; j < DIM; j++) {
            fprintf(fout, "%24.16le ", agg[i]->x[j]);
        }
        for (int j = 0; j < DIM; j++) {
            fprintf(fout, "%24.16le ", agg[i]->v[j]);
        }
        fprintf(fout, "%24.16le ", agg[i]->mass);
        fprintf(fout, "%24.16le ", agg[i]->rel_mass);
        if (1 == n_materials)
            fprintf(fout, "%24.16le\n", agg[i]->mat0_frac);
        else if (2 == n_materials)
            fprintf(fout, "%24.16le %24.16le\n", agg[i]->mat0_frac, agg[i]->mat1_frac);
        else if (3 == n_materials)
            fprintf(fout, "%24.16le %24.16le %24.16le\n", agg[i]->mat0_frac, agg[i]->mat1_frac, agg[i]->mat2_frac);
        else if (4 == n_materials)
            fprintf(fout, "%24.16le %24.16le %24.16le %24.16le\n", agg[i]->mat0_frac, agg[i]->mat1_frac, agg[i]->mat2_frac, agg[i]->mat3_frac);
        else
            fprintf(fout, "\n");
    }
}



void clear(_fragment* agg)
{
	for (int j = 0; j < DIM; j++) {
        agg->x[j] = 0.0;
        agg->v[j] = 0.0;
    }
    agg->mass = agg->rel_mass = 0.0;
    agg->mat0_frac = agg->mat1_frac = agg->mat2_frac = agg->mat3_frac = 0.0;
}
