#!/usr/bin/env python

"""
Generates xdmf file from .h5 miluphcuda/miluphhpc output files for Paraview postprocessing.

Authors: Christoph Schäfer, Christoph Burger
Last updated: 30/May/2025

"""


try:
    import h5py
    import sys
    import argparse
    import os
except Exception as e:
    print("Error. Could not properly import modules. Exiting...")
    print(str(type(e).__name__))
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)


def parse_args():
    # get all *.h5 files in current dir
    h5files = []
    for file in sorted(os.listdir(os.getcwd())):
        if file.endswith('.h5'):
            h5files.append(file)

    parser = argparse.ArgumentParser(
        description='Generates xdmf file from .h5 miluphcuda/miluphpc output files for Paraview postprocessing. '
                    'Then open the generated .xdmf file with Paraview. '
                    'Usually the defaults will produce what you want for miluphcuda (if cwd contains the .h5 files). '
                    'For miluphpc add --miluphpc.')

    parser.add_argument('--output', help='output file name, default is paraview.xdmf', default='paraview.xdmf')
    parser.add_argument('--dim', help='dimension, default is 3', default='3')
    parser.add_argument('--miluphpc', help='hdf5 files are miluphpc output files', action='store_true')
    parser.add_argument('--input_files', nargs='+', help='input file names, default is *.h5', default=h5files)
    parser.add_argument('--add_attr', nargs='+',
                        help='Additional attributes to read from the .h5 files. You only need this if the automatic '
                             'detection fails. Use <h5ls HDF5-file> to get list of all attributes in a file.')

    args = parser.parse_args()
    return args


def write_xdmf_header(fh):
    header = """
<Xdmf>
<Domain Name="MSI">
<Grid Name="CellTime" GridType="Collection" CollectionType="Temporal">
"""
    fh.write(header)


def write_xdmf_footer(fh):
    footer = """
</Grid>
</Domain>
</Xdmf>
"""
    fh.write(footer)


# main
if __name__ == "__main__":
    args = parse_args()

    xdmfh = open(args.output, 'w')

    # write header of xdmf file
    write_xdmf_header(xdmfh)


    # get relevant fields from first h5 file in the list
    hfile = args.input_files[0]
    try:
        f = h5py.File(hfile, 'r')
    except IOError:
        print("Cannot open %s, exiting." % hfile)
        sys.exit(1)

    # list of all attributes to search for (only scalars for now, see processing below)
    modern = args.miluphpc
    if (args.miluphpc): 
        print("Assuming miluphc output files.") 
        possible_attributes = [ 'Sxx', 'Sxy', 'Syy', 'cs', 'dSdtxx', 'dSdtxy', 'dSdtyy', 'drhodt', 'e', 'localStrain', 'm', 'noi', 'p', 'proc', 'rho', 'sml' ]
    else:
        possible_attributes = [ 'aneos_T', 'aneos_cs', 'aneos_entropy', 'aneos_phase_flag',
                                'rho', 'p', 'e', 'm', 'local_strain', 'material_type', 'soundspeed',
                                'sml', 'sml_initial', 'number_of_interactions', 'tree_depth',
                                'cs_min', 'e_min', 'p_min', 'rho_min', 'cs_max', 'e_max', 'p_max', 'rho_max',
                                'deviatoric_stress', 'DIM_root_of_damage_tensile', 'number_of_activated_flaws',
                                'alpha_jutzi', 'DIM_root_of_damage_porjutzi', 'damage_total', 'total_plastic_strain' ]
    # find matching attributes in h5 file
    wanted_attributes = []
    for attr in possible_attributes:
        if f.__contains__(attr):
            wanted_attributes.append(attr)

    f.close()

    # add additional attributes from the cmd-line, if any
    if args.add_attr:
        for attr in args.add_attr:
            wanted_attributes.append(attr)

    print("\nAttributes to be read:")
    for attr in wanted_attributes:
        print("    {}".format(attr) )
    print("")


    # now process input files
    n_processed = 0
    for hfile in args.input_files:
        print("Processing {}".format(hfile), end='\r')

        try: 
            f = h5py.File(hfile, 'r')
        except IOError:
            print("Cannot open %s, exiting." % hfile)
            sys.exit(1)

        current_time = f['time'][...]
        # write current time entry
        xdmfh.write('<Grid Name="particles" GridType="Uniform">\n')
        xdmfh.write('<Time Value="%s" />\n' % current_time[0])

        # write geometry info
        mylen = len(f['x'])
        xdmfh.write('<Topology TopologyType="Polyvertex" NodesPerElement="%s"></Topology>\n' % mylen)
        if args.dim == '3':
            xdmfh.write('<Geometry GeometryType="XYZ">\n')
            xdmfh.write('<DataItem DataType="Float" Precision="8" Dimensions="%s 3" Format="HDF">\n' % mylen)
        else:
            xdmfh.write('<Geometry GeometryType="XY">\n')
            xdmfh.write('<DataItem DataType="Float" Precisions="8" Dimensions="%s 2" Format="HDF">\n' % mylen)
        xdmfh.write('%s:/x\n' % hfile) 
        xdmfh.write('</DataItem>\n')
        xdmfh.write('</Geometry>\n')

        # velocities
        xdmfh.write('<Attribute AttributeType="Vector" Centor="Node" Name="velocity">\n')
        xdmfh.write('<DataItem DataType="Float" Dimensions="%s %s" Format="HDF">\n' % (mylen, args.dim))
        xdmfh.write('%s:/v\n' % hfile)
        xdmfh.write('</DataItem>\n')
        xdmfh.write('</Attribute>\n')

        # wanted_attributes
        for myattr in wanted_attributes:
            if myattr == 'deviatoric_stress':
                dim = int(args.dim)
                xdmfh.write('<Attribute AttributeType="Tensor" Centor="Node" Name="deviatoric_stress">\n')
                xdmfh.write('<DataItem DataType="Float" Dimensions="%s %s" Format="HDF">\n' % (mylen, str(dim*dim)))
                xdmfh.write('%s:/deviatoric_stress\n' % hfile)
                xdmfh.write('</DataItem>\n')
                xdmfh.write('</Attribute>\n')
            else:
                datatype = "Float"
                if myattr in [ 'noi', 'proc', 'key', 'number_of_interactions', 'material_type', 'number_of_activated_flaws', 'aneos_phase_flag']:
                    datatype = "Integer"
                xdmfh.write('<Attribute AttributeType="Scalar" Centor="Node" Name="%s">\n' % myattr)
                xdmfh.write('<DataItem DataType="%s" Dimensions="%s 1" Format="HDF">\n' % (datatype, mylen))
                xdmfh.write('%s:/%s\n' % (hfile, myattr))
                xdmfh.write('</DataItem>\n')
                xdmfh.write('</Attribute>\n')
        xdmfh.write('</Grid>\n')

        f.close()
        n_processed += 1

    # write footnote of xdmf file
    write_xdmf_footer(xdmfh)

    xdmfh.close()

    print("Done. Processed {} files.\n".format(n_processed) )

