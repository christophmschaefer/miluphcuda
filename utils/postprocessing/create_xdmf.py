#!/usr/bin/env python3

import h5py
import sys
import argparse



parser = argparse.ArgumentParser(description='Generate xdmf file from .h5 miluph output files for paraview postprocessing.\n Open the generated .xdmf file with paraview.')

parser.add_argument('--output', help='output file name', default='disk.xdmf')
parser.add_argument('--dim', help='dimension', default='3')
parser.add_argument('--input_files', nargs='+', help='input file names', default='none')

args = parser.parse_args()


if len(sys.argv) < 3:
    print("Try --help to get a usage.")
    sys.exit(0);


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


xdmfh = open(args.output, 'w')

# write header of xdmf file
write_xdmf_header(xdmfh)



# now process input files

# scalar floats only 
#wanted_attributes = ['rho', 'p', 'sml', 'e', 'material_type', 'number_of_interactions', 'fragments', 'DIM_root_of_damage']
#wanted_attributes = ['rho', 'p', 'sml', 'e', 'material_type', 'number_of_interactions', 'alpha_jutzi' ]
#wanted_attributes = ['rho', 'p', 'sml', 'e', 'material_type', 'number_of_interactions', 'fragments' ]
#wanted_attributes = ['rho', 'p', 'sml', 'e', 'material_type', 'number_of_interactions', 'deviatoric_stress' ]
wanted_attributes = ['rho', 'p', 'sml', 'e', 'material_type', 'number_of_interactions', 'deviatoric_stress',  'aneos_T', 'aneos_cs', 'aneos_entropy', 'aneos_phase_flag' ]

for hfile in args.input_files:
    print("Processing %s " % hfile)

    try: 
        f = h5py.File(hfile, 'r')

    except IOError:
        print("Cannot open %s, exiting." % hfile)
        sys.exit(1)


    current_time = f['time'][...]
    # write current time entry
    xdmfh.write('<Grid Name="particles" GridType="Uniform">\n')
    xdmfh.write('<Time Value="%s" />\n' % current_time[0])
    mylen = len(f['x'])
    xdmfh.write('<Topology TopologyType="Polyvertex" NodesPerElement="%s"></Topology>\n' % mylen)
    if args.dim == '3':
        xdmfh.write('<Geometry GeometryType="XYZ">\n')
        xdmfh.write('<DataItem DataType="Float" Dimensions="%s 3" Format="HDF">\n' % mylen)
    else:
        xdmfh.write('<Geometry GeometryType="XY">\n')
        xdmfh.write('<DataItem DataType="Float" Dimensions="%s 2" Format="HDF">\n' % mylen)
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
            if myattr in ['number_of_interactions', 'material_type']:
                datatype = "Integer"
            xdmfh.write('<Attribute AttributeType="Scalar" Centor="Node" Name="%s">\n' % myattr)
            xdmfh.write('<DataItem DataType="%s" Dimensions="%s 1" Format="HDF">\n' % (datatype, mylen))
            xdmfh.write('%s:/%s\n' % (hfile, myattr))
            xdmfh.write('</DataItem>\n')
            xdmfh.write('</Attribute>\n')
    xdmfh.write('</Grid>\n')
    f.close()

# write footnote of xdmf file
write_xdmf_footer(xdmfh)
