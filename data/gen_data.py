from utils.gen_xyz import gen_xyz
from utils.remove_duplicates import remove_duplicates
from utils.gen_graphs import gen_graphs
import argparse, sys

def main(args=None):
    print('\nCreating xyz monometallic nanoparticles!')
    data_base = gen_xyz(args.atoms, args.structure_type, args.num_atoms, args.interpolation, args.directory)

    print('\nCloning unique monometallic nanoparticles!')
    remove_duplicates(data_base)

    pdf_dict = {
        'qmin': args.qmin,
        'qmax': args.qmax,
        'rmin': args.rmin,
        'rmax': args.rmax,
        'rstep': args.rstep,
        'biso': args.biso
    }
    print('\nCreating graphs of monometallic nanoparticles!')
    gen_graphs(data_base, pdf_dict)

_BANNER = """
Generating structures, graphs and conditional PDFs
for DeepStruc.
"""

parser = argparse.ArgumentParser(description=_BANNER, formatter_class=argparse.RawTextHelpFormatter)

# Generel directory to save data.
parser.add_argument("-d", "--directory", default=None, type=str,
                    help="Name or path of output as str")

# Structure parameters
atom_list = [
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
            'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Tc',
            'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta',
            'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
            ]
parser.add_argument("-a", "--atoms", default=atom_list, nargs='+',
                    help="A list of atoms e.g. '-a W Nb Mo'")

stru_type = [
            "SC", "FCC", "BCC", "HCP", "Ico", "Dec", "Oct"
            ]
parser.add_argument("-t", "--structure_type", default=stru_type, nargs='+',
                    help="A list of structure types e.g. '-t FCC BCC HCP'",
                    choices=["SC", "FCC", "BCC", "HCP", "Ico", "Dec", "Oct"])

parser.add_argument("-n", "--num_atoms", default=100, type=int,
                    help="Positive integer of maximum number of atoms in structures")

parser.add_argument("-i", "--interpolation", default=1, type=int,
                    help="Positive integer of interpolations per structure")

# PDF parameters
parser.add_argument("-q", "--qmin", default=0.7, type=float,
                    help="Smallest scattering amplitude, float")

parser.add_argument("-Q", "--qmax", default=25., type=float,
                    help="Largest scattering amplitude, float")

parser.add_argument("-r", "--rmin", default=2., type=float,
                    help="Smallest grid value, float")

parser.add_argument("-R", "--rmax", default=30., type=float,
                    help="Largest grid value, float")

parser.add_argument("-rs", "--rstep", default=0.01, type=float,
                    help="Grid step size, float")

parser.add_argument("-b", "--biso", default=0.3, type=float,
                    help="isotropic Atomic Displacement Parameter (ADP), float")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)