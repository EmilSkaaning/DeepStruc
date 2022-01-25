from utils.gen_xyz import gen_xyz
from utils.remove_duplicates import remove_duplicates
from utils.gen_graphs import gen_graphs

def main(atom_list, type_list):
    print('\nCreating xyz monometallic nanoparticles!')
    data_base = gen_xyz(atom_list, type_list)

    print('\nCloning unique monometallic nanoparticles!')
    remove_duplicates(data_base)

    print('\nCreating graphs of monometallic nanoparticles!')
    gen_graphs(data_base)



if __name__ == '__main__':
    # Gen_xyz inputs
    atom_list = ['Sc',
                 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                 'Y', 'Zr', 'Nb', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
                 ]
    type_list = ["SC",
                 "FCC", "BCC", "HCP", "Icosahedron", "Decahedron", "Octahedron"
                 ]
    max_atoms = 200
    interpolate = 1
    direct = "strus_atoms_{:03d}_interpolate_{:03d}/xyz_db_raw_/".format(max_atoms, interpolate)
    main(atom_list, type_list)