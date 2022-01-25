import os,yaml
from diffpy.Structure import loadStructure, Lattice
import numpy as np
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py

class graph_maker_xyz_nn():
    def __init__(self, path, save_dir, pdf_dict):
        self.stru_dir = path
        self.save_dir = save_dir
        self.strus = sorted(os.listdir(path))
        print('Number of structures:', len(self.strus))

        print('PDF simulation parameters:')
        for key in pdf_dict.keys():
            print(f'{key} = {pdf_dict[key]:.2f}')

        # Set parameters
        self.qmin = pdf_dict['qmin']
        self.qmax = pdf_dict['qmax']
        self.rmin = pdf_dict['rmin']
        self.rmax = pdf_dict['rmax']
        self.rstep = pdf_dict['rstep']
        self.biso = pdf_dict['biso']

    def setup_calculator(self):
        dbc = DebyePDFCalculator()
        dbc.qmin = self.qmin
        dbc.qmax = self.qmax
        dbc.rmin = self.rmin
        dbc.rmax = self.rmax
        dbc.rstep = self.rstep

        return dbc

    def gen_graphs(self):
        dbc = self.setup_calculator()

        pbar = tqdm(total=len(self.strus))

        for stru in self.strus:
            if os.path.isfile(self.save_dir+'/'+'graph_{}.h5'.format(stru[:-4])):
                pbar.update()
                continue
            lc = self.read_lc(stru)
            lc *= 1.02
            cluster = loadStructure(self.stru_dir + '/' + stru)

            cluster.B11 = self.biso
            cluster.B22 = self.biso
            cluster.B33 = self.biso
            cluster.B12 = 0
            cluster.B13 = 0
            cluster.B23 = 0

            dbc.setStructure(cluster)
            r, g0 = dbc()

            # Todo: make this a function
            #dist_mat = np.zeros((len(cluster),len(cluster)))  # Edge info

            node_mat = []  # For angles Todo: add atom label
            index_list1 = []
            index_list2 = []
            edge_f_list = []
            for i in range(len(cluster)):

                for j in range(i,len(cluster)):
                    if i == j: continue
                        #dist_mat[i][j] = 99  # Todo: this is not a pretty solution
                    else:
                        position_0 = np.array([cluster.x[i], cluster.y[i], cluster.z[i]])
                        position_1 = np.array([cluster.x[j], cluster.y[j], cluster.z[j]])

                        dist = calc_dist(position_0, position_1)
                        #dist_mat[i][j] = dist
                        #dist_mat[j][i] = dist

                    if dist <= lc:
                        index_list1.append(i)
                        index_list2.append(j)
                        edge_f_list.append(dist)
                        index_list1.append(j)
                        index_list2.append(i)
                        edge_f_list.append(dist)



                node_mat.append([cluster.x[i], cluster.y[i], cluster.z[i]])

            #dist_mat = np.array(#dist_mat)
            #np.fill_diagonal(#dist_mat, 0)
            edge_f_list = np.array(edge_f_list)

            node_mat = np.array(node_mat)
            direction = np.array([index_list1, index_list2])
            g0 = np.array(g0)

            h5_file = h5py.File(self.save_dir+'/'+'graph_{}.h5'.format(stru[:-4]), 'w')
            h5_file.create_dataset('Edge Feature Matrix', data=edge_f_list)
            h5_file.create_dataset('Node Feature Matrix', data=node_mat)
            h5_file.create_dataset('Edge Directions', data=direction)
            h5_file.create_dataset('PDF label', data=g0)
            h5_file.close()

            pbar.update()
        pbar.close
        return None

    def read_lc(self, str_name):
        if 'lc1' in str_name:
            lc1_idx = str_name.index('lc1_') + 4
            lc2_idx = str_name.index('lc2_') - 1
            bd = float(str_name[lc1_idx:lc2_idx])
        else:
            lc_idx = str_name.index('lc_') + 3
            bd = float(str_name[lc_idx:-4])

        return bd



def calc_dist(position_0, position_1):
    """ Returns the distance between vectors position_0 and position_1 """
    return np.sqrt((position_0[0] - position_1[0]) ** 2 + (position_0[1] - position_1[1]) ** 2 + (
            position_0[2] - position_1[2]) ** 2)

def gen_graphs(data_base, pdf_dict):
    try:
        os.makedirs(f'{data_base}/graphs')
    except FileExistsError:
        pass

    with open(f'{data_base}/PDF_simulation_parameters.yaml', 'w') as outfile:
        yaml.dump(pdf_dict, outfile, allow_unicode=True, default_flow_style=False)

    obj = graph_maker_xyz_nn(f'{data_base}/xyz_unique', f'{data_base}/graphs', pdf_dict)
    obj.gen_graphs()


