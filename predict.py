import yaml, sys, os, argparse
import torch, random
import numpy as np
import pytorch_lightning as pl
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
"""
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()
path, model_name = filename.split('/models', 1)
epochs = model_name.split('epoch=')
max_epochs = int(epochs[1][:-5]) + 1
sys.path.append(path)"""
from tools.data_loader import graph_loader, save_xyz_file
from tools.module import Net

seed = 37
torch.manual_seed(seed)
pl.seed_everything(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def get_data(this_path, samples):
    if os.path.isdir(this_path):
        files = sorted(os.listdir(this_path))
    else:
        files = [this_path]
        this_path = '.'

    x_list, y_list, name_list = [], [], []
    idxx = 0
    np_data = np.zeros((len(files)*samples, 2800))
    for idx, file in enumerate(files):
        for skip_row in range(100):
            try:
                data = np.loadtxt(f'{this_path}/{file}', skiprows=skip_row)
            except ValueError:
                continue
            data = data.T
            x_list.append(data[0])
            y_list.append(data[1])
            for i in range(samples):
                if len(data[1]) < 3001:
                    # np_data[idx][:len(data[1])] = data[1]
                    np_data[idxx][200:len(data[1])] = data[1][200:len(data[1])]
                else:
                    np_data[idxx] = data[1][200:3000]
                np_data[idxx] /= np.amax(np_data[idxx])
                idxx += 1
                name_list.append(file)
            break

    np_data = np_data.reshape((len(files)*samples, 2800, 1))
    np_data = torch.tensor(np_data, dtype=torch.float)
    return np_data, name_list

def make_prediction_dirs(path, names):
    for name in names:
        if not os.path.isdir(f'{path}/real_data/{name[:-3]}'):
            os.mkdir(f'{path}/real_data/{name[:-3]}')
    return None

_BANNER = """
Predict data
"""

parser = argparse.ArgumentParser(description=_BANNER, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-d", "--data", default='./data/experimental_PDFs', type=str,
                    help="Path to data or data directory. If pointing to data directory all datasets must have same format.")

parser.add_argument("-m", "--model", default='./models/DeepStruc', type=str,
                    help="Path to model. If 'None' GUI will open.")  # todo: implement TKinter GUI

parser.add_argument("-n", "--num_samples", default=5, type=int,
                    help="Number of samples/structures generated.")

parser.add_argument("-s", "--sigma", default=1, type=float,
                    help="Multiplier of the normaldistributions sigma.")

parser.add_argument("-p", "--plot_sampling", default=False, type=bool,  # todo: implement
                    help="Plots sampled structures ontop of DeepStruc training data. Model must be DeepStruc.")

if __name__=='__main__':
    args = parser.parse_args()

    data, data_names=get_data(args.data, args.num_samples)

    with open(f'{args.model}/model_arch.yaml') as file:
        model_arch = yaml.full_load(file)

    Net(model_arch=model_arch)
    DeepStruc = Net.load_from_checkpoint(f'{args.model}/model-vld_rec_pdf=0.01222-beta=0.004-vld_kld=6.17422-epoch=0000013845.ckpt',model_arch=model_arch)
    xyz_pred, latent_space, kl = DeepStruc(data, mode='prior', sigma_scale=args.sigma)

    sys.exit()

    for idx, (xyz, my_name) in enumerate(zip(xyz_pred, real_data_names)):  # todo: save predictions
        ls = latent_space[idx].detach().cpu().numpy()

        save_xyz_file(f'{path}/real_data_2', xyz.detach().cpu().numpy(),
                      f'test_epoch_{max_epochs:09}_{my_name}_{num_runs}_ls_{ls[0]}_{ls[1]}',
                      [graph_data.largest_x_dist, graph_data.largest_y_dist, graph_data.largest_z_dist])







