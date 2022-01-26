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

def get_data(this_dir, data_type=None):
    files = sorted(os.listdir(this_dir))

    if data_type != None:
        files = [file for file in files if file[-len(data_type):]==data_type]

    x_list, y_list, = [], []
    np_data = np.zeros((len(files), 3000))

    for idx, file in enumerate(files):
        for skip_row in range(100):
            try:
                data = np.loadtxt(f'{this_dir}/{file}', skiprows=skip_row)
            except ValueError:
                continue

            data = data.T
            x_list.append(data[0])
            y_list.append(data[1])
            if len(data[1]) < 3001:
                np_data[idx][:len(data[1])] = data[1]
            else:
                np_data[idx] = data[1][:3000]
            np_data[idx] /= np.amax(np_data[idx])

            break

    np_data = np_data.reshape((len(files), 3000, 1))
    np_data = torch.tensor(np_data, dtype=torch.float)

    return np_data, files

def make_prediction_dirs(path, names):
    for name in names:
        if not os.path.isdir(f'{path}/real_data/{name[:-3]}'):
            os.mkdir(f'{path}/real_data/{name[:-3]}')
    return None

# Inputs parameters.
# model
# num samples
# sigma
# plot

_BANNER = """
Predict data
"""

parser = argparse.ArgumentParser(description=_BANNER, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-d", "--data", default='./data/experimental_PDFs', type=str,
                    help="Path to data or data directory. If pointing to data directory all datasets must have same format.")

parser.add_argument("-m", "--model", default='./models/DeepStruc/model-vld_rec_pdf=0.01222-beta=0.004-vld_kld=6.17422-epoch=0000013845.ckpt', type=str,
                    help="Path to model. If 'None' GUI will open.")

parser.add_argument("-n", "--num_samples", default=5, type=int,
                    help="Number of samples/structures generated.")

parser.add_argument("-s", "--sigma", default=1, type=float,
                    help="Multiplier of the normaldistributions sigma.")

parser.add_argument("-p", "--plot_sampling", default=1, type=bool,
                    help="Plots sampled structures ontop of DeepStruc training data. Models must be DeepStruc")

if __name__=='__main__':
    args = parser.parse_args()

    model = Net().load_from_checkpoint(args.model)

    trainer = pl.Trainer(resume_from_checkpoint=args.model)



    sys.exit()
    print(f'Model chosen: {path}')
    print(f'Imported file directory: {model_name}')
    data_dir = 'D:/Work/PhD/Articles/CVAE_paper/real_data_predictions/data'
    real_data, real_data_names = get_data(data_dir)
    real_data = real_data[0]
    real_data_names = real_data_names[0]
    make_prediction_dirs(path, real_data_names)

    save_xyz = True
    num_samples = 100
    # Loading dictionaries
    with open(f'{path}/input_dict.yaml') as file:
        input_dict = yaml.full_load(file)

    with open(f'{path}/model_arch.yaml') as file:
        model_arch = yaml.full_load(file)

    # Manipulate dictionaries
    input_dict['data_dir'] = 'D:/machine_learning/Mono_metals_db/xyz_db_raw_atoms_200_interpolate_001/xyz_db_raw_atoms_200_interpolate_001_03Biso' # stacking_fault_graphs

    # Load data and init model
    graph_data = graph_loader(input_dict['data_dir'], cluster_size=input_dict['cluster_size'], batchsize=1,
                              num_files=input_dict['n_files'])
    init_data = graph_loader(input_dict['data_dir'], cluster_size=input_dict['cluster_size'], batchsize=1,
                              num_files=3)

    init_batch = next(iter(graph_data.train_dataloader()))

    model = Net(graph_data.train_dataloader().dataset[0], model_arch, beta=input_dict['beta'], lr=0)
    trainer = pl.Trainer(max_epochs=max_epochs,
                         resume_from_checkpoint=filename)

    trainer.fit(model, init_data,)
    trainer.test(ckpt_path=filename)

    model.eval()
    from tqdm import tqdm
    pbar = tqdm(total=num_samples)
    for num_runs in range(num_samples):
        xyz_pred, latent_space, kl = model(real_data, mode='prior')

        for idx, (xyz, my_name) in enumerate(zip(xyz_pred, real_data_names)):
            ls = latent_space[idx].detach().cpu().numpy()

            save_xyz_file(f'{path}/real_data_2', xyz.detach().cpu().numpy(),
                          f'test_epoch_{max_epochs:09}_{my_name}_{num_runs}_ls_{ls[0]}_{ls[1]}',
                          [graph_data.largest_x_dist, graph_data.largest_y_dist, graph_data.largest_z_dist])
        pbar.update()
    sys.exit()






