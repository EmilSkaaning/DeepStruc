import os, torch, h5py, random, sys, shutil, yaml
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
import pytorch_lightning as pl

seed = 37
torch.manual_seed(seed)
pl.seed_everything(seed)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class graph_loader(pl.LightningDataModule):
    def __init__(self, data_dir, cluster_size=None, num_files=None, batchsize=1, shuffle=True, num_workers=0):
        super(graph_loader, self).__init__()
        """

        Parameters
        ----------
        data_dir
        num_files
        batchsize
        shuffle

        Returns
        -------

        """
        self.batchsize = int(batchsize)
        self.num_workers = num_workers
        self.files_sorted = sorted(os.listdir(data_dir))
        self.cluster_size = cluster_size
        files = self.files_sorted.copy()
        # files = [file for file in files if 'FCC' in file]

        if shuffle == True:
            random.shuffle(files)
        if files != None:
            files = files[:num_files]
        else:
            pass

        nTrain = int(0.6 * len(files))
        nValid = int((len(files) - nTrain) / 2)
        nTest = len(files) - (nTrain + nValid)

        print('\nBatch size: {}'.format(batchsize))
        print('Total number of graphs {}.'.format(len(files)))
        print('\tTraining files:', nTrain)
        print('\tValidation files:', nValid)
        print('\tTest files:', nTest, '\n')

        self.trSamples, self.vlSamples, self.teSamples = list(), list(), list()
        print('Loading graphs:')

        for idx in range(len(files)):
            h5f = h5py.File(data_dir + '/' + files[idx], 'r')
            b = h5f['Node Feature Matrix'][:]
            h5f.close()

            if self.cluster_size == None:
                self.cluster_size = len(b)
            elif len(b) > self.cluster_size:
                self.cluster_size = len(b)

        largest_x_dist, largest_y_dist, largest_z_dist, edge_f_max = 0, 0, 0, 0
        for idx in range(nTrain):
            h5f = h5py.File(data_dir + '/' + files[idx], 'r')
            a = h5f['Edge Feature Matrix'][:]
            b = h5f['Node Feature Matrix'][:]

            h5f.close()

            diff_ph = abs(np.amin(b, axis=0)) + np.amax(b, axis=0)
            if largest_x_dist < diff_ph[0]:
                largest_x_dist = diff_ph[0]
            if largest_y_dist < diff_ph[1]:
                largest_y_dist = diff_ph[1]
            if largest_z_dist < diff_ph[2]:
                largest_z_dist = diff_ph[2]
            if np.amax(a) > edge_f_max:
                edge_f_max = np.amax(a)

        self.largest_x_dist = largest_x_dist
        self.largest_y_dist = largest_y_dist
        self.largest_z_dist = largest_z_dist

        for idx in tqdm(range(len(files))):
            h5f = h5py.File(data_dir + '/' + files[idx], 'r')
            a = h5f['Edge Feature Matrix'][:]  # todo: norm this
            b = h5f['Node Feature Matrix'][:]
            c = h5f['Edge Directions'][:]
            d = h5f['PDF label'][:]
            h5f.close()

            a /= edge_f_max
            min_vals = np.amin(b, axis=0)
            if min_vals[0] < 0.0:  # Make all coordinates positive
                b[:, 0] -= min_vals[0]
            if min_vals[1] < 0.0:  # Make all coordinates positive
                b[:, 1] -= min_vals[1]
            if min_vals[2] < 0.0:  # Make all coordinates positive
                b[:, 2] -= min_vals[2]

            b[:, 0] /= largest_x_dist
            b[:, 1] /= largest_y_dist
            b[:, 2] /= largest_z_dist

            cord_ph = np.zeros((self.cluster_size, np.shape(b)[1])) - 1
            cord_ph[:np.shape(b)[0]] = b

            d /= np.amax(d)  # Standardize PDF

            pdf = torch.tensor([d], dtype=torch.float)
            x = torch.tensor(b, dtype=torch.float)
            y = torch.tensor([cord_ph], dtype=torch.float)
            edge_index = torch.tensor(c, dtype=torch.long)
            edge_attr = torch.tensor(a, dtype=torch.float)
            name_idx = torch.tensor(self.files_sorted.index(files[idx]), dtype=torch.int16)

            if idx < nTrain:
                self.trSamples.append(
                    tuple((Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr), pdf.T, name_idx)))
            elif idx < nTrain + nValid:
                self.vlSamples.append(
                    tuple((Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr), pdf.T, name_idx)))
            else:
                self.teSamples.append(
                    tuple((Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr), pdf.T, name_idx)))

    def train_dataloader(self):
        return DataLoader(self.trSamples, batch_size=self.batchsize, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.vlSamples, batch_size=self.batchsize, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.teSamples, batch_size=self.batchsize, num_workers=self.num_workers)


def save_xyz_file(save_dir, cords, file_name, xyz_scale=[1,1,1]):

    cords = [xyz for xyz in cords if np.mean(xyz) >= -0.2]
    these_cords = []
    for count, xyz in enumerate(cords):
        if count == 0:
            these_cords.append(['{:d}'.format(len(cords))])
            these_cords.append([''])

        these_cords.append(['W   {:.4f}   {:.4f}   {:.4f}'.format(xyz[0]*xyz_scale[0], xyz[1]*xyz_scale[1], xyz[2]*xyz_scale[2])])

    np.savetxt(save_dir + '/{}.xyz'.format(file_name), these_cords, fmt='%s')
    return None


def folder_manager(input_dict, model_arch):
    this_trainer = None
    epoch = input_dict['epochs']
    if not os.path.isdir(input_dict['save_dir']):
        os.mkdir(input_dict['save_dir'])
        os.mkdir(input_dict['save_dir'] + '/models')
        shutil.copy2('train.py', input_dict['save_dir'] + '/train.py')
        shutil.copy2('./tools/data_loader.py', input_dict['save_dir'] + '/data_loader.py')
        shutil.copy2('./tools/module.py', input_dict['save_dir'] + '/module.py')
        os.mkdir(input_dict['save_dir'] + '/prior')
        os.mkdir(input_dict['save_dir'] + '/posterior')
    else:
        shutil.copy2('train.py', input_dict['save_dir'] + '/train.py')
        shutil.copy2('./tools/data_loader.py', input_dict['save_dir'] + '/data_loader.py')
        shutil.copy2('./tools/module.py', input_dict['save_dir'] + '/module.py')

    if input_dict['load_trainer']:
        best_model = sorted(os.listdir(input_dict['save_dir'] + '/models'))
        print(f'\nUsing {best_model[0]} as starting model!\n')
        this_trainer = input_dict['save_dir'] + '/models/' + best_model[0]
        #input_dict = yaml.load(f'{input_dict["save_dir"]}/input_dict.yaml', Loader=yaml.FullLoader)

        try:
            with open(f'{input_dict["save_dir"]}/input_dict.yaml') as file:
                input_dict = yaml.full_load(file)
            input_dict['load_trainer'] = True
            input_dict['epochs'] = epoch
            with open(f'{input_dict["save_dir"]}/model_arch.yaml') as file:
                model_arch = yaml.full_load(file)
        except FileNotFoundError:  # todo: transition - need to be deleted at some point
            with open(f'{input_dict["save_dir"]}/input_dict.yaml', 'w') as outfile:
                yaml.dump(input_dict, outfile, allow_unicode=True, default_flow_style=False)

            with open(f'{input_dict["save_dir"]}/model_arch.yaml', 'w') as outfile:
                yaml.dump(model_arch, outfile, allow_unicode=True, default_flow_style=False)
    else:
        with open(f'{input_dict["save_dir"]}/input_dict.yaml', 'w') as outfile:
            yaml.dump(input_dict, outfile, allow_unicode=True, default_flow_style=False)

        with open(f'{input_dict["save_dir"]}/model_arch.yaml', 'w') as outfile:
            yaml.dump(model_arch, outfile, allow_unicode=True, default_flow_style=False)
    return this_trainer, input_dict, model_arch


def get_callbacks(save_dir):
    checkpoint_callback_tot = ModelCheckpoint(
        monitor='vld_tot',
        dirpath=save_dir + '/models',
        filename='model-{vld_tot:.5f}-{beta:.3f}-{vld_rec_pdf:.5f}-{epoch:010d}',
        save_top_k=5,
        mode='min',
        save_last=True,
    )

    checkpoint_callback_rec = ModelCheckpoint(
        monitor='vld_rec',
        dirpath=save_dir + '/models',
        filename='model-{vld_rec:.5f}-{beta:.3f}-{vld_rec_pdf:.5f}-{vld_tot:.5f}-{epoch:010d}',
        save_top_k=5,
        mode='min',
    )

    checkpoint_callback_kld = ModelCheckpoint(
        monitor='vld_kld',
        dirpath=save_dir + '/models',
        filename='model-{vld_kld:.5f}-{beta:.3f}-{vld_rec_pdf:.5f}-{vld_tot:.5f}-{epoch:010d}',
        save_top_k=5,
        mode='min',
    )

    checkpoint_callback_vld_rec_pdf = ModelCheckpoint(
        monitor='vld_rec_pdf',
        dirpath=save_dir + '/models',
        filename='model-{vld_rec_pdf:.5f}-{beta:.3f}-{vld_tot:.5f}-{epoch:010d}',
        save_top_k=5,
        mode='min',
    )

    return [checkpoint_callback_tot, checkpoint_callback_rec, checkpoint_callback_kld,
            checkpoint_callback_vld_rec_pdf]