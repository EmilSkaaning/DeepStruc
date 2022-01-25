import os, sys, yaml, torch, time, shutil, random, time
import pytorch_lightning as pl
from tools.data_loader import graph_loader, folder_manager,get_callbacks
from tools.module import Net
import numpy as np

seed = 37
torch.manual_seed(seed)
pl.seed_everything(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    start_time = time.time()
    input_dict = {
        'epochs': 2,
        'data_dir': './data/strus_atoms_100_interpolate_001/graphs',
        'save_dir': './models/test',
        'batchsize': 2,
        'model': {
            'lr': 5e-5,
            'dropout_p': 0.2,

            'beta': 0.0,
            'beta_inc': 0.001,
            'beta_max': 100,

            'rec_th': 0.0001
        },
        'cluster_size': None,
        'n_files': None,
        'load_trainer': False,
    }

    graph_data = graph_loader(input_dict['data_dir'], cluster_size=input_dict['cluster_size'], batchsize=input_dict['batchsize'], num_files=input_dict['n_files'])
    input_dict['cluster_size'] = graph_data.cluster_size

    model_arch = {
                  # Latent-space dim
                  'latent_space': 2,

                  'encoder':{
                      'e0': 256 * 4,
                      'e1': 128 * 4,
                      'e2': 64 * 4,
                      'e3': 32 * 4,
                      'e4': 16 * 4,
                  },

                  'decoder':{
                      'd0': 8 * 4,
                      'd1': 16 * 4,
                      'd2': 32 * 4,
                      'd3': 64 * 4,
                      'd4': 128 * 4,
                      'd5': 256 * 4,
                      'out_dim': input_dict['cluster_size'],
                  },

                  'mlps':{
                      'm0': 64 * 4,
                      'm1': 32 * 4,
                      'm2': 16 * 4,
                  },

                  'prior':{
                      'prior_0': 24*16,
                      'prior_1': 24*8,
                      'prior_2': 24,
                  },

                  'posterior':{
                      'prior_0': 24*16,
                      'prior_1': 24*8,
                      'prior_2': 24,
                  }
    }


    checkpoint_list = get_callbacks(input_dict['save_dir'])

    # Make save dir and trainer
    this_trainer, input_dict, model_arch = folder_manager(input_dict, model_arch)
    tb_logger = pl.loggers.TensorBoardLogger(input_dict['save_dir'])


    # Init our model
    init_batch = next(iter(graph_data.train_dataloader()))
    model = Net(graph_data.train_dataloader().dataset[0], model_arch, **input_dict['model'])

    print(model)

    #gpus=1, precision=16, profiler=True,
    trainer = pl.Trainer(accelerator='cpu', num_processes=2,checkpoint_callback=True, max_epochs=input_dict['epochs'],
                         progress_bar_refresh_rate=1,logger=tb_logger,callbacks=checkpoint_list,
                         resume_from_checkpoint=this_trainer)

    trainer.fit(model, graph_data)
    trainer.test(ckpt_path='best')

    end_time = time.time()
    print(f'took: {(end_time-start_time)/60:.2f} min')
    # Updating Beta
    input_dict['beta'] = model.beta
    with open(f'{input_dict["save_dir"]}/input_dict.yaml', 'w') as outfile:
        yaml.dump(input_dict, outfile, allow_unicode=True, default_flow_style=False)
