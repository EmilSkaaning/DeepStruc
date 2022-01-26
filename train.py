import yaml, torch, random, time, argparse
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

_BANNER = """
Train your own DeepStruc model.
"""

parser = argparse.ArgumentParser(description=_BANNER, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-d", "--data_dir", default='./data/graphs', type=str,
                    help="Directory containing graph data.")

parser.add_argument("-s", "--save_dir", default='test', type=str,
                    help="Directory where models will be saved. This is also used for loading a learner.")

parser.add_argument("-r", "--resume_model", default=False, type=bool,
                    help="If 'True' the save_dir model is loaded and training is continued.")

parser.add_argument("-e", "--epochs", default=2, type=int,
                    help="Number of maximum epochs.")

parser.add_argument("-b", "--batch_size", default=2, type=int,
                    help="Size of batch.")

parser.add_argument("-l", "--learning_rate", default=1e-3, type=float,
                    help="Learning rate.")

parser.add_argument("-B", "--beta", default=0, type=float,
                    help="Initial beta value for scaling KLD.")

parser.add_argument("-i", "--beta_increase", default=0.001, type=float,
                    help="Increments of beta when the threshold is met.")

parser.add_argument("-x", "--beta_max", default=1, type=float,
                    help="Highst value beta can increase to.")

parser.add_argument("-t", "--reconstruction_th", default=0.0001, type=float,
                    help="Reconstruction threshold required before beta is increased.")

parser.add_argument("-n", "--num_files", default=None, type=int,
                    help="Total number of files loaded. Files will be split 60/20/20. If 'None' then all files are loaded.")

parser.add_argument("-c", "--compute", default='cpu', type=str, choices=['cpu', 'gpu16', 'gpu32', 'gpu64'],
                    help="Train model on CPU or GPU. Choices: 'cpu', 'gpu16', 'gpu32' and 'gpu64'.")

parser.add_argument("-L", "--latent_dim", default=2, type=int,
                    help="Number of latent space dimensions.")

if __name__ == '__main__':
    start_time = time.time()
    args = parser.parse_args()

    input_dict = {
        'data_dir': args.data_dir,
        'save_dir': f'./models/{args.save_dir}',
        'epochs': args.epochs,
        'batchsize': args.batch_size,
        'n_files': args.num_files,
        'load_trainer': args.resume_model,  # Todo: something with this

        'model': {
            'lr': args.learning_rate,
            'beta': args.beta,
            'beta_inc': args.beta_increase,
            'beta_max': args.beta_max,
            'rec_th': args.reconstruction_th
        },
    }

    graph_data = graph_loader(input_dict['data_dir'], batchsize=input_dict['batchsize'], num_files=input_dict['n_files'])
    input_dict['cluster_size'] = graph_data.cluster_size

    model_arch = {  # Defines the architecture of the network
              'latent_space': args.latent_dim,
              'PDF_len': np.shape(graph_data.train_dataloader().dataset[0][1])[0],
              'node_features': graph_data.train_dataloader().dataset[0][0].num_node_features,
              'norm_vals': {
                  'x':float(graph_data.largest_x_dist),
                  'y':float(graph_data.largest_y_dist),
                  'z':float(graph_data.largest_z_dist),
              },

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


    # Make save dir and trainer
    checkpoint_list = get_callbacks(input_dict['save_dir'])
    this_trainer, input_dict, model_arch = folder_manager(input_dict, model_arch)
    tb_logger = pl.loggers.TensorBoardLogger(input_dict['save_dir'])


    # Init our model
    model = Net(model_arch=model_arch, **input_dict['model'])
    print(model)

    if args.compute == 'cpu':  # Define where to cast model
        trainer = pl.Trainer(accelerator='cpu', num_processes=1,checkpoint_callback=True, max_epochs=input_dict['epochs'],
                             progress_bar_refresh_rate=1,logger=tb_logger,callbacks=checkpoint_list,
                             resume_from_checkpoint=this_trainer)
    else:
        precision = int(args.compute[-2:])
        trainer = pl.Trainer(gpus=1, precision=precision, num_processes=1, checkpoint_callback=True,
                             max_epochs=input_dict['epochs'],
                             progress_bar_refresh_rate=1, logger=tb_logger, callbacks=checkpoint_list,
                             resume_from_checkpoint=this_trainer)

    trainer.fit(model, graph_data)
    trainer.test(ckpt_path='best')

    end_time = time.time()
    print(f'took: {(end_time-start_time)/60:.2f} min')
    # Updating Beta
    input_dict['beta'] = model.beta
    with open(f'{input_dict["save_dir"]}/input_dict.yaml', 'w') as outfile:
        yaml.dump(input_dict, outfile, allow_unicode=True, default_flow_style=False)
