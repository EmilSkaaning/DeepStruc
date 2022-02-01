import sys, argparse
from tools.module import Net
import torch, random
import numpy as np
import pytorch_lightning as pl
from tools.utils import get_data, format_predictions, plot_ls, get_model, save_predictions

seed = 37
torch.manual_seed(seed)
pl.seed_everything(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def main(args):
    data, data_name, project_name = get_data(args)
    model_path, model_arch = get_model(args.model)

    Net(model_arch=model_arch)
    DeepStruc = Net.load_from_checkpoint(model_path,model_arch=model_arch)
    xyz_pred, latent_space, kl, mu, sigma = DeepStruc(data, mode='prior', sigma_scale=args.sigma)
    samling_pairs = format_predictions(latent_space, data_name, mu, sigma, args.sigma)

    if args.plot_sampling == True and args.model == 'DeepStruc':
        plot_ls(samling_pairs, project_name)
    elif args.plot_sampling == True and args.model != 'DeepStruc':
        print("Argument '--model' needs to be default DeepStruc value for plot to be generated!")

    save_predictions(xyz_pred, samling_pairs, project_name, model_arch, args)

    return None

_BANNER = """
Predict data.
"""

parser = argparse.ArgumentParser(description=_BANNER, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-d", "--data", default='./data/experimental_PDFs', type=str,
                    help="Path to data or data directory. If pointing to data directory all datasets must have same format.")

parser.add_argument("-m", "--model", default='DeepStruc', type=str,
                    help="Path to model. 'DeepStruc' to use pretrained model.")

parser.add_argument("-n", "--num_samples", default=5, type=int,
                    help="Number of samples/structures generated.")

parser.add_argument("-s", "--sigma", default=3, type=float,
                    help="Multiplier of the normaldistributions sigma.")

parser.add_argument("-p", "--plot_sampling", default=False, type=bool,
                    help="Plots sampled structures ontop of DeepStruc training data. Model must be DeepStruc.")

parser.add_argument("-g", "--save_path", default='.', type=str,  # todo: add in README
                    help="Save predictions path.")



if __name__=='__main__':
    args = parser.parse_args()
    main(args)










