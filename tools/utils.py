import torch, random, os, yaml
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import datetime
from tools.data_loader import save_xyz_file

seed = 37
torch.manual_seed(seed)
pl.seed_everything(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def get_data(args):  # Todo: write your own dataloader.
    ct = str(datetime.datetime.now()).replace(' ', '_').replace(':','-').replace('.','-')
    project_name = f'{args.save_path}/DeepStruc_{ct}'
    print(f'\nProject name is: {project_name}')

    this_path = args.data
    samples = args.num_samples
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
            Gr_ph = data[1]
            if round(data[0][1] - data[0][0],2) != 0.01:
                raise ValueError("The PDF does not have an r-step of 0.01 Å")
            try:
                start_PDF = np.where((data[0] > 1.995) & (data[0] < 2.005))[0][0]
            except:
                Gr_ph = np.concatenate((np.zeros((int((data[0][0])/0.01))), Gr_ph))
                print("The PDFs first value is above 2 Å. We have added 0's down to 2 Å as a quick fix.")
            try:
                end_PDF = np.where((data[0] > 29.995) & (data[0] < 30.005))[0][0]
            except:
                Gr_ph = np.concatenate((Gr_ph, np.zeros((3000-len(Gr_ph)))))
                print("The PDFs last value is before 30 Å. We have added 0's up to 30 Å as a quick fix.")
            Gr_ph = Gr_ph[200:3000]
            for i in range(samples):
                np_data[idxx] = Gr_ph
                
                np_data[idxx] /= np.amax(np_data[idxx])
                idxx += 1
                name_list.append(file)
            break



    np_data = np_data.reshape((len(files)*samples, 2800, 1))
    np_data = torch.tensor(np_data, dtype=torch.float)

    return np_data, name_list, project_name


def format_predictions(latent_space, data_names, mus, sigmas, sigma_inc):
    df_preds = pd.DataFrame(columns=['x', 'y', 'file_name', 'mu', 'sigma', 'sigma_inc'])
    for i,j, mu, sigma in zip(latent_space, data_names, mus, sigmas):
        if '/' in j:
            j = j.split('/')[-1]

        if '.' in j:
            j_idx = j.rindex('.')
            j = j[:j_idx]


        info_dict = {
            'x': i[0].detach().cpu().numpy(),
            'y': i[1].detach().cpu().numpy(),
            'file_name': j,
            'mu': mu.detach().cpu().numpy(),
            'sigma': sigma.detach().cpu().numpy(),
            'sigma_inc': sigma_inc,

        }
        df_preds = df_preds.append(info_dict, ignore_index=True)
    return df_preds


def plot_ls(df, mk_dir):
    if not os.path.isdir(mk_dir):
        os.mkdir(mk_dir)
    ideal_ls = './tools/ls_points.csv'
    color_dict = {
        'FCC': '#19ADFF',
        'BCC': '#4F8F00',
        'SC': '#941100',
        'Octahedron': '#212121',
        'Icosahedron': '#005493',
        'Decahedron': '#FF950E',
        'HCP': '#FF8AD8',
    }
    df_ideal = pd.read_csv(ideal_ls, index_col=0)  # Get latent space data
    # Plotting inputs
    ## Training and validation data
    MARKER_SIZE_TR = 60
    EDGE_LINEWIDTH_TR = 0.0
    ALPHA_TR = 0.3

    ## Figure
    FIG_SIZE = (10, 4)
    MARKER_SIZE_FG = 60
    MARKER_FONT_SIZE = 10
    MARKER_SCALE = 1.5

    fig = plt.figure(figsize=FIG_SIZE)
    gs = GridSpec(1, 5, figure=fig)
    ax = fig.add_subplot(gs[0, :4])
    ax_legend = fig.add_subplot(gs[0, 4])
    print('\nPlotting DeepStruc training + validation data.')
    pbar = tqdm(total=len(df_ideal))
    for idx in range(len(df_ideal)):
        ax.scatter(df_ideal.iloc[idx]['x'], df_ideal.iloc[idx]['y'],
                   c=color_dict[df_ideal.iloc[idx]['stru_type']], s=MARKER_SIZE_TR * df_ideal.iloc[idx]['size'],
                   edgecolors='k', linewidth=EDGE_LINEWIDTH_TR,
                   alpha=ALPHA_TR)

        pbar.update()
    pbar.close()

    mlines_list = []
    for key in color_dict.keys():
        mlines_list.append(
            mlines.Line2D([], [], MARKER_SIZE_FG, marker='o', c=color_dict[key], linestyle='None', label=key,
                          mew=1))

    from matplotlib import cm
    cm_subsection = np.linspace(0, 1, len(df.file_name.unique()))
    data_color = [cm.magma(x) for x in cm_subsection]

    print('\nPlotting DeepStruc structure sampling.')
    pbar = tqdm(total=len(df.file_name.unique()))
    for idx, file_name in enumerate(df.file_name.unique()):
        this_c = np.array([data_color[idx]])

        df_ph = df[df.file_name==file_name]
        df_ph.reset_index(drop=True, inplace=True)

        ax.scatter(df_ph['mu'][0][0],df_ph['mu'][0][1], c=this_c, s=10, edgecolors='k',
                   linewidth=0.5, marker='D',zorder=1)
        ellipse = Ellipse((df_ph['mu'][0][0],df_ph['mu'][0][1]),df_ph['sigma'][0][0],df_ph['sigma'][0][1], ec='k', fc=this_c, alpha=0.5, fill=True, zorder=-1)
        ax.add_patch(ellipse)

        ellipse = Ellipse((df_ph['mu'][0][0],df_ph['mu'][0][1]),df_ph['x'].var(),df_ph['y'].var(), ec='k', fc=this_c, alpha=0.2, fill=True, zorder=-1)
        ax.add_patch(ellipse)

        mlines_list.append(
            mlines.Line2D([], [], MARKER_SIZE_FG, marker='D', c=this_c, linestyle='None', label=file_name, mec='k',
                          mew=1))

        for index, sample in df_ph.iterrows():
            ax.scatter(sample['x'], sample['y'], c=this_c, s=10, edgecolors='k',
                       linewidth=0.8, marker='o', zorder=2)
        pbar.update()
    pbar.close()
    ax_legend.legend(handles=mlines_list,fancybox=True, #ncol=2,  #, bbox_to_anchor=(0.8, 0.5)
          markerscale=MARKER_SCALE, fontsize=MARKER_FONT_SIZE, loc='center')

    ax.set_xlabel('Latent space x', size=10)  # Latent Space Feature 1
    ax.set_ylabel('Latent space y', size=10)

    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.get_xaxis().set_ticks([])
    ax_legend.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.tight_layout()
    plt.savefig(f'{mk_dir}/ls.png',dpi=300)
    return None


def get_model(model_dir):
    if model_dir == 'DeepStruc':
        with open(f'./models/DeepStruc/model_arch.yaml') as file:
            model_arch = yaml.full_load(file)
        model_path = './models/DeepStruc/models/DeepStruc.ckpt'
        return model_path, model_arch
    if os.path.isdir(model_dir):
        if 'models' in os.listdir(model_dir):
            models = sorted(os.listdir(f'{model_dir}/models'))
            models = [model for model in models if '.ckpt' in model]
            print(f'No specific model was provided. {models[0]} was chosen.')
            print('Dataloader might not be sufficient in loading dimensions.')
            model_path = f'{model_dir}/models/{models[0]}'
            with open(f'{model_dir}/model_arch.yaml') as file:
                model_arch = yaml.full_load(file)

            return model_path, model_arch
        else:
            print(f'Path not understood: {model_dir}')
    else:
        idx = model_dir.rindex('/')
        with open(f'{model_dir[:idx-6]}model_arch.yaml') as file:
            model_arch = yaml.full_load(file)

        return model_dir, model_arch


def save_predictions(xyz_pred, df, project_name, model_arch, args):
    print('\nSaving predicted structures as XYZ files.')
    if not os.path.isdir(f'{project_name}'):
        os.mkdir(f'{project_name}')

    with open(f'{project_name}/args.yaml', 'w') as outfile:
        yaml.dump(vars(args), outfile, allow_unicode=True, default_flow_style=False)

    pbar = tqdm(total=len(df))
    for idx, row in df.iterrows():
        if not os.path.isdir(f'{project_name}/{row["file_name"]}'):
            os.mkdir(f'{project_name}/{row["file_name"]}')
        x = f'{float(row["x"]):+.3f}'.replace('.', '-')
        y = f'{float(row["y"]):+.3f}'.replace('.', '-')

        save_xyz_file(f'{project_name}/{row["file_name"]}',
                      xyz_pred[idx].detach().cpu().numpy(),
                      f'{row["file_name"]}ls_{x}_{y}',
                      [model_arch['norm_vals']['x'],model_arch['norm_vals']['y'],model_arch['norm_vals']['z']])
        pbar.update()
    pbar.close()
    return None