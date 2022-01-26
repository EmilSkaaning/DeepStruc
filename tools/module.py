import torch.nn as nn
import torch, sys, random
import torch.nn.functional as F
import torch.nn
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv, GraphConv, SAGEConv, GCN2Conv, EdgeConv, AGNNConv, DNAConv
import pytorch_lightning as pl
from collections import OrderedDict
from torch_geometric.nn.glob import global_add_pool, GlobalAttention
import numpy as np
from torch.distributions import Normal, Independent
from torch.distributions.kl import kl_divergence as KLD

seed = 37
torch.manual_seed(seed)
pl.seed_everything(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class Net(pl.LightningModule):
    def __init__(self, model_arch, lr=1e-4, beta=0, beta_inc=0.001, beta_max=1, rec_th=0.0001):
        super(Net, self).__init__()
        self.actFunc = nn.LeakyReLU()
        self.actFunc_ReLU = nn.ReLU()
        self.cluster_size = model_arch['decoder']['out_dim']
        self.latent_space = model_arch['latent_space']
        self.beta = beta  # starting val
        self.beta_inc = beta_inc  # beta increase
        self.rec_th = rec_th  # Update beta if loss_rec is =< this value
        self.last_beta_update = 0
        self.beta_max = beta_max
        self.lr = lr
        self.num_node_features = model_arch['node_features']
        self.encoder_layers = self.Encoder(model_arch['node_features'], model_arch['encoder'], model_arch['mlps']['m0'])
        self.decoder_layers = self.Decoder(model_arch['node_features'], model_arch['decoder'], model_arch['latent_space'])
        self.mlp_layers = self.MLPs(model_arch['mlps'], model_arch['latent_space'])

        self.prior_layers = self.conditioning_nw(model_arch['PDF_len'], model_arch['prior'], self.latent_space * 2)
        self.posterior_layers = self.conditioning_nw(model_arch['PDF_len'], model_arch['posterior'], model_arch['mlps']['m0'])  # Posterior
        self.glob_at = GlobalAttention(torch.nn.Linear(model_arch['mlps']['m0'], 1), torch.nn.Linear(model_arch['mlps']['m0'], model_arch['mlps']['m0']))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


    def MLPs(self, model_arch, latent_dim):
        layers = OrderedDict()

        for idx, key in enumerate(model_arch.keys()):
            if idx == 0:
                layers[str(key)] = torch.nn.Linear(model_arch[key]*2, model_arch[key])
            else:
                layers[str(key)] = torch.nn.Linear(former_nhid, model_arch[key])

            former_nhid = model_arch[key]

        layers['-1'] = torch.nn.Linear(former_nhid, latent_dim*2)


        return nn.Sequential(layers)


    def Encoder(self, init_data, model_arch, out_dim):
        layers = OrderedDict()

        for idx, key in enumerate(model_arch.keys()):
            if idx == 0:
                layers[str(key)] = GATConv(init_data, model_arch[key])
            else:
                layers[str(key)] = GATConv(former_nhid, model_arch[key])

            former_nhid = model_arch[key]


        #layers['-1'] = GATConv(former_nhid, model_arch['m0'])
        layers[str('e{}'.format(idx + 1))] = GATConv(former_nhid, out_dim)

        return nn.Sequential(layers)

    def Decoder(self, init_data, model_arch, latent_dim):
        layers = OrderedDict()

        for idx, key in enumerate(model_arch.keys()):
            if idx == 0 :
                layers[str(key)] = nn.Linear(latent_dim, model_arch[key])
            elif key == 'out_dim':
                continue
            else:
                layers[str(key)] = nn.Linear(former_nhid, model_arch[key])

            former_nhid = model_arch[key]


        layers[str('d{}'.format(idx+1))] = nn.Linear(former_nhid, model_arch['out_dim']*init_data)

        return nn.Sequential(layers)

    def conditioning_nw(self, pdf, model_arch, out):
        ### Conditioning network on prior for atom list
        ### Creates additional node features per node
        ### Assumes 1xself.atomRangex1 one hot encoding vector as input
        ### Output: 1x2*latent_dimx1
        """conditioning_layers = nn.Sequential(
            GatedConv1d(pdf, 48, kernel_size=1, stride=1), nn.ReLU(),
            GatedConv1d(48, 24, kernel_size=1, stride=1), nn.ReLU(),
            GatedConv1d(24, out, kernel_size=1, stride=1))"""


        conditioning_layers = torch.nn.Sequential()
        for idx, key in enumerate(model_arch.keys()):
            if idx == 0:
                conditioning_layers.add_module(str(key), GatedConv1d(pdf, model_arch[key], kernel_size=1, stride=1))
            else:
                conditioning_layers.add_module(str(key), GatedConv1d(former_nhid, model_arch[key], kernel_size=1, stride=1))

            former_nhid = model_arch[key]
        conditioning_layers.add_module('-1', GatedConv1d(former_nhid, out, kernel_size=1, stride=1))

        return conditioning_layers


    def forward(self, data, mode='posterior'):
        """

        Parameters
        ----------
        data :
        mode : str - posterior, prior or generate

        Returns
        -------

        """

        if mode == 'posterior':
            pdf_cond = data[1].to(self.device)
            data = data[0].to(self.device)
            try:
                this_batch_size = len(data.batch.unique())
            except:
                this_batch_size = 1

            # Prior
            prior = self.get_prior_dist(pdf_cond)

            # Posterior
            posterior = self.get_posterior_dist(data, pdf_cond, this_batch_size)

            # Divergence between posterior and prior
            kl = KLD(posterior, prior) / this_batch_size

            # Draw z from posterior distribution
            z_sample = posterior.rsample()
            z = z_sample.clone()

        elif mode == 'prior':
            try:
                hej = data.clone()
                pdf_cond = data.to(self.device)
                this_batch_size = len(data)
            except:
                #print(data)
                pdf_cond = data[1].to(self.device)
                this_batch_size = 1


            # Prior
            prior = self.get_prior_dist(pdf_cond)

            # Draw z from prior distribution
            z_sample = prior.rsample()
            z = z_sample.clone()
            kl = torch.zeros(this_batch_size) -1

        elif mode == 'generate':
            # Set is given
            z = data.clone()
            z_sample = data.clone()
            this_batch_size = 1
            kl = torch.zeros(this_batch_size) -1

        # Decoder
        for idx, layer in enumerate(self.decoder_layers):
            if idx == len(self.decoder_layers)-1:
                z_sample = layer(z_sample)
            else:
                z_sample = self.actFunc(layer(z_sample))

        z_sample = z_sample.view(this_batch_size, self.cluster_size, self.num_node_features)  # Output

        return z_sample, z, kl#.mean()


    def get_prior_dist(self, pdf_cond):
        cond_prior = pdf_cond.clone()

        for idx, layer in enumerate(self.prior_layers):
            if idx == len(self.prior_layers) - 1:
                cond_prior = layer(cond_prior)
            else:
                cond_prior = self.actFunc(layer(cond_prior))

        cond_prior = cond_prior.squeeze(-1)
        prior = self.get_distribution(cond_prior)
        return prior


    def get_posterior_dist(self, data, pdf_cond, this_batch_size):
        cond_post = pdf_cond.clone()

        # Posterior
        for idx, layer in enumerate(self.posterior_layers):
            if idx == len(self.posterior_layers) - 1:
                cond_post = layer(cond_post)
            else:
                cond_post = self.actFunc(layer(cond_post))

        # Encoder
        z = data.x.clone()
        for idx, layer in enumerate(self.encoder_layers):
            if idx == len(self.encoder_layers) - 1:
                z = layer(z, data.edge_index)
            else:
                edge_index = data.edge_index

                z = self.actFunc(layer(z, edge_index))
        test = z.clone()

        #z = global_add_pool(z, data.batch, size=this_batch_size)  # Sum note features
        z = self.glob_at(test, data.batch, size=this_batch_size)

        cond_post = cond_post.squeeze(-1)

        z = torch.cat((z, cond_post), -1)

        for idx, layer in enumerate(self.mlp_layers):
            if idx == len(self.mlp_layers) - 1:
                z = layer(z)
            else:
                z = self.actFunc(layer(z))

        # Draw from distribution
        posterior = self.get_distribution(z)
        return posterior


    def get_distribution(self, z):
        mu, log_var = torch.chunk(z, 2, dim=-1)
        log_var = nn.functional.softplus(log_var)  # Sigma can't be negative
        sigma = torch.exp(log_var / 2)

        distribution = Independent(Normal(loc=mu, scale=sigma), 2)
        return distribution


    def training_step(self, batch, batch_nb):
        prediction, _, kl = self.forward(batch)

        loss = weighted_mse_loss(prediction, batch[0]['y'], self.device)

        #loss = F.mse_loss(prediction, batch[0]['y'])
        log_loss = loss#torch.log(loss)

        tot_loss = log_loss + (self.beta * kl)

        self.log('trn_tot', tot_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log('trn_rec', loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log('trn_log_rec', log_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log('trn_kld', kl, prog_bar=False, on_step=False, on_epoch=True)

        return tot_loss


    def validation_step(self, batch, batch_nb):
        prediction, _, kl = self.forward(batch)
        prediction_pdf, _, _ = self.forward(batch[1], mode='prior')

        #loss = weighted_mse_loss(prediction, batch[0]['y'], self.device, node_weight=5)
        #loss_pdf = weighted_mse_loss(prediction_pdf, batch[0]['y'], self.device, node_weight=5)

        loss = F.mse_loss(prediction, batch[0]['y'])
        loss_pdf = F.mse_loss(prediction_pdf, batch[0]['y'])

        log_loss = loss#torch.log(loss)

        tot_loss = log_loss + (self.beta * kl)

        if (self.last_beta_update != self.current_epoch and self.beta < self.beta_max) and loss <= self.rec_th:
            self.beta += self.beta_inc
            self.last_beta_update = self.current_epoch

        beta = self.beta
        self.log('vld_tot', tot_loss, prog_bar=True, on_epoch=True)
        self.log('vld_rec', loss, prog_bar=True, on_epoch=True)
        self.log('vld_log_rec', log_loss, prog_bar=True, on_epoch=True)
        self.log('vld_rec_pdf', loss_pdf, prog_bar=True, on_epoch=True)
        self.log('vld_kld', kl, prog_bar=True, on_epoch=True)
        self.log('beta', beta, prog_bar=True, on_step=False, on_epoch=True)

        return tot_loss


    def test_step(self, batch, batch_nb):
        prediction, _, kl = self.forward(batch)
        prediction_pdf, _, _ = self.forward(batch[1], mode='prior')

        #loss = weighted_mse_loss(prediction, batch[0]['y'], self.device, node_weight=5)
        #loss_pdf = weighted_mse_loss(prediction_pdf, batch[0]['y'], self.device, node_weight=5)

        loss = F.mse_loss(prediction, batch[0]['y'])
        loss_pdf = F.mse_loss(prediction_pdf, batch[0]['y'])

        log_loss = loss#torch.log(loss)

        tot_loss = log_loss + (self.beta * kl)

        self.log('tst_tot', tot_loss, prog_bar=False, on_epoch=True)
        self.log('tst_rec', loss, prog_bar=False, on_epoch=True)
        self.log('tst_log_rec', log_loss, prog_bar=False, on_epoch=True)
        self.log('tst_rec_pdf', loss_pdf, prog_bar=False, on_epoch=True)
        self.log('tst_kld', kl, prog_bar=False, on_epoch=True)

        return tot_loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class GatedConv1d(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size, stride, padding=0, dilation=1, activation=None):
        super(GatedConv1d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv1d(input_channels, output_channels, kernel_size,
                           stride, padding, dilation)
        self.g = nn.Conv1d(input_channels, output_channels, kernel_size,
                           stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))

        return h * g


def weighted_mse_loss(pred, label,device, dummy_weight=0.1, node_weight=1):
    """

    Parameters
    ----------
    pred : Predictions. (tensor)
    label : True labels. (tensor)
    dummy_weight : Weight of dummy nodes, default is 0.1. (float)

    Returns
    -------
    this_loss : Computed loss. (tensor)
    """
    mask = torch.ones(label.shape).to(device)
    mask[label == -1.] = dummy_weight
    mask[label >= -0] = node_weight

    loss_func = nn.MSELoss(reduction='none')
    this_loss = loss_func(pred, label)
    this_loss = this_loss*mask

    return this_loss.mean()