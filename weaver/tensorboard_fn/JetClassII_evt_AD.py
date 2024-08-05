import torch
import re
import os
import tqdm
import glob
import numpy as np
import sklearn.metrics as m

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

from torch.utils.data import DataLoader
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.import_tools import import_module

### configuration ###
dataloader_info = {
    'H2B': (glob.glob('/home/olympus/lyazj/software/ad-part/run/sm-md/data4/H2B/0.root'), ),
    'HH4B': (glob.glob('/home/olympus/lyazj/software/ad-part/run/sm-md/data4/HH4B/0.root'), ),
    'Wkk': (glob.glob('/home/olympus/lyazj/software/ad-part/run/sm-dijet/data4/WkkTo3WTo6Q/0.root'), ),
}
eval_batch_size = 512
eval_steps_per_epoch = 100
discr_fn = lambda _losses: _losses['loss_kld'] # define the discriminant used to make ROC curve
#####################

class VAELossNoMean(torch.nn.Module):
    '''
        Computes the VAE loss function (without taking the mean over the batch dim)
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    '''

    def __init__(self, kld_weight=1.):
        super(VAELossNoMean, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, recons_colls, input_colls, mask_colls, mu, log_var):
        '''
            recons_colls, input_colls, and mask_colls: list of corresponding tensors for each input group.
                recons_colls: (batch, seq, input_dim) for each element
                input_colls: (batch, seq, input_dim) ...
                mask_colls: (batch, seq) ...
            mu, log_var: in dim (batch, seq, dim_latent)
        '''

        # both losses need to omit masked tokens
        #! mask_colls[inds]: dim (batch, seq_i); 0/1
        
        # do mean on batch 
        def mse_loss(A, B):  
            squared_error = (A - B) ** 2
            # return torch.sum(torch.mean(squared_error, dim=0))
            # sum on the last two dimension
            return torch.sum(squared_error, dim=[-1, -2]) #! sum on (seq, dim) #!!! no mean on batch dim
            
        loss_recons = [mse_loss(recons_colls[inds] * mask_colls[inds].unsqueeze(-1), \
            input_colls[inds] * mask_colls[inds].unsqueeze(-1)) for inds in range(len(recons_colls))]
        loss_recons = sum(loss_recons)

        mask = torch.cat(mask_colls, dim=1)  # ! [(batch, seq_1), ..., (batch, seq_4)] ==> (batch, seq)
        loss_kld = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1) # (batch, seq)
        loss_kld = loss_kld.masked_fill(~mask, 0).sum(dim=1)  #! do mean on batch dimensiom; sum on (seq, dim) #!!! no mean on batch dim

        loss = loss_recons + self.kld_weight * loss_kld
        return {'loss': loss, 'loss_recons': loss_recons.detach(), 'loss_kld': -loss_kld.detach()}


def get_tensorboard_custom_fn(tb, model_output, model, epoch, i_batch, mode, inputs=None, **kwargs):
    
    ## initialization
    if not hasattr(tb, 'bsm_dataloaders'):
        # initialize weaver data loader for BSM datasets

        match = re.search(r'(?<=phy.pku.edu.cn_)[^.]*', tb.writer.log_dir)
        if match:
            data_card = match.group(0)
        else:
            raise ValueError(f"Cannot find data card from log_dir: {tb.writer.log_dir}")

        data_config = os.path.join(os.path.dirname(__file__), '../data_pheno/EvtClass/', data_card + '.yaml')

        tb.bsm_dataloaders = {}
        for name, (path, ) in dataloader_info.items():
            print('Initializing BSM dataloader: ', name, '...')
            _data = SimpleIterDataset(
                {"_": path}, data_config, for_training=True,
                load_range_and_fraction=((0, 1), 1), fetch_by_files=True, fetch_step=1, infinity_mode=True, in_memory=True, name='eval_' + name
                )
            tb.bsm_dataloaders[name] = DataLoader(_data, batch_size=eval_batch_size, drop_last=True, pin_memory=True, num_workers=1, persistent_workers=True)
            if not hasattr(tb, 'data_config'):
                tb.data_config = _data.config

        # initialate loss function
        tb.loss_func = VAELossNoMean(kld_weight=1.0) # the value of kld_weight doesn't matter as we only need to extract component losses

        tb.discrs = {name: None for name in tb.bsm_dataloaders}
        tb.discrs['qcd'] = None

    # at evaluation routine: get loss from qcd
    if mode == 'eval' and i_batch >= 0:
        if i_batch == 1:
            for n in tb.discrs:
                tb.discrs[n] = []

        with torch.no_grad():
            if not isinstance(model_output, tuple):
                model_output = (model_output,)
            losses_v = tb.loss_func(*model_output)
            tb.discrs['qcd'].append(discr_fn(losses_v).detach().cpu().numpy())


    if mode == 'eval' and i_batch == -1: # in evaluation, summary stage

        # evaluate model on BSM datasets
        model.eval()
        dev = next(model.parameters()).device
        with torch.no_grad():
            for name, dataloader in tb.bsm_dataloaders.items():
                print('Evaluating BSM data: ', name, '...')
                total_losses = None
                losses = {}

                with tqdm.tqdm(dataloader) as tq:
                    for i_batch, (X, *_) in enumerate(tq):
                        inputs = [X[k].to(dev) for k in tb.data_config.input_names]
                        model_output = model(*inputs)
                        if not isinstance(model_output, tuple):
                            model_output = (model_output,)
                        losses_v = tb.loss_func(*model_output)
                        tb.discrs[name].append(discr_fn(losses_v).detach().cpu().numpy())

                        if total_losses is None:
                            total_losses = {k: 0. for k in losses_v}
                        for k in losses_v:
                            losses[k] = losses_v[k].mean().item()
                            total_losses[k] += losses[k]
                        tq.set_postfix({
                            **{k: '%.5f' % losses[k] for k in list(losses.keys())[:3] if k != 'loss'},
                        })

                        if i_batch >= eval_steps_per_epoch:
                            break

                # write component losses
                tb.write_scalars(
                    [(k + f'/eval_{name} (epoch)', total_losses[k] / eval_steps_per_epoch, epoch) for k in losses if k != 'loss'])

        # make ROC curve
        for n in tb.discrs:
            tb.discrs[n] = np.concatenate(tb.discrs[n])
        
        f, ax = plt.subplots(figsize=(5, 5))
        for i, name in enumerate(tb.bsm_dataloaders):
            fpr, tpr, _ = m.roc_curve(
                np.concatenate([np.zeros_like(tb.discrs['qcd']), np.ones_like(tb.discrs[name])]),
                np.concatenate([tb.discrs['qcd'], tb.discrs[name]])
            )
            ax.plot(fpr, tpr, label=name + ' vs QCD (AUC=%.4f)' % m.auc(fpr, tpr))
        ax.legend()
        ax.set_xlabel('False positive rate (QCD selection eff.)', ha='right', x=1.0); ax.set_ylabel('True positive rate (BSM events selec. eff.)', ha='right', y=1.0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        tb.writer.add_figure(f'ROC/eval/epoch{str(epoch).zfill(4)}', f)
