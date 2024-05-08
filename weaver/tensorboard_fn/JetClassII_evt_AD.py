import torch
import re
import os
import tqdm
import glob

from torch.utils.data import DataLoader
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.import_tools import import_module

VAELoss = import_module(os.path.join(os.path.dirname(__file__), '../', 'networks/TransformerVAE.py'), 'TransformerVAE').VAELoss

### configuration ###
dataloader_info = {
    'H2B': (glob.glob('/home/olympus/lyazj/software/ad-part/run/sm-md/data4/H2B/0.root'), ),
    'HH4B': (glob.glob('/home/olympus/lyazj/software/ad-part/run/sm-md/data4/HH4B/0.root'), ),
    'Wkk': (glob.glob('/home/olympus/lyazj/software/ad-part/run/sm-dijet/data4/WkkTo3WTo6Q/0.root'), ),
}
eval_batch_size = 512
eval_steps_per_epoch = 100
#####################

def get_tensorboard_custom_fn(tb, model_output, model, epoch, i_batch, mode, inputs=None, **kwargs):
    
    if mode == 'eval' and i_batch == -1: # in evaluation, summary stage

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
            tb.loss_func = VAELoss(kld_weight=1.0) # the value of kld_weight doesn't matter as we only need to extract component losses

        # evaluate model on BSM datasets
        model.eval()
        dev = next(model.parameters()).device
        with torch.no_grad():
            for name, dataloader in tb.bsm_dataloaders.items():
                print('Evaluating BSM data: ', name, '...')
                total_losses = None

                with tqdm.tqdm(dataloader) as tq:
                    for i_batch, (X, *_) in enumerate(tq):
                        inputs = [X[k].to(dev) for k in tb.data_config.input_names]
                        model_output = model(*inputs)
                        if not isinstance(model_output, tuple):
                            model_output = (model_output,)
                        losses = tb.loss_func(*model_output)

                        if total_losses is None:
                            total_losses = {k: 0. for k in losses}
                        for k in losses:
                            losses[k] = losses[k].item()
                            total_losses[k] += losses[k]
                        tq.set_postfix({
                            **{k: '%.5f' % losses[k] for k in list(losses.keys())[:3] if k != 'loss'},
                        })

                        if i_batch >= eval_steps_per_epoch:
                            break

                # write component losses
                tb.write_scalars(
                    [(k + f'/eval_{name} (epoch)', total_losses[k] / eval_steps_per_epoch, epoch) for k in losses if k != 'loss'])
