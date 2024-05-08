import torch
import re
import os
import glob
import tqdm
import uproot
import numpy as np
import awkward as ak
import vector
import boost_histogram as bh
import sklearn.metrics as m
from types import SimpleNamespace

from torch.utils.data import DataLoader
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.import_tools import import_module

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

if os.uname()[1].startswith('zeus'):
    basedir = '/mldata/licq/datasets/JetClassII'
elif os.uname()[1].startswith('farm221'):
    basedir = '/data/licq/samples/JetClassII'

### configuration ###
filelist = glob.glob(f'{basedir}/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part1.root')
nevt_read = 20000
eval_batch_size = 200
sig_weight = 0.01
config = SimpleNamespace(
    edge_full = (2200, 3400),
    edge_sr = (2500, 3100),
    signal_name = 'Wkk',
    signal_idx = 21,
)
#####################


def get_tensorboard_custom_fn(tb, model_output, model, epoch, i_batch, mode, inputs=None, **kwargs):
    
    if mode == 'eval' and i_batch == -1: # in evaluation, summary stage
        # if epoch % 2 != 0:
        #     return

        # if not hasattr(tb, 'eval_input_arrays'):
        #     # read data
        #     nevt_read = 50000
        #     sig_weight = 0.01


        #     tb._config = SimpleNamespace(
        #         edge_full = (2200, 3400),
        #         edge_sr = (2500, 3100),
        #         signal_name = 'Wkk',
        #         signal_idx = 21,
        #     )
        #     df = uproot.lazy()
        #     df = df[ak.numexpr.evaluate(f"(dijet_mass >= {tb._config.edge_full[0]}) & (dijet_mass < {tb._config.edge_full[1]}) & ((event_class <= 18) | (event_class == {tb._config.signal_idx}))", df)]
        #     _df = df[:nevt_read]
        #     tb._df = _df
        #     tb._df_nevt_scale = len(df) / nevt_read

        #     _arr1 = torch.Tensor(ak.to_numpy(tb._df.jet_1_hidneurons)).to(model_output[0][0]).unsqueeze(1) # (N, 1, C)
        #     _arr2 = torch.Tensor(ak.to_numpy(tb._df.jet_2_hidneurons)).to(model_output[0][0]).unsqueeze(1) # (N, 1, C)
        #     tb._eval_input_arrays = torch.cat([_arr1, _arr2], dim=1)

        if not hasattr(tb, 'dataloader'):

            # initialize weaver data loader for BSM datasets

            match = re.search(r'(?:phy.pku.edu.cn_|farm221_)([^.]*)', tb.writer.log_dir)
            if match:
                data_card = match.group(1)
            else:
                raise ValueError(f"Cannot find data card from log_dir: {tb.writer.log_dir}")

            data_config = os.path.join(os.path.dirname(__file__), '../data_pheno/AD_dijet/', data_card + '.yaml')

            tb.nevt_scale = len(uproot.lazy(filelist)) / nevt_read

            _data = SimpleIterDataset(
                {"_": filelist}, data_config, for_training=False,
                load_range_and_fraction=((0, 1. / tb.nevt_scale), 1), fetch_by_files=True, fetch_step=1, name='eval_tb'
                )
            tb.dataloader = DataLoader(_data, batch_size=eval_batch_size, drop_last=True, pin_memory=True, num_workers=1, persistent_workers=True)
            tb.dataloader_iter = iter(tb.dataloader)
            if not hasattr(tb, 'data_config'):
                tb.data_config = _data.config
            
            tb._config = config

            # load root file
            df = uproot.lazy(filelist)
            tb._df = df[ak.numexpr.evaluate(tb.data_config.test_time_selection, df)][:nevt_read]


        # evaluate model on eval dataset
        model.eval()
        dev = next(model.parameters()).device
        output_v = []
        with torch.no_grad():
            with tqdm.tqdm(tb.dataloader) as tq:
                for i_batch, (X, *_) in enumerate(tq):

                    inputs = [X[k].to(dev) for k in tb.data_config.input_names]
                    output = model(*inputs)
                    # for ensemble output:
                    if isinstance(output, tuple):
                        output = output[0]
                    else:
                        output = [output]

                    output_v.append(output)

        # concat output
        output = [torch.cat([out[ie] for out in output_v], dim=0) for ie in range(len(output_v[0]))]

        disc_val_ensemble = [torch.softmax(out, dim=1).detach().cpu().numpy()[:, 0] for out in output] # score_label_sr
        disc_val = sum(disc_val_ensemble) / len(disc_val_ensemble)
        print(len(disc_val), len(tb._df))

        # make plots
        nbin, xmin, xmax, density = 50, np.min(disc_val), np.max(disc_val), True

        select_mass_region = {
            'mass region': f'(dijet_mass >= {tb._config.edge_sr[0]}) & (dijet_mass < {tb._config.edge_sr[1]})',
            'mass sideband': f'((dijet_mass >= {tb._config.edge_full[0]}) & (dijet_mass < {tb._config.edge_sr[0]})) | ((dijet_mass >= {tb._config.edge_sr[1]}) & (dijet_mass < {tb._config.edge_full[1]}))',
        }
        samples = ["QCD", "WJetsToQQ", "WJetsToLNu", "ZJetsToQQ", "ZJetsToLL", "ZJetsToNuNu", "TTbar", "SingleTop", "WW", "TW", "ZW", "ZZ", "TZ", "SingleHiggs", "WH", "ZH", "TTbarH", "TTbarW", "TTbarZ", "Xbb", "Xbs", "Wkk"]
        sample_info = {sam: i for i, sam in enumerate(samples)}

        f, ax = plt.subplots(figsize=(5, 5))
        for sam in [tb._config.signal_name, 'SM', 'mass region', 'mass sideband']:
            print('reading... ', sam)
            hist = bh.Histogram(bh.axis.Regular(nbin, xmin, xmax), storage=bh.storage.Weight())

            if sam in ['mass region', 'mass sideband']:
                hist.fill(disc_val[ak.numexpr.evaluate(select_mass_region[sam], tb._df)])
            elif sam in ['SM']:
                hist.fill(disc_val[tb._df.event_class <= 18])
            else:
                idx = sample_info[sam]
                hist.fill(disc_val[tb._df.event_class == idx])
            content, yerr = hist.view().value, np.sqrt(hist.view().variance)
            if density:
                content, yerr = content / sum(content), yerr / sum(content)

            hep.histplot(content, yerr=yerr, bins=hist.axes[0].edges, label=sam)

        ## vertical line
        sel_sig = tb._df.event_class == sample_info[tb._config.signal_name]
        sel_bkg = tb._df.event_class <= 18
        x0 = np.quantile(disc_val[sel_sig], q=0.5)
        plt.axvline(x0, color='k', linestyle='--', linewidth=1)
        tb.writer.add_scalar('Bkg_eff/eval', len(disc_val[sel_bkg][disc_val[sel_bkg] > x0]) / len(disc_val[sel_bkg]), global_step=epoch)
        # print('bkg efficiency at eff_s = 0.5: ', len(disc_val[sel_bkg][disc_val[sel_bkg] > x0]) / len(disc_val[sel_bkg]))

        ax.legend()
        ax.set_xlabel('Disc', ha='right', x=1.0); ax.set_ylabel('Events / bins', ha='right', y=1.0)

        tb.writer.add_figure(f'Dist/eval/epoch{str(epoch).zfill(4)}', f)


        # ROC curve and efficiency
        sel_sr = ak.numexpr.evaluate(select_mass_region['mass region'], tb._df)
        f, axes = plt.subplots(2, figsize=(5, 7.5))

        disc_val_ave = disc_val
        for i in list(range(len(disc_val_ensemble))) + [-1]:
            if i != -1:
                disc_val = disc_val_ensemble[i]
            else:
                disc_val = disc_val_ave

            y_score, y_true = [], []
            for sam in ['SM', tb._config.signal_name]:
                if sam == 'SM':
                    val = disc_val[sel_bkg & sel_sr]
                else:
                    val = disc_val[sel_sig & sel_sr]

                y_score.append(val)
                y_true.append(np.ones_like(y_score[-1]) * (1. if sam != 'SM' else 0.))
            y_score, y_true = np.concatenate(y_score), np.concatenate(y_true)

            # draw roc curves
            fpr, tpr, thres = m.roc_curve(y_true, y_score)

            # draw significance vs signal eff
            s = len(y_true[y_true == 1]) * tb.nevt_scale * tpr * sig_weight
            b = len(y_true[y_true == 0]) * tb.nevt_scale * fpr
            z = np.sqrt(np.maximum(2*((s + b)*(np.log(1 + s/(b+1e-10))) - s), 0.))
            if i != -1:
                axes[0].plot(tpr, z)
                axes[1].plot(fpr, z)
            else:
                axes[0].plot(tpr, z, color='black', linewidth=2)
                axes[1].plot(fpr, z, color='black', linewidth=2)

        axes[0].set_xlim([0.0, 1.0]); axes[1].set_xlim([1e-6, 1.0]); axes[1].set_xscale('log'); 
        axes[0].set_ylim([0.0, 10]); axes[1].set_ylim([0.0, 10])
        axes[0].set_xlabel('Signal efficiency', ha='right', x=1.0); axes[1].set_xlabel('Backgronud efficiency', ha='right', x=1.0); axes[0].set_ylabel('Significance', ha='right', y=1.0)
        axes[0].grid(); axes[1].grid()
        tb.writer.add_figure(f'Signif/eval/epoch{str(epoch).zfill(4)}', f)
