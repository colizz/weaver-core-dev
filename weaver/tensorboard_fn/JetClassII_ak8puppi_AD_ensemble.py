import torch
import re
import os
import uproot
import numpy as np
import awkward as ak
import boost_histogram as bh
import sklearn.metrics as m
from types import SimpleNamespace

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

def get_tensorboard_custom_fn(tb, model_output, model, epoch, i_batch, mode, inputs=None, **kwargs):
    
    if mode == 'eval' and i_batch == -1: # in evaluation, summary stage
        # if epoch % 2 != 0:
        #     return

        if not hasattr(tb, 'eval_input_arrays'):
            # read data
            nevt_read = 50000

            if os.uname()[1].startswith('zeus'):
                basedir = '/mldata/licq/datasets/JetClassII'
            elif os.uname()[1].startswith('farm221'):
                basedir = '/data/licq/samples/JetClassII'

            tb._config = SimpleNamespace(
                edge_full = (110, 270) if 'expandsr' in tb.writer.log_dir else (150, 230),
                edge_sr = (150, 230) if 'expandsr' in tb.writer.log_dir else (170, 210),
                signal_name = 'Xbb' if 'xbb' in tb.writer.log_dir else 'Xbs' if 'xbs' in tb.writer.log_dir else None,
                signal_idx = 19 if 'xbb' in tb.writer.log_dir else 20 if 'xbs' in tb.writer.log_dir else None,
            )
            df = uproot.lazy(f"{basedir}/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root")
            df = df[ak.numexpr.evaluate(f"(jet_sdmass >= {tb._config.edge_full[0]}) & (jet_sdmass < {tb._config.edge_full[1]}) & ((event_class <= 18) | (event_class == {tb._config.signal_idx}))", df)]
            tb._df = df[:nevt_read]
            tb._df_nevt_scale = len(df) / nevt_read

            if 'hlvars' not in tb.writer.log_dir:
                tb._eval_input_arrays = torch.Tensor(ak.to_numpy(tb._df.jet_hidneurons)).to(model_output[0][0]).unsqueeze(1) # (N, 1, C)
            else:
                input = ak.concatenate([
                    tb._df.jet_nparticles * 0.05 - 3,
                    np.log(np.maximum(tb._df.jet_tau1, 1e-10)) + 1.0,
                    np.log(np.maximum(tb._df.jet_tau2, 1e-10)) + 2.5,
                    np.log(np.maximum(tb._df.jet_tau3, 1e-10)) + 3.0,
                    np.log(np.maximum(tb._df.jet_tau4, 1e-10)) + 3.0,
                ], axis=1)
                tb._eval_input_arrays = torch.Tensor(ak.to_numpy(input)).to(model_output[0][0]) # (N, C)
        
        output = model(tb._eval_input_arrays)[0]
        disc_val_ensemble = [torch.softmax(out, dim=1).detach().cpu().numpy()[:, 0] for out in output] # score_label_sr
        disc_val = sum(disc_val_ensemble) / len(disc_val_ensemble)

        # make plots
        nbin, xmin, xmax, density = 50, np.min(disc_val), np.max(disc_val), True

        select_mass_region = {
            'mass region': f'(jet_sdmass >= {tb._config.edge_sr[0]}) & (jet_sdmass < {tb._config.edge_sr[1]})',
            'mass sideband': f'((jet_sdmass >= {tb._config.edge_full[0]}) & (jet_sdmass < {tb._config.edge_sr[0]})) | ((jet_sdmass >= {tb._config.edge_sr[1]}) & (jet_sdmass < {tb._config.edge_full[1]}))',
        }
        samples = ["QCD", "WJetsToQQ", "WJetsToLNu", "ZJetsToQQ", "ZJetsToLL", "ZJetsToNuNu", "TTbar", "SingleTop", "WW", "TW", "ZW", "ZZ", "TZ", "SingleHiggs", "WH", "ZH", "TTbarH", "TTbarW", "TTbarZ", "Xbb", "Xbs"]
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
            s = len(y_true[y_true == 1]) * tb._df_nevt_scale * tpr
            b = len(y_true[y_true == 0]) * tb._df_nevt_scale * fpr
            z = np.sqrt(np.maximum(2*((s + b)*(np.log(1 + s/(b+1e-10))) - s), 0.))
            if i != -1:
                axes[0].plot(tpr, z)
                axes[1].plot(fpr, z)
            else:
                axes[0].plot(tpr, z, color='black', linewidth=2)
                axes[1].plot(fpr, z, color='black', linewidth=2)

        axes[0].set_xlim([0.0, 1.0]); axes[1].set_xlim([1e-4, 1.0]); axes[1].set_xscale('log'); 
        axes[0].set_ylim([0.0, 10]); axes[1].set_ylim([0.0, 10])
        axes[0].set_xlabel('Signal efficiency', ha='right', x=1.0); axes[1].set_xlabel('Backgronud efficiency', ha='right', x=1.0); axes[0].set_ylabel('Significance', ha='right', y=1.0)
        axes[0].grid(); axes[1].grid()
        tb.writer.add_figure(f'Signif/eval/epoch{str(epoch).zfill(4)}', f)
