import torch
import re
import os
import uproot
import numpy as np
import awkward as ak
import vector
import boost_histogram as bh
import sklearn.metrics as m
from types import SimpleNamespace

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

def get_tensorboard_custom_fn(tb, model_output, model, epoch, i_batch, mode, inputs=None, **kwargs):
    
    # keep the model output at eval interations
    if mode == 'eval' and i_batch >= 0:
        if i_batch == 1:
            tb.model_output_v = []
            tb.label_index_v = []
            tb.observers_v = []
        tb.model_output_v.append(model_output[0])
        tb.label_index_v.append(model_output[1])
        tb.observers_v.append(model_output[-1])

    if mode == 'eval' and i_batch == -1: # in summary stage

        # concat output
        output = [torch.cat([out[ie] for out in tb.model_output_v], dim=0) for ie in range(len(tb.model_output_v[0]))]
        disc_val_ensemble = [torch.softmax(out, dim=1).detach().cpu().numpy()[:, 0] for out in output] # score_label_sr
        disc_val = sum(disc_val_ensemble) / len(disc_val_ensemble)

        label_index = torch.cat(tb.label_index_v, dim=0).detach().cpu().numpy().astype(int)
        observers = torch.cat(tb.observers_v, dim=0).detach().cpu().numpy()
        event_class = observers[:, 0].astype(int)
        jet_sdmass = observers[:, 1]

        # make plots
        match = re.search(r'bin(\d+)to(\d+)', tb.writer.log_dir)
        if match:
            edge_bin = (int(match.group(1)), int(match.group(2)))
        edge_sr = (edge_bin[0]-10, edge_bin[1]+10)
        edge_full = (edge_bin[0]-20, edge_bin[1]+20)

        # 1. plot discriminator distribution
        f, axes = plt.subplots(3, figsize=(5, 9))
        for iax, ax in enumerate(axes):
            if iax == 0:
                eval_flag = ((jet_sdmass > edge_full[0]) & (jet_sdmass < edge_sr[0])) | ((jet_sdmass > edge_sr[1]) & (jet_sdmass < edge_full[1]))
            elif iax == 1:
                eval_flag = (jet_sdmass > edge_sr[0]) & (jet_sdmass < edge_sr[1])
            elif iax == 2:
                eval_flag = (jet_sdmass > edge_bin[0]) & (jet_sdmass < edge_bin[1])

            eval_flag = eval_flag & (label_index == 0) 
            samples = ["QCD", "WJetsToQQ", "WJetsToLNu", "ZJetsToQQ", "ZJetsToLL", "ZJetsToNuNu", "TTbar", "SingleTop", "WW", "TW", "ZW", "ZZ", "TZ", "SingleHiggs", "WH", "ZH", "TTbarH", "TTbarW", "TTbarZ", "Xbb", "Xbs", "Wkk"]
            sample_info = {sam: i for i, sam in enumerate(samples)}
            sample_sm_info = {
                'QCD': ('event_class == 0', 'QCD'),
                'VJets': ('(event_class >= 1) & (event_class <= 5)', 'V+jets'),
                'WJets': ('(event_class >= 1) & (event_class <= 2)', 'W+jets'),
                'ZJets': ('(event_class >= 3) & (event_class <= 5)', 'Z+jets'),
                'TTbarST': ('(event_class == 6) | (event_class == 7) | (event_class == 9)', r'$t\overline{t}$+ST'),
                'VV': ('(event_class == 8) | (event_class == 10) | (event_class == 11)', r'VV'),
                'Higgs': ('(event_class >= 13) & (event_class <= 16)', 'Higgs'),
                'Xbb': ('(event_class == 19)', r'$X\rightarrow bb$'),
                'Xbs': ('(event_class == 20)', r'$X\rightarrow bs$'),
                'Wkk': ('(event_class == 21)', r'$W_{kk}\rightarrow 3W$'),
            }

            nbin, xmin, xmax, density = 50, np.min(disc_val), np.max(disc_val), True

            sam_sm_draw = ['QCD', 'WJets', 'ZJets', 'TTbarST', 'VV', 'Higgs']
            for sam in sam_sm_draw:
                selexpr, label = sample_sm_info[sam]
                print('reading... ', sam)
                hist = bh.Histogram(bh.axis.Regular(nbin, xmin, xmax), storage=bh.storage.Weight())
                hist.fill(disc_val[eval_flag & eval(selexpr)])
                content, yerr = hist.view().value, np.sqrt(hist.view().variance)
                if density:
                    content, yerr = content / sum(content), yerr / sum(content)

                hep.histplot(content, yerr=yerr, bins=hist.axes[0].edges, label=label, ax=ax)

            ax.legend()
            ax.set_xlabel('Disc', ha='right', x=1.0); ax.set_ylabel('Events / bins', ha='right', y=1.0)
            ax.set_ylim(0, 0.15)

        tb.writer.add_figure(f'Dist/eval/epoch{str(epoch).zfill(4)}', f)

        if 'step1' in tb.writer.log_dir:
            return

        # 2. plot proportion of different classes
        f, axes = plt.subplots(3, figsize=(5, 9))

        bkg_flag = (jet_sdmass > edge_bin[0]) & (jet_sdmass < edge_bin[1]) & (label_index == 1) # use bkg events to estimate wps
        for ax, effb in zip(axes, [1e-2, 1e-3, 1e-4]):
            wps_ensemble = [np.quantile(disc[bkg_flag], q=1-effb) for disc in disc_val_ensemble]
            wps_ave = np.quantile(disc_val[bkg_flag], q=1-effb)

            points = []
            for wp, disc in zip(wps_ensemble, disc_val_ensemble):
                wpsel = eval_flag & (disc > wp)
                for isam, sam in enumerate(sam_sm_draw):
                    selexpr, label = sample_sm_info[sam]
                    points.append((isam, len(event_class[wpsel & eval(selexpr)])))

            # average
            points_ave = []
            wpsel = eval_flag & (disc_val > wps_ave)
            for isam, sam in enumerate(sam_sm_draw):
                selexpr, label = sample_sm_info[sam]
                points_ave.append((isam, len(event_class[wpsel & eval(selexpr)])))

            # draw scattering plot
            points = np.array(points)
            points_ave = np.array(points_ave)
            ax.scatter(points[:,0], points[:,1], marker='o')
            ax.scatter(points_ave[:,0], points_ave[:,1], marker='x')
            ax.set_xticks(range(len(sam_sm_draw)))
            ax.set_xticklabels([sample_sm_info[sam][1] for sam in sam_sm_draw])
            ax.set_ylabel('Events')
            ax.set_yscale('log')

        tb.writer.add_figure(f'Proportion/eval/epoch{str(epoch).zfill(4)}', f)
