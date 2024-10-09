import numpy as np
import awkward as ak
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import tqdm
import time
import os
from collections import defaultdict, Counter

import sklearn.metrics as m
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

from utils.logger import _logger
from utils.nn.tools import (
    _concat,
    AllGather,
)
from utils.import_tools import import_module

ParticleTransformer = import_module(os.path.join(os.path.dirname(__file__), '../ParticleTransformer2024Plus.py'), 'ParT').ParticleTransformer

'''
This code is adapted from Sophon's official repository: https://github.com/jet-universe/sophon/blob/main/networks/example_ParticleTransformer_sophon.py
Additional features:
 - a univeral wrapper for Sophon models for CLIP-only, CLIP+classification and post-CLIP fine-tuning
'''

class FFN(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, fc_params=[], bias_last=True):
        super().__init__()
        fcs = []
        in_dim = input_dim
        for out_dim, drop_rate in fc_params:
            fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, output_dim, bias=bias_last))
        self.fc = nn.Sequential(*fcs)
    
    def forward(self, x):
        return self.fc(x)


class ParticleTransformerSophonCLIPWrapper(torch.nn.Module):
    '''
        A univeral wrapper for Sophon models for CLIP-only, CLIP+classification and post-CLIP fine-tuning
    '''

    def __init__(self, **kwargs) -> None:
        super().__init__()
        gen_model_kw = kwargs.pop('gen_model_kw')
        clip_kw = kwargs.pop('clip_kw')
        self.clip_mode = clip_kw['mode']
        self.clip_share_token = clip_kw['share_token']

        assert self.clip_mode in ['clip-only', 'clip-with-cls', 'clip-finetune'], 'Invalid mode %s' % self.clip_mode
        if self.clip_mode in ['clip-only', 'clip-with-cls']:

            # remove FC in the model and use outer FC
            fc_params = kwargs.get('fc_params', None)
            kwargs['fc_params'] = None
            gen_model_kw['fc_params'] = None

            # use a second class token in the main model if share_token is False
            if self.clip_mode == 'clip-with-cls' and not self.clip_share_token:
                kwargs['num_cls_tokens'] = 2

            # initialize model
            self.mod = ParticleTransformer(**kwargs)
            self.gen = ParticleTransformer(**gen_model_kw)

            # define outer FCs
            self.mod_proj = FFN(input_dim=kwargs['embed_dims'][-1], output_dim=clip_kw['proj_dim'], fc_params=clip_kw['main_cont_fc_parmas'], bias_last=False)
            self.gen_proj = FFN(input_dim=gen_model_kw['embed_dims'][-1], output_dim=clip_kw['proj_dim'], fc_params=clip_kw['gen_cont_fc_parmas'], bias_last=False)
            if self.clip_mode == 'clip-with-cls':
                # also define FC for classification
                assert fc_params is not None, 'fc_params must be provided for clip-with-cls mode'
                self.mod_fc = FFN(input_dim=kwargs['embed_dims'][-1], output_dim=kwargs['num_classes'], fc_params=fc_params, bias_last=True)

        elif self.clip_mode == 'clip-finetune':

            self.mod = ParticleTransformer(**kwargs)

            # initiate model for clip-finetune mode
            assert clip_kw['init_path'] is not None, 'init_path must be provided for clip-finetune mode'
            init_model_state = {}
            for k, v in torch.load(clip_kw['init_path'], map_location='cpu').items():
                if k.startswith('mod.'):
                    if k == 'mod.cls_token':
                        # special treatment for cls_token
                        # if in shape [1, 2, dim] (two class tokens, one for classification, and one for CLIP contrastive loss), only take the first one
                        if v.shape[1] == 2:
                            init_model_state[k.replace('mod.', '', 1)] = v[:, 0:1]
                    else:
                        init_model_state[k.replace('mod.', '', 1)] = v
                if k.startswith('mod_fc.'):
                    # this only exists if the CLIP model is trained with clip-with-cls mode
                    init_model_state[k.replace('mod_fc.', '', 1)] = v

            missing, unexpected = self.mod.load_state_dict(init_model_state, strict=False)
            _logger.info('Loaded model state from %s: missing keys %s, unexpected keys %s' % (clip_kw['init_path'], missing, unexpected))


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', 'gen.cls_token'}

    def forward(self, *args):
        # return self.mod(features, v=lorentz_vectors, mask=mask) # not using the default foward implementation. Should add emport_embed flag

        if self.clip_mode in ['clip-only', 'clip-with-cls']:
            points, features, lorentz_vectors, mask, gen_points, gen_features, gen_lorentz_vectors, gen_mask = args
            x_mod = self.mod(features, v=lorentz_vectors, mask=mask) # class token output
            x_gen = self.gen(gen_features, v=gen_lorentz_vectors, mask=gen_mask) # class token output

            # FC (for classifications) and projections (for contrastive loss)
            if self.clip_mode == 'clip-with-cls':
                if not self.clip_share_token:
                    # separate class tokens for classification and CLIP contrastive loss
                    assert x_mod.shape[1] == 2, 'Invalid shape %s' % str(x_mod.shape) # dim: (bsz, 2, dim)
                    logits = self.mod_fc(x_mod[:, 0])
                    x_mod = self.mod_proj(x_mod[:, 1])
                    x_gen = self.gen_proj(x_gen)
                else:
                    # use a shared class token for both classification and CLIP contrastive loss
                    assert len(x_mod.shape) == 2, 'Invalid shape %s' % str(x_mod.shape) # dim: (bsz, dim)
                    logits = self.mod_fc(x_mod)
                    x_mod = self.mod_proj(x_mod)
                    x_gen = self.gen_proj(x_gen)

            elif self.clip_mode == 'clip-only':
                logits = None
                x_mod = self.mod_proj(x_mod)
                x_gen = self.gen_proj(x_gen)

        elif self.clip_mode == 'clip-finetune':
            points, features, lorentz_vectors, mask = args
            logits = self.mod(features, v=lorentz_vectors, mask=mask)
            x_mod = None
            x_gen = None
        
        return logits, x_mod, x_gen


class CLIPLoss(torch.nn.Module):
    '''
        Computes the CLIP loss and classification loss
    '''

    def __init__(self, clip_mode=None, beta=1.):
        super().__init__()
        self.clip_mode = clip_mode
        self.beta = beta
        if clip_mode in ['clip-only', 'clip-with-cls']:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, logits, x_mod, x_gen, labels):
        '''
            logits: (batch, out_dim), the output logits of main ParT for standard classification.
            x_mod: (batch, proj_dim), the latent features of main ParT to compute contrastive loss
            x_gen: (batch, proj_dim), the latent features of GEN-level ParT to compute contrastive loss
            labels: (batch,), labels for classification
        '''
        # compute classification loss
        if logits is not None:
            loss_cls = F.cross_entropy(logits, labels)
        else:
            loss_cls = torch.tensor(0., device=labels.device)

        # CLIP constrastive learning
        # normalize the features
        if x_mod is not None:
            x_mod = x_mod / x_mod.norm(dim=-1, keepdim=True)
            x_gen = x_gen / x_gen.norm(dim=-1, keepdim=True)

            # compute cosine similarity
            logit_scale = self.logit_scale.exp()
            logits_cont_2d = logit_scale * x_mod @ x_gen.t() # (batch, batch)
            logits_cont_2d_t = logits_cont_2d.t()
            indices = torch.arange(x_mod.size(0)).to(x_mod.device) # (batch,)

            loss_cont = F.cross_entropy(logits_cont_2d, indices) + F.cross_entropy(logits_cont_2d_t, indices)
        else:
            loss_cont = torch.tensor(0., device=labels.device)

        loss = loss_cls + self.beta * loss_cont
        return loss, {'loss_cls': loss_cls.detach().item(), 'loss_cont': loss_cont.detach().item()}


def get_model(data_config, **kwargs):

    # default configurations
    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=None,
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=True,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
        use_amp=False,
        # gen model kwargs
        gen_model_kw=dict(),
        # clip kwargs
        clip_kw=dict()
    )
    cfg['gen_model_kw'].update(
        input_dim=len(data_config.input_dicts.get('gen_features', [])),
        num_classes=None,
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=True,
        embed_dims=[64, 64, 64],
        pair_embed_dims=[32, 32, 32],
        num_heads=4,
        num_layers=4,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=None,
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg['clip_kw'].update(
        mode='clip-only',
        proj_dim=128,
        share_token=False,
        main_cont_fc_parmas=[],
        gen_cont_fc_parmas=[],
        init_path=None,
        beta=1., # loss weight for contrastive loss
    )

    # update configurations
    for k, v in kwargs.pop('gen_model_kw', dict()).items():
        assert k in cfg['gen_model_kw'], 'Invalid key %s in "gen_model_kw"' % k
        cfg['gen_model_kw'][k] = v
    for k, v in kwargs.pop('clip_kw', dict()).items():
        assert k in cfg['clip_kw'], 'Invalid key %s in "clip_kw"' % k
        cfg['clip_kw'][k] = v
    for k, v in kwargs.items():
        assert k in cfg, 'Invalid key %s' % k
        cfg[k] = v
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerSophonCLIPWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    clip_kw = kwargs.get('clip_kw')
    return CLIPLoss(clip_mode=clip_kw['mode'], beta=clip_kw['beta'])


def get_train_fn(data_config, **kwargs):
    return train_classification_sophon_clip


def get_evaluate_fn(data_config, **kwargs):
    return evaluate_classification_sophon_clip


# Customized training and evaluation functions for Sophon
# functions are adapted from https://github.com/hqucms/weaver-core/blob/main/weaver/utils/nn/tools.py

def train_classification_sophon_clip(
        model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None,
        tb_helper=None, extra_args=None):
    model.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    total_loss_cls = 0
    total_loss_cont = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(dev) # label is obtained from inputs
            entry_count += label.shape[0]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                logits, x_mod, x_gen = model(*inputs)
                loss, loss_dict = loss_func(logits, x_mod, x_gen, label)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            loss = loss.item()

            num_examples = label.shape[0]
            label_counter.update(label.numpy(force=True))
            num_batches += 1
            count += num_examples
            total_loss += loss
            total_loss_cls += loss_dict['loss_cls']
            total_loss_cont += loss_dict['loss_cont']
            tq_dict = {
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'LossCls': '%.5f' % loss_dict['loss_cls'],
                'LossCont': '%.5f' % loss_dict['loss_cont'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
            }

            if logits is not None:
                _, preds = logits.max(1)
                correct = (preds == label).sum().item()
                total_correct += correct
                tq_dict.update({
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count),
                })
            
            tq.set_postfix(tq_dict)

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))
    _logger.info('Max CUDA memory: %.1f MB' % (torch.cuda.max_memory_allocated(dev) / 1024.**2,))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("LossCls/train (epoch)", total_loss_cls / num_batches, epoch),
            ("LossCont/train (epoch)", total_loss_cont / num_batches, epoch),
        ])
        if logits is not None:
            tb_helper.write_scalars([
                ("Acc/train (epoch)", total_correct / count, epoch),
            ])

        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_classification_sophon_clip(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            tb_helper=None, extra_args=None):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    total_loss_cls = 0
    total_loss_cont = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                y = {k: AllGather.apply(v.to(dev)) for k, v in y.items()}
                label = y[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]
                logits, x_mod, x_gen = map(AllGather.apply, model(*inputs))
                if logits is not None:
                    scores.append(torch.softmax(logits.float(), dim=1).numpy(force=True))

                for k, v in y.items():
                    labels[k].append(v.numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v)

                num_examples = label.shape[0]
                label_counter.update(label.numpy(force=True))

                loss, loss_dict = loss_func(logits, x_mod, x_gen, label)
                loss = loss.item()

                num_batches += 1
                count += num_examples
                total_loss += loss * num_examples
                total_loss_cls += loss_dict['loss_cls']
                total_loss_cont += loss_dict['loss_cont']
                tq_dict = {
                    'LossCls': '%.5f' % loss_dict['loss_cls'],
                    'LossCont': '%.5f' % loss_dict['loss_cont'],
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                }

                if logits is not None:
                    _, preds = logits.max(1)
                    correct = (preds == label).sum().item()
                    total_correct += correct
                    tq_dict.update({
                        'Acc': '%.5f' % (correct / num_examples),
                        'AvgAcc': '%.5f' % (total_correct / count),
                    })

                tq.set_postfix(tq_dict)

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("LossCls/%s (epoch)" % tb_mode, total_loss_cls / count, epoch),
            ("LossCont/%s (epoch)" % tb_mode, total_loss_cont / count, epoch),
        ])
        if logits is not None:
            tb_helper.write_scalars([
                ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
            ])

    if logits is not None:
        scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}

    # customized evaluation: making ROC curves for tensorboard monitoring
    if tb_helper and logits is not None:
        scores_dict = {
            'Xbb': scores[:, 0],
            'Xcc': scores[:, 1],
            'QCD': np.sum(scores[:, 161:188], axis=1), # sum of the last 27 scores to form the QCD score
        }
        flag_dict = {
            'Xbb': labels['truth_label'] == 0,
            'Xcc': labels['truth_label'] == 1,
            'QCD': (labels['truth_label'] >= 161) & (labels['truth_label'] < 188),
        }
        comp_list = [('Xbb', 'QCD'), ('Xcc', 'QCD'), ('Xcc', 'Xbb')] # ROC curves for A vs B
        bkgrej = {}

        f, ax = plt.subplots(figsize=(5, 5))
        ax.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), linestyle='--', color='gray', label='Random guess')

        for name_sig, name_bkg in comp_list:
            discr = scores_dict[name_sig] / (scores_dict[name_sig] + scores_dict[name_bkg])
            discr_sig, discr_bkg = discr[flag_dict[name_sig]], discr[flag_dict[name_bkg]]
            fpr, tpr, _ = m.roc_curve(
                np.concatenate([np.ones_like(discr_sig), np.zeros_like(discr_bkg)]),
                np.concatenate([discr_sig, discr_bkg])
            )
            ax.plot(tpr, fpr, label='%s vs %s (AUC=%.4f)' % (name_sig, name_bkg, m.auc(fpr, tpr)))
            bkgrej[(name_sig, name_bkg)] = np.interp(0.3, tpr, 1. / np.maximum(fpr, 1e-10)) # bkgrej at eff_sig=30%
        ax.legend()
        ax.set_xlabel('True positive rate (signal eff.)', ha='right', x=1.0); ax.set_ylabel('False positive rate (BKG eff.)', ha='right', y=1.0)
        ax.set_xlim(0, 1); ax.set_ylim(1e-4, 1), ax.set_yscale('log')

        # write ROC curve figure
        tb_helper.writer.add_figure('ROC/%s/epoch%s' % (tb_mode, str(epoch).zfill(4)), f)

        # write bkgrej values
        for name_sig, name_bkg in comp_list:
            tb_helper.write_scalars([
                ('BkgRej_%s_vs_%s/%s (epoch)' % (name_sig, name_bkg, tb_mode), bkgrej[(name_sig, name_bkg)], epoch),
            ])


    if for_training:
        return total_correct / count
    else:
        raise NotImplementedError('Evaluation metrics are not implemented yet')
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = ak.unflatten(scores, labels_counts)
                for k, v in labels.items():
                    labels[k] = ak.unflatten(v, labels_counts)
            else:
                assert (count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / count, scores, labels, observers
