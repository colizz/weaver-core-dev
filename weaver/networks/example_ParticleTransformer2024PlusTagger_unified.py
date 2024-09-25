import os
import math
import torch
import torch.nn as nn
from torch import Tensor
import tqdm
import time
from collections import defaultdict, Counter
import numpy as np

from weaver.utils.logger import _logger
from weaver.utils.nn.tools import (
    evaluate_metrics,
    _flatten_preds,
    _flatten_label,
    _concat
)
from weaver.utils.import_tools import import_module

ParticleTransformerTagger_ncoll = import_module(os.path.join(os.path.dirname(__file__), 'ParticleTransformer2024Plus.py'), 'ParT').ParticleTransformerTagger_ncoll


class GlobalParticleTransformerWrapper(ParticleTransformerTagger_ncoll):
    def __init__(self, *args, **kwargs) -> None:

        use_external_fc = kwargs.pop('use_external_fc', False)
        freeze_main_params = kwargs.pop('freeze_main_params', False)
        self.input_highlevel_dim = kwargs.pop('input_highlevel_dim', 0)
        self.for_inference = kwargs['for_inference']

        if use_external_fc:
            fc_params = kwargs.get('fc_params', None)
            kwargs['fc_params'] = None  # remove the default FC layer so that the ParT model will just output the last embed layer
            super().__init__(*args, **kwargs)
            
            # when doing transfer learning (freezing the entire ParT), 
            # it is crucial to freeze the running stats of all batch norm layers
            if freeze_main_params:
                for params in self.parameters():
                    params.requires_grad_(False)
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        m.requires_grad_(False)
                        m.track_running_stats = False

            if fc_params is not None:
                fcs = []
                in_dim = kwargs['embed_dims'][-1] + self.input_highlevel_dim # concat high-level input dims to the embed layer
                for out_dim, drop_rate in fc_params:
                    fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                    in_dim = out_dim
                fcs.append(nn.Linear(in_dim, kwargs['num_classes']))
                self.fc = nn.Sequential(*fcs)
            else:
                self.fc = None

        else:
            super().__init__(*args, **kwargs)
            self.fc = None
    
    def forward(self, *args):
        if self.fc is not None:
            if self.input_highlevel_dim > 0:
                x = super().forward(*args[:-1])
                x = torch.cat([x, args[-1].squeeze(2)], dim=1)
            else:
                x = super().forward(*args)
            output = self.fc(x)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            return output
        else:
            return super().forward(*args)


def get_model(data_config, **kwargs):
    assert 'num_nodes' in kwargs, 'num_nodes must be provided'
    assert 'num_cls_nodes' in kwargs, 'num_cls_nodes must be provided'
    num_nodes = kwargs.pop('num_nodes')
    num_cls_nodes = kwargs.pop('num_cls_nodes')
    label_cls_nodes = kwargs.pop('label_cls_nodes', None)
    eval_kw = kwargs.pop('eval_kw', dict())

    # use SwiGLU-default setup
    cfg = dict(
        input_dims=tuple(map(lambda x: len(data_config.input_dicts[x]), ['cpf_features', 'npf_features', 'sv_features'])),
        share_embed=False,
        num_classes=num_nodes,
        # network configurations
        pair_input_type='pp',
        pair_input_dim=4,
        pair_extra_dim=0,
        use_pair_norm=False,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={},
        fc_params=(),
        activation='gelu',
        # GloParT wrapper configurations
        input_highlevel_dim=len(data_config.input_dicts.get('jet_features', [])),
        use_external_fc=False,
        # misc
        trim=True,
        for_inference=False,
    )

    kwargs.pop('loss_gamma') # must-have argument
    kwargs.pop('loss_split_reg', False)
    kwargs.pop('loss_composed_split_reg', None)
    kwargs.pop('three_coll', False) # v2 setup, not used for v3
    if kwargs.pop('use_swiglu_config', False):
        cfg.update(
            block_params={"scale_attn_mask": True, "scale_attn": False, "scale_fc": False, "scale_heads": False, "scale_resids": False, "activation": "swiglu"},
            cls_block_params={"scale_attn": False, "scale_fc": False, "scale_heads": False, "scale_resids": False, "activation": "swiglu"},
        )
    if kwargs.pop('use_pair_norm_config', False):
        cfg.update(
            use_pair_norm=True,
            pair_input_dim=6,
        )

    cfg.update(**kwargs)
    model = GlobalParticleTransformerWrapper(**cfg)
    # set special args
    model.num_nodes = num_nodes
    model.num_cls_nodes = num_cls_nodes
    model.eval_kw = eval_kw

    _logger.info('Model config: %s' % str(cfg))

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['output'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output': {0: 'N'}}},
    }

    return model, model_info


class LogCoshLoss(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction='mean', split_reg=False):
        super(LogCoshLoss, self).__init__(None, None, reduction)
        self.split_reg = split_reg

    def forward(self, input, target_reg, target_cls=None, n_cls=None):

        if not self.split_reg:
            x = input - target_reg # dim: (B, n_reg)
            loss = x + torch.nn.functional.softplus(-2. * x) - math.log(2)
        
        else:
            # calculate regression loss for each class separately
            # input: (N, C * n_reg), target_reg: (N, n_reg), target_cls: (N)
            n_reg = target_reg.shape[1]
            input = input.view(-1, n_cls, n_reg)
            target_reg = target_reg.view(-1, 1, n_reg)
            target_cls = torch.nn.functional.one_hot(target_cls, num_classes=n_cls).bool().view(-1, n_cls, 1)
            x = input - target_reg # dim: (B, n_cls, n_reg)
            loss = x + torch.nn.functional.softplus(-2. * x) - math.log(2)
            loss = (loss * target_cls).sum(dim=1) # dim: (B, n_reg)
        
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean(dim=0)
        elif self.reduction == 'sum':
            return loss.sum(dim=0)


class HybridLoss(torch.nn.Module):

    def __init__(self, reduction='mean', split_reg=False, gamma=1.):
        super().__init__()
        self.loss_cls_fn = torch.nn.CrossEntropyLoss()
        self.loss_reg_fn = LogCoshLoss(reduction=reduction, split_reg=split_reg)
        self.gamma = gamma

    def forward(self, input_cls, input_reg, target_cls, target_reg):

        loss_cls = self.loss_cls_fn(input_cls, target_cls)
        loss_reg = self.loss_reg_fn(input_reg, target_reg, target_cls=target_cls, n_cls=input_cls.shape[1])
        loss = loss_cls + self.gamma * loss_reg.sum()
        loss_dict = {'cls': loss_cls.item(), 'reg': loss_reg.sum().item()}
        loss_dict.update({f'reg_{i}': loss_reg[i].item() for i in range(loss_reg.shape[0])})
        return loss, loss_dict


class ComposedHybridLoss(torch.nn.Module):

    def __init__(self, reduction='mean', composed_split_reg=None, gamma=1.):
        unifd, split = composed_split_reg
        # unifd, split: lists of True/False for each regression target. True means using this regression target
        assert any(unifd) and any(split), 'At least one regression target must be used for unified and split regression'

        super().__init__()
        self.loss_cls_fn = torch.nn.CrossEntropyLoss()
        self.loss_reg_unifd_fn = LogCoshLoss(reduction=reduction, split_reg=False)
        self.loss_reg_split_fn = LogCoshLoss(reduction=reduction, split_reg=True)
        self.unifd = unifd
        self.split = split
        self.num_unifd = sum(unifd)
        self.gamma = gamma

    def forward(self, input_cls, input_reg, target_cls, target_reg):

        loss_cls = self.loss_cls_fn(input_cls, target_cls)

        # compute unified regression loss
        n_target_reg = target_reg.shape[1]
        loss_reg_unifd = self.loss_reg_unifd_fn(input_reg[:, :self.num_unifd], target_reg[:, self.unifd])

        # compute split-class regression loss. Only do split-class regression for specific targets defined by composed_split_reg
        loss_reg_split = self.loss_reg_split_fn(input_reg[:, self.num_unifd:], target_reg[:, self.split], target_cls=target_cls, n_cls=input_cls.shape[1])
        loss = loss_cls + self.gamma * (loss_reg_unifd.sum() + loss_reg_split.sum())
        loss_dict = {'cls': loss_cls.item(), 'reg_unifd': loss_reg_unifd.sum().item(), 'reg_split': loss_reg_split.sum().item(), 'reg': loss_reg_unifd.sum().item() + loss_reg_split.sum().item()}
        return loss, loss_dict


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def forward(self, input_cls, input_reg, target_cls, target_reg):
        loss = super().forward(input_cls, target_cls)
        return loss, {'cls': loss.item(), 'reg': 0.}


def get_loss(data_config, **kwargs):
    gamma = kwargs.get('loss_gamma', 1)
    split_reg = kwargs.get('loss_split_reg', False)
    composed_split_reg = kwargs.get('loss_composed_split_reg', None)
    if gamma == 0:
        return CrossEntropyLoss()
    else:
        if composed_split_reg is None:
            return HybridLoss(split_reg=split_reg, gamma=gamma)
        else:
            return ComposedHybridLoss(composed_split_reg=composed_split_reg, gamma=gamma)


def get_train_fn(data_config, **kwargs):
    return train_hybrid


def get_evaluate_fn(data_config, **kwargs):
    return evaluate_hybrid


def get_save_fn(data_config, **kwargs):
    return save_custom


#### ================== Custom train/eval/save functions ================== ####

def train_hybrid(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, train_loss=None, tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    total_loss_cls = 0
    total_loss_reg = 0
    total_loss_reg_i = defaultdict(float)
    total_loss_reg_split = 0
    total_loss_reg_unifd = 0
    num_batches = 0
    total_correct = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            # for classification
            label_cls = y['truth_label'].long() # _label_ -> truth_label
            label_counter.update(label_cls.cpu().numpy())
            label_cls = label_cls.to(dev)
            n_cls = model.module.num_cls_nodes if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.num_cls_nodes
            num_examples = label_cls.shape[0]

            # for regression
            if len(data_config.label_names) > 1:
                label_reg = [y[n].float().to(dev).unsqueeze(1) for n in data_config.label_names[1:]] # can support multiple regression target
                label_reg = torch.cat(label_reg, dim=1)
            else:
                label_reg = None
            n_reg_target = len(data_config.label_names) - 1

            opt.zero_grad()
            # with torch.autograd.detect_anomaly():
            with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
                model_output = model(*inputs)
                logits = model_output[:, :n_cls]
                preds_reg = model_output[:, n_cls:]
                loss, loss_monitor = loss_func(logits, preds_reg, label_cls, label_reg)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            _, preds_cls = logits.max(1)
            loss = loss.item()

            num_batches += 1
            count += num_examples
            correct = (preds_cls == label_cls).sum().item()
 
            total_loss += loss
            total_loss_cls += loss_monitor['cls']
            total_loss_reg += loss_monitor['reg']
            if 'reg_split' in loss_monitor:
                total_loss_reg_split += loss_monitor['reg_split']
                total_loss_reg_unifd += loss_monitor['reg_unifd']
            elif n_reg_target > 1:
                for i in range(n_reg_target):
                    total_loss_reg_i[i] += loss_monitor[f'reg_{i}']
            total_correct += correct

            ## not adapted for composed regression loss
            # if len(data_config.label_names) > 1:
            #     e = preds_reg - label_reg
            #     abs_err = e.abs().sum().item()
            #     sum_abs_err += abs_err
            #     sqr_err = e.square().sum().item()
            #     sum_sqr_err += sqr_err
            # else:
            #     abs_err = 0
            #     sqr_err = 0

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss_monitor['cls'],
                'LossReg': '%.5f' % loss_monitor['reg'],
                'LossTot': '%.5f' % loss,
                # 'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                # 'AvgAcc': '%.5f' % (total_correct / count),
                # 'MSE': '%.5f' % (sqr_err / num_examples),
                # 'AvgMSE': '%.5f' % (sum_sqr_err / count),
                # 'MAE': '%.5f' % (abs_err / num_examples),
                # 'AvgMAE': '%.5f' % (sum_abs_err / count),
            })

            # stop writing to tensorboard after 500 batches
            if tb_helper and num_batches < 500:
                tb_helper.write_scalars([
                    ("Loss/train", loss_monitor['cls'], tb_helper.batch_train_count + num_batches), # to compare cls loss to previous loss
                    ("LossReg/train", loss_monitor['reg'], tb_helper.batch_train_count + num_batches),
                    # ("LossTot/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                    # ("MSE/train", sqr_err / num_examples, tb_helper.batch_train_count + num_batches),
                    # ("MAE/train", abs_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ])
                if 'reg_split' in loss_monitor:
                    tb_helper.write_scalars([
                        ("LossRegSplit/train", loss_monitor['reg_split'], tb_helper.batch_train_count + num_batches),
                        ("LossRegUnifd/train", loss_monitor['reg_unifd'], tb_helper.batch_train_count + num_batches),
                        ])
                elif n_reg_target > 1:
                    for i in range(n_reg_target):
                        tb_helper.write_scalars([
                            (f"LossReg{i}/train", loss_monitor[f'reg_{i}'], tb_helper.batch_train_count + num_batches),
                            ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgLossReg: %.5f, AvgLossTot: %.5f, AvgAcc: %.5f' %
                 (total_loss_cls / num_batches, total_loss_reg / num_batches, total_loss / num_batches,
                 total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss_cls / num_batches, epoch), # to compare cls loss to previous loss
            ("LossReg/train (epoch)", total_loss_reg / num_batches, epoch),
            ("LossTot/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
            # ("MSE/train (epoch)", sum_sqr_err / count, epoch),
            # ("MAE/train (epoch)", sum_abs_err / count, epoch),
            ])
        if 'reg_split' in loss_monitor:
            tb_helper.write_scalars([
                ("LossRegSplit/train (epoch)", total_loss_reg_split / num_batches, epoch),
                ("LossRegUnifd/train (epoch)", total_loss_reg_unifd / num_batches, epoch),
                ])
        elif n_reg_target > 1:
            for i in range(n_reg_target):
                tb_helper.write_scalars([
                    (f"LossReg{i}/train (epoch)", total_loss_reg_i[i] / num_batches, epoch),
                    ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_hybrid(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                        eval_metrics_cls=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                        eval_metrics_reg=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                                          'mean_gamma_deviance'],
                        tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    total_loss_cls = 0
    total_loss_reg = 0
    total_loss_reg_i = defaultdict(float)
    total_loss_reg_split = 0
    total_loss_reg_unifd = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    count = 0
    scores_cls = []
    scores_reg = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    start_time = time.time()
    model_embed_output_array = []
    label_cls_array = []
    eval_kw = model.module.eval_kw \
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.eval_kw
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                # for classification
                label_cls = y['truth_label'].long() # _label_ -> truth_label
                entry_count += label_cls.shape[0]
                num_examples = label_cls.shape[0]
                label_counter.update(label_cls.cpu().numpy())
                label_cls = label_cls.to(dev)
                n_cls = model.module.num_cls_nodes if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.num_cls_nodes

                # for regression
                if len(data_config.label_names) > 1:
                    label_reg = [y[n].float().to(dev).unsqueeze(1) for n in data_config.label_names[1:]]
                    label_reg = torch.cat(label_reg, dim=1)
                else:
                    label_reg = None
                n_reg_target = len(data_config.label_names) - 1

                # with torch.autograd.detect_anomaly():
                model_output = model(*inputs)
                # ## a temporary hack: save the embeded space
                # model_output, model_embed_output = model(*inputs, return_embed=True)
                # model_embed_output_array.append(model_embed_output.detach().cpu().numpy())
                # label_cls_array.append(label_cls.detach().cpu().numpy())

                logits = model_output[:, :n_cls].float()
                preds_reg = model_output[:, n_cls:].float()

                if not for_training:
                    scores_cls.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                    scores_reg.append(preds_reg.detach().cpu().numpy())
                    for k, v in y.items():
                        labels[k].append(v.cpu().numpy())
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.cpu().numpy())
                if for_training and eval_kw.get('roc_kw', None):
                    # for making ROC curves
                    scores_cls.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                    labels['truth_label'].append(y['truth_label'].cpu().numpy())

                _, preds_cls = logits.max(1)
                if for_training:
                    loss, loss_monitor = loss_func(logits, preds_reg, label_cls, label_reg)
                    loss = loss.item()
                else:
                    loss, loss_monitor = 0., {'cls': 0., 'reg': 0.}

                num_batches += 1
                count += num_examples
                correct = (preds_cls == label_cls).sum().item()
                total_correct += correct
                total_loss += loss * num_examples
                total_loss_cls += loss_monitor['cls'] * num_examples
                total_loss_reg += loss_monitor['reg'] * num_examples
                if 'reg_split' in loss_monitor:
                    total_loss_reg_split += loss_monitor['reg_split'] * num_examples
                    total_loss_reg_unifd += loss_monitor['reg_unifd'] * num_examples
                elif n_reg_target > 1 and for_training:
                    for i in range(n_reg_target):
                        total_loss_reg_i[i] += loss_monitor[f'reg_{i}'] * num_examples

                ## not adapted for composed regression loss
                # if len(data_config.label_names) > 1:
                #     e = preds_reg - label_reg
                #     abs_err = e.abs().sum().item()
                #     sum_abs_err += abs_err
                #     sqr_err = e.square().sum().item()
                #     sum_sqr_err += sqr_err
                # else:
                #     abs_err = 0
                #     sqr_err = 0

                tq.set_postfix({
                    'Loss': '%.5f' % loss_monitor['cls'],
                    'LossReg': '%.5f' % loss_monitor['reg'],
                    'LossTot': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count),
                    # 'MSE': '%.5f' % (sqr_err / num_examples),
                    # 'AvgMSE': '%.5f' % (sum_sqr_err / count),
                    # 'MAE': '%.5f' % (abs_err / num_examples),
                    # 'AvgMAE': '%.5f' % (sum_abs_err / count),
                })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss_cls / count, epoch),
            ("LossReg/%s (epoch)" % tb_mode, total_loss_reg / count, epoch),
            ("LossTot/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
            # ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
            # ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
            ])
        if 'reg_split' in loss_monitor:
            tb_helper.write_scalars([
                ("LossRegSplit/%s (epoch)" % tb_mode, total_loss_reg_split / count, epoch),
                ("LossRegUnifd/%s (epoch)" % tb_mode, total_loss_reg_unifd / count, epoch),
                ])
        elif n_reg_target > 1:
            for i in range(n_reg_target):
                tb_helper.write_scalars([
                    (f"LossReg{i}/{tb_mode} (epoch)", total_loss_reg_i[i] / count, epoch),
                    ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)
    ## a temporary hack: save the embeded space
    # tb_helper.writer.add_embedding(np.concatenate(model_embed_output_array), metadata=[data_config.label_value_cls_names[val].replace('label_','') for val in np.concatenate(label_cls_array)], tag='embed')

    # customized evaluation: making ROC curves for tensorboard monitoring
    if tb_helper and for_training and eval_kw.get('roc_kw', None):
        roc_kwargs = eval_kw['roc_kw']
        scores_cls = np.concatenate(scores_cls)
        truth_label = np.concatenate(labels['truth_label'])
        scores_dict, flag_dict = {}, {}
        for name, inds in roc_kwargs.get('label_inds_map').items():
            flag_dict[name] = np.any([truth_label == i for i in inds], axis=0)
            scores_dict[name] = np.sum(scores_cls[:, inds], axis=1)
            print(name, flag_dict[name].shape, scores_dict[name].shape)
        comp_list = roc_kwargs.get('comp_list') # e.g. [('Xbb', 'QCD'), ('Xcc', 'QCD'), ('Xcc', 'Xbb')] # ROC curves for A vs B
        bkgrej = {}

        import matplotlib.pyplot as plt
        import sklearn.metrics as m
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
        ax.set_xlabel('Signal eff.', ha='right', x=1.0); ax.set_ylabel('BKG eff.', ha='right', y=1.0)
        ax.set_xlim(0, 1); ax.set_ylim(1e-4, 1), ax.set_yscale('log')

        # write ROC curve figure
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.writer.add_figure('ROC/%s (epoch)' % tb_mode, f, epoch)

        # write bkgrej values
        for name_sig, name_bkg in comp_list:
            tb_helper.write_scalars([
                ('BkgRej_%s_vs_%s/%s (epoch)' % (name_sig, name_bkg, tb_mode), bkgrej[(name_sig, name_bkg)], epoch),
            ])

    if not for_training:
        scores_cls = np.concatenate(scores_cls)
        scores_reg = np.concatenate(scores_reg)
        labels = {k: _concat(v) for k, v in labels.items()}
        metric_results_cls = evaluate_metrics(labels['truth_label'], scores_cls, eval_metrics=eval_metrics_cls)
        _logger.info('Evaluation metric for cls: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results_cls.items()]))

    if for_training:
        return total_loss / count
    else:
        # convert 2D labels/scores
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_loss / count, (scores_cls, scores_reg), labels, observers


def save_custom(args, data_config, scores, labels, observers):
    import ast
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    split_reg = network_options.get('loss_split_reg', False)
    composed_split_reg = network_options.get('loss_composed_split_reg', None)
    label_cls_nodes = network_options.get('label_cls_nodes', None)
    label_stored = network_options.get('label_stored', None)
    assert label_cls_nodes is not None, 'label_cls_nodes must be provided as a network option in the test mode'

    if label_stored is None:
        label_stored = [
            "label_Top_bWcs", "label_Top_bWqq", "label_Top_bWc", "label_Top_bWs", "label_Top_bWq", "label_Top_bWev", "label_Top_bWmv", "label_Top_bWtauev", "label_Top_bWtaumv", "label_Top_bWtauhv", "label_Top_Wcs", "label_Top_Wqq", "label_Top_Wev", "label_Top_Wmv", "label_Top_Wtauev", "label_Top_Wtaumv", "label_Top_Wtauhv",
            "label_Top_bWpcs", "label_Top_bWp'pqq", "label_Top_bWpc", "label_Top_bWps", "label_Top_bWpq", "label_Top_bWpev", "label_Top_bWpmv", "label_Top_bWptauev", "label_Top_bWptaumv", "label_Top_bWptauhv", "label_Top_Wpcs", "label_Top_Wpqq", "label_Top_Wpev", "label_Top_Wpmv", "label_Top_Wptauev", "label_Top_Wptaumv", "label_Top_Wptauhv",
            "label_Top_bWmcs", "label_Top_bWm'pqq", "label_Top_bWmc", "label_Top_bWms", "label_Top_bWmq", "label_Top_bWmev", "label_Top_bWmmv", "label_Top_bWmtauev", "label_Top_bWmtaumv", "label_Top_bWmtauhv", "label_Top_Wmcs", "label_Top_Wmqq", "label_Top_Wmev", "label_Top_Wmmv", "label_Top_Wmtauev", "label_Top_Wmtaumv", "label_Top_Wmtauhv",
            "label_H_bb", "label_H_cc", "label_H_ss", "label_H_qq", "label_H_bc", "label_Hp_bc", "label_Hm_bc", "label_H_bs", "label_H_cs", "label_Hp_cs", "label_Hm_cs", "label_H_gg", "label_H_aa", "label_H_ee", "label_H_mm", "label_H_tauhtaue", "label_H_tauhtaum", "label_H_tauhtauh", "label_H_WW_cscs", "label_H_WW_csqq", "label_H_WW_qqqq", "label_H_WW_csc", "label_H_WW_css", "label_H_WW_csq", "label_H_WW_qqc", "label_H_WW_qqs", "label_H_WW_qqq", "label_H_WW_csev", "label_H_WW_qqev", "label_H_WW_csmv", "label_H_WW_qqmv", "label_H_WW_cstauev", "label_H_WW_qqtauev", "label_H_WW_cstaumv", "label_H_WW_qqtaumv", "label_H_WW_cstauhv", "label_H_WW_qqtauhv", 
            "label_H_WxWx_cscs", "label_H_WxWx_csqq", "label_H_WxWx_qqqq", "label_H_WxWx_csc", "label_H_WxWx_css", "label_H_WxWx_csq", "label_H_WxWx_qqc", "label_H_WxWx_qqs", "label_H_WxWx_qqq", "label_H_WxWx_csev", "label_H_WxWx_qqev", "label_H_WxWx_csmv", "label_H_WxWx_qqmv", "label_H_WxWx_cstauev", "label_H_WxWx_qqtauev", "label_H_WxWx_cstaumv", "label_H_WxWx_qqtaumv", "label_H_WxWx_cstauhv", "label_H_WxWx_qqtauhv", 
            "label_H_WxWxStar_cscs", "label_H_WxWxStar_csqq", "label_H_WxWxStar_qqqq", "label_H_WxWxStar_csc", "label_H_WxWxStar_css", "label_H_WxWxStar_csq", "label_H_WxWxStar_qqc", "label_H_WxWxStar_qqs", "label_H_WxWxStar_qqq", "label_H_WxWxStar_csev", "label_H_WxWxStar_qqev", "label_H_WxWxStar_csmv", "label_H_WxWxStar_qqmv", "label_H_WxWxStar_cstauev", "label_H_WxWxStar_qqtauev", "label_H_WxWxStar_cstaumv", "label_H_WxWxStar_qqtaumv", "label_H_WxWxStar_cstauhv", "label_H_WxWxStar_qqtauhv", 
            "label_QCD_bb", "label_QCD_cc", "label_QCD_b", "label_QCD_c", "label_QCD_others"
            ]

    output = {}
    scores_cls, scores_reg = scores
    assert scores_cls.shape[1] == len(label_cls_nodes), 'Number of classification nodes does not match'

    # write regression nodes
    if len(data_config.label_names) > 1:
        if composed_split_reg is not None:
            idx_reg = 0
            # write unified regression nodes
            for idx in range(1, len(data_config.label_names)):
                name = data_config.label_names[idx]
                if composed_split_reg[0][idx-1]: # do unified regression
                    print('write unified regression nodes:', name)
                    output[name] = labels[name]
                    output['output_' + name] = scores_reg[:, idx_reg]
                    idx_reg += 1
            # write split regression nodes
            for idx in range(1, len(data_config.label_names)):
                name = data_config.label_names[idx]
                if composed_split_reg[1][idx-1]: # do split regression
                    print('write split regression nodes:', name)
                    if name not in output:
                        output[name] = labels[name]
                    for idx_cls, label_name in enumerate(label_cls_nodes):
                        if label_name not in label_stored:
                            continue
                        output['output_' + name + '_' + label_name] = scores_reg[:, idx_reg + idx_cls]
                    idx_reg += len(label_cls_nodes)

        elif split_reg:
            # write split regression nodes
            for idx in range(1, len(data_config.label_names)):
                name = data_config.label_names[idx]
                output[name] = labels[name]
                for idx_cls, label_name in enumerate(label_cls_nodes):
                    if label_name not in label_stored:
                        continue
                    output['output_' + name + '_' + label_name] = scores_reg[:, (idx-1) * len(label_cls_nodes) + idx_cls]
        else:
            # write normal (unified) regression nodes
            for idx in range(1, len(data_config.label_names)):
                name = data_config.label_names[idx]
                output[name] = labels[name]
                output['output_' + name] = scores_reg[:, idx-1]

    # write classification nodes
    output['cls_index'] = labels['truth_label'] # classes can be too many, only store the index
    for idx, label_name in enumerate(label_cls_nodes):
        if label_name not in label_stored:
            continue
        output['score_' + label_name] = scores_cls[:, idx]

    for k, v in labels.items():
        if k == data_config.label_names[0]:
            continue
        assert v.ndim == 1
        output[k] = v

    for k, v in observers.items():
        assert v.ndim == 1
        output[k] = v

    return output
