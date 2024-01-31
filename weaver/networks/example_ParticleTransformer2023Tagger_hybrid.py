import os
import math
import torch
from torch import Tensor
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

def get_model(data_config, **kwargs):

    cfg = dict(
        # pf_input_dim=len(data_config.input_dicts['pf_features']),
        # sv_input_dim=len(data_config.input_dicts['sv_features']),
        num_classes=data_config.label_value_cls_num + data_config.label_value_reg_num,
        # network configurations
        pair_input_dim=4,
        pair_extra_dim=0,
        remove_self_pair=False,
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
        enable_mem_efficient=False,
        # misc
        trim=True,
        for_inference=False,
        num_classes_cls=data_config.label_value_cls_num, # for onnx export
    )
    kwargs.pop('loss_gamma')
    kwargs.pop('loss_split_reg')
    norm_pair = kwargs.pop('norm_pair', False)
    if not norm_pair:
        ParT = import_module(os.path.join(os.path.dirname(__file__), 'ParticleTransformer2023.py'), 'ParT')
    else:
        ParT = import_module(os.path.join(os.path.dirname(__file__), 'ParticleTransformer2023_normpair.py'), 'ParT')
        cfg['pair_input_dim'] = 6

    three_coll = kwargs.pop('three_coll', False)
    if not three_coll:
        cfg.update(dict(
            pf_input_dim=len(data_config.input_dicts['pf_features']),
            sv_input_dim=len(data_config.input_dicts['sv_features']),
        ))
        cfg.update(**kwargs)
        model = ParT.ParticleTransformerTagger(**cfg)
    else:
        cfg.update(dict(
            cpf_input_dim=len(data_config.input_dicts['cpf_features']),
            npf_input_dim=len(data_config.input_dicts['npf_features']),
            sv_input_dim=len(data_config.input_dicts['sv_features']),
        ))
        cfg.update(**kwargs)
        model = ParT.ParticleTransformerTagger_3coll(**cfg)
    
    _logger.info('Model config: %s' % str(cfg))

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
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
        super(HybridLoss, self).__init__()
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


def get_loss(data_config, **kwargs):
    gamma = kwargs.get('loss_gamma', 1)
    split_reg = kwargs.get('loss_split_reg', True)
    return HybridLoss(split_reg=split_reg, gamma=gamma)
