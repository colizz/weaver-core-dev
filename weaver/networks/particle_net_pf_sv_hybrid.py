import numpy as np
import math
import torch
from torch import Tensor

# from utils.nn.model.ParticleNet import ParticleNetTagger
from weaver.nn.model.ParticleNet import ParticleNetTagger
from weaver.nn.model.ParticleNetJittable import ParticleNetTaggerJittable


def get_model(data_config, jittable=False, **kwargs):
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
        ]
    fc_params = [(256, 0.1)]
    use_fusion = True

    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    num_classes = len(data_config.label_value) + 1 # one dim for regression

    pnet_model = ParticleNetTaggerJittable if jittable else ParticleNetTagger

    model = pnet_model(pf_features_dims, sv_features_dims, num_classes,
                       conv_params, fc_params,
                       use_fusion=use_fusion,
                       use_fts_bn=kwargs.get('use_fts_bn', False),
                       use_counts=kwargs.get('use_counts', True),
                       pf_input_dropout=kwargs.get('pf_input_dropout', 0.0),
                       sv_input_dropout=kwargs.get('sv_input_dropout', 0.0),
                       for_inference=False,
                       )

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['output'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output': {0: 'N'}}},
        }

    return model, model_info

class LogCoshLoss(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(LogCoshLoss, self).__init__(None, None, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        x = input - target
        loss = x + torch.nn.functional.softplus(-2. * x) - math.log(2)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class HybridLoss(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean', gamma=1.) -> None:
        super(HybridLoss, self).__init__(None, None, reduction)
        self.loss_cls_fn = torch.nn.CrossEntropyLoss()
        self.loss_reg_fn = LogCoshLoss()
        self.gamma = gamma

    def forward(self, input_cls: Tensor, input_reg: Tensor, target_cls: Tensor, target_reg: Tensor) -> Tensor:
        loss_cls = self.loss_cls_fn(input_cls, target_cls)
        loss_reg = self.loss_reg_fn(input_reg, target_reg)
        loss = loss_cls + self.gamma * loss_reg
        return loss, {'cls': loss_cls.item(), 'reg': loss_reg.item()}


def get_loss(data_config, **kwargs):
    gamma = kwargs.get('loss_gamma', 1)
    return HybridLoss(gamma=gamma)
