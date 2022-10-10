import os
import math
import torch
from torch import Tensor
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

ParticleTransformerTagger = import_module(
    os.path.join(os.path.dirname(__file__), 'ParticleTransformer.py'), 'ParT').ParticleTransformerTagger


def get_model(data_config, **kwargs):

    cfg = dict(
        pf_input_dim=len(data_config.input_dicts['pf_features']),
        sv_input_dim=len(data_config.input_dicts['sv_features']),
        num_classes=len(data_config.label_value) + 1, # one dim for regression
        # network configurations
        pair_input_dim=4,
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
    )
    kwargs.pop('loss_gamma')
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerTagger(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
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
