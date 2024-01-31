import os
import math
import torch
from torch import Tensor
from torch import nn
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

class ParticleTransformerFC(nn.Module):

    def __init__(self,
                 num_classes=None,
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
                 num_classes_cls=None, # for onnx export
                 use_amp=False,
                 return_embed=False,
                 export_embed=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.for_inference = for_inference
        self.num_classes_cls = num_classes_cls
        self.use_amp = use_amp
        self.return_embed = return_embed

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else None

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }
    def forward(self, x):
        # x: ft vars in 2D branch, dim: (N, 1, C)
        x_cls = x.squeeze(1)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # fc
            output = self.fc(x_cls)
            if self.for_inference:
                # for onnx export
                assert self.num_classes_cls is not None
                output_cls, output_reg = output.split([self.num_classes_cls, output.size(1) - self.num_classes_cls], dim=1)
                output_cls = torch.softmax(output_cls, dim=1)
                output = torch.cat([output_cls, output_reg], dim=-1)

            # print('output:\n', output)
            if self.return_embed == False:
                return output
            else:
                return output, x_cls


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
    norm_pair = kwargs.pop('norm_pair', False)
    if norm_pair:
        cfg['pair_input_dim'] = 6
    kwargs.pop('three_coll')

    cfg.update(**kwargs)
    model = ParticleTransformerFC(**cfg)
    
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

    def __init__(self, reduction: str = 'mean') -> None:
        super(LogCoshLoss, self).__init__(None, None, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        x = input - target
        loss = x + torch.nn.functional.softplus(-2. * x) - math.log(2)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean(dim=0)
        elif self.reduction == 'sum':
            return loss.sum(dim=0)


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
        loss = loss_cls + self.gamma * loss_reg.sum()
        loss_dict = {'cls': loss_cls.item(), 'reg': loss_reg.sum().item()}
        loss_dict.update({f'reg_{i}': loss_reg[i].item() for i in range(loss_reg.shape[0])})
        return loss, loss_dict


def get_loss(data_config, **kwargs):
    gamma = kwargs.get('loss_gamma', 1)
    return HybridLoss(gamma=gamma)
