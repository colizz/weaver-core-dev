import os
import math
import torch
import torch.nn as nn
from torch import Tensor
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

ParticleTransformerTagger_ncoll = import_module(os.path.join(os.path.dirname(__file__), 'ParticleTransformer2024Plus.py'), 'ParT').ParticleTransformerTagger_ncoll

def get_model(data_config, **kwargs):

    # use SwiGLU-default setup
    cfg = dict(
        input_dims=tuple(map(lambda x: len(data_config.input_dicts[x]), ['cpf_features', 'npf_features', 'sv_features'])),
        share_embed=False,
        num_classes=data_config.label_value_cls_num + data_config.label_value_reg_num,
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
        # misc
        trim=True,
        for_inference=False,
    )

    kwargs.pop('loss_gamma')
    kwargs.pop('loss_split_reg', False)
    kwargs.pop('three_coll', False) # v2 setup
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
    model = ParticleTransformerTagger_ncoll(**cfg)
    
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
    split_reg = kwargs.get('loss_split_reg', False)
    return HybridLoss(split_reg=split_reg, gamma=gamma)


def get_save_fn(data_config, **kwargs):
    return save_custom


def save_custom(args, data_config, scores, labels, observers):
    assert args.train_mode == 'hybrid'
    selected_labels = [
        "label_Top_bWcs", "label_Top_bWqq", "label_Top_bWc", "label_Top_bWs", "label_Top_bWq", "label_Top_bWev", "label_Top_bWmv", "label_Top_bWtauev", "label_Top_bWtaumv", "label_Top_bWtauhv", "label_Top_Wcs", "label_Top_Wqq", "label_Top_Wev", "label_Top_Wmv", "label_Top_Wtauev", "label_Top_Wtaumv", "label_Top_Wtauhv", "label_H_bb", "label_H_cc", "label_H_ss", "label_H_qq", "label_H_bc", "label_H_cs", "label_H_gg", "label_H_ee", "label_H_mm", "label_H_tauhtaue", "label_H_tauhtaum", "label_H_tauhtauh", "label_H_WW_cscs", "label_H_WW_csqq", "label_H_WW_qqqq", "label_H_WW_csc", "label_H_WW_css", "label_H_WW_csq", "label_H_WW_qqc", "label_H_WW_qqs", "label_H_WW_qqq", "label_H_WW_csev", "label_H_WW_qqev", "label_H_WW_csmv", "label_H_WW_qqmv", "label_H_WW_cstauev", "label_H_WW_qqtauev", "label_H_WW_cstaumv", "label_H_WW_qqtaumv", "label_H_WW_cstauhv", "label_H_WW_qqtauhv", 
        "label_QCD_bb", "label_QCD_cc", "label_QCD_b", "label_QCD_c", "label_QCD_others"
        ]

    output = {}
    scores_cls, scores_reg = scores
    # write regression nodes
    for idx in range(1, len(data_config.label_names)):
        name = data_config.label_names[idx]
        output[name] = labels[name]
        if not data_config.split_per_cls:
            output['output_' + name] = scores_reg[:, idx-1]
        else:
            for idx_cls, label_name in enumerate(data_config.label_value_cls_names):
                if label_name not in selected_labels:
                    continue
                output['output_' + name + '_' + label_name] = scores_reg[:, (idx-1) * data_config.label_value_cls_num + idx_cls]
    # write classification nodes
    output['cls_index'] = labels['_label_'] # classes can be too many, only store the index
    for idx, label_name in enumerate(data_config.label_value_cls_names):
        if label_name not in selected_labels:
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
