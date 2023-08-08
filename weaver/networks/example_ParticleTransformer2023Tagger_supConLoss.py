import os
import math
import torch
from torch import Tensor
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

ParT = import_module(os.path.join(os.path.dirname(__file__), 'ParticleTransformer2023.py'), 'ParT')


def get_model(data_config, **kwargs):

    cfg = dict(
        # pf_input_dim=len(data_config.input_dicts['pf_features']),
        # sv_input_dim=len(data_config.input_dicts['sv_features']),
        num_classes=128, ## manually defined to calculate contrastive loss!
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
    )
    kwargs.pop('temperature')
    kwargs.pop('base_temperature')

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


SupConLoss = import_module(os.path.join(os.path.dirname(__file__), 'sup_contrast_loss.py'), 'sup_contrast_loss').SupConLoss


def get_loss(data_config, **kwargs):
    temperature = kwargs.pop('temperature', 0.07)
    base_temperature = kwargs.pop('base_temperature', 0.07)
    return SupConLoss(
        contrast_mode='one', temperature=temperature, base_temperature=base_temperature
        )
