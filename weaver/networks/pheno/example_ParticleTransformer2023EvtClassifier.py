import os
import torch
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

ParticleTransformerTagger_ncoll = import_module(
    os.path.join(os.path.dirname(__file__), 'ParticleTransformer2023.py'), 'ParT').ParticleTransformerTagger_ncoll


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformerTagger_ncoll(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, *args):
        return self.mod(*args)


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dims=None,
        num_classes=len(data_config.label_value),
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
        num_classes_cls=len(data_config.label_value), # for onnx export
    )
    if all([k in data_config.input_dicts for k in ['evt_features', 'jet_features', 'lep_features', 'pho_features']]):
        cfg['input_dims'] = (len(data_config.input_dicts['evt_features']), len(data_config.input_dicts['jet_features']), len(data_config.input_dicts['lep_features']), len(data_config.input_dicts['pho_features']))
    
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
