import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import os
import copy
from weaver.utils.import_tools import import_module

ParticleTransformer = import_module(
    os.path.join(os.path.dirname(__file__), '../', 'ParticleTransformer2023.py'), 'ParT').ParticleTransformer

MultiLayerPerceptron = import_module(os.path.join(os.path.dirname(__file__), 'mlp.py'), 'mlp').MultiLayerPerceptron

class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


class ModelWithPretrainedParT(nn.Module):

    def __init__(self, mlp, part):
        super(ModelWithPretrainedParT, self).__init__()
        self.mlp = mlp
        self.part = part
        self.part.eval() # ensure ParT is in eval mode
    
    def forward(self, *args):

        # process ParT model
        with torch.no_grad():
            _, x_cls = self.part(*args) # x_cls: (N, dim)
        
        return self.mlp(x_cls)


def get_model(data_config, **kwargs):

    # first load pre-trained ParT (arguments start with "part_")
    cfg_part = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=data_config.label_value_cls_num,
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
        num_classes_cls=data_config.label_value_cls_num, # for onnx export
    )
    for k in list(kwargs.keys()):
        if re.match(r'^part_', k):
            cfg_part[re.sub(r'^part_', '', k)] = kwargs.pop(k)

    cfg_part['num_classes'] = 188 ## hard-coded for now
    cfg_part['return_embed'] = True ## must return embeded features
    
    model_part = ParticleTransformerWrapper(**cfg_part)

    # then load MLP
    cfg = dict(
        ft_layer_params=[(32, 0.)],
        aux_layer_params=[],
    )
    kwargs.pop('loss_type', None)
    cfg.update(**kwargs)
    print(cfg)

    ft_dims = cfg_part['embed_dims'][-1]
    aux_dims = None
    num_classes = (data_config.label_value_cls_num if 'label_value_cls_num' in dir(data_config) else 0) \
                  + (data_config.label_value_reg_num if 'label_value_reg_num' in dir(data_config) else 0)
    model_mlp = MultiLayerPerceptron(ft_dims, aux_dims, num_classes, **cfg)

    model = ModelWithPretrainedParT(model_mlp, model_part)

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    print(model, model_info)
    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
