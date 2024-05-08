import torch
import torch.nn as nn
from torch import Tensor
import math
import re
import os
from weaver.utils.import_tools import import_module

_mod = import_module(os.path.join(os.path.dirname(__file__), 'mlp.py'), 'mlp')
Emsemble = _mod.Emsemble
EmsembleLoss = _mod.EmsembleLoss

def apply_seq(module_list, x):
    for m in module_list:
        x = m(x)
    return x

class SlicedSoftmax(nn.Module):

    def __init__(self, dim=-1, start=None, end=None):
        super(SlicedSoftmax, self).__init__()
        self.dim = dim
        self.start = start
        self.end = end

    def forward(self, x):
        if self.start is not None:
            slice = x[..., self.start:self.end]
            slice = torch.softmax(slice, dim=self.dim)
            x = torch.cat((x[..., :self.start], slice, x[..., self.end:]), dim=self.dim)
        else:
            x = torch.softmax(x, dim=self.dim)
        return x

class Indexing(nn.Module):
    
        def __init__(self, dim=-1, indices=[]):
            super(Indexing, self).__init__()
            self.dim = dim
            self.indices = indices
    
        def forward(self, x):
            return x[..., self.indices]

class MLPWeightSharing(nn.Module):
    r"""Parameters
    ----------
    input_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    layer_params : list
        List of the feature size for each layer.
    """

    def __init__(self, ft_dims, num_classes,
                 ft_layer_params=[(32, 0)],
                 merge_after_nth_layer=None,
                 **kwargs):

        super(MLPWeightSharing, self).__init__(**kwargs)

        # MLP in the ft path
        in_dim = ft_dims
        self.ft_mlp = nn.ModuleList()
        for out_dim, drop_rate, *args in ft_layer_params:
            self.ft_mlp.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            if len(args) > 0:
                # softmax layer
                assert args[0].startswith('sm:')
                start, end = args[0][3:].split(':')
                self.ft_mlp.append(SlicedSoftmax(dim=-1, start=int(start), end=int(end)))
            in_dim = out_dim

            if len(args) > 1:
                # indexing layer
                assert args[1].startswith('idx:')
                indices = []
                for ind in args[1][4:].split(','):
                    if ':' not in ind:
                        indices.append(int(ind))
                    else:
                        start, end =  ind.split(':')
                        indices.extend([e for e in range(int(start), int(end))])
                self.ft_mlp.append(Indexing(dim=-1, indices=indices))
                in_dim = len(indices) # update in_dim
    
        self.ft_mlp.append(nn.Linear(in_dim, num_classes))

        self.merge_after_nth_layer = merge_after_nth_layer
        assert self.merge_after_nth_layer is not None

    def forward(self, *x):
        # x[0]: ft vars in 2D branch, dim: (N, 2, C)
        x_ft = x[0]
        x_ft = apply_seq(self.ft_mlp[:(self.merge_after_nth_layer+1)], x_ft)
        x_ft = x_ft.mean(dim=1)
        return apply_seq(self.ft_mlp[(self.merge_after_nth_layer+1):], x_ft)


def get_model(data_config, **kwargs):
    cfg = dict(
        ft_layer_params=[(32, 0.)],
    )
    num_ensemble = kwargs.pop('num_ensemble', None)
    cfg.update(**kwargs)
    print(cfg)
    ft_dims = data_config.input_shapes['ft_vars'][-1] # ft vars are given as a single 2D branch

    num_classes = 2

    if num_ensemble is None:
        model = MLPWeightSharing(ft_dims, num_classes, **cfg)
    else:
        model = Emsemble(nn.ModuleList([MLPWeightSharing(ft_dims, num_classes, **cfg) for _ in range(num_ensemble)]))

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    print(model, model_info)
    return model, model_info


def get_loss(data_config, **kwargs):
    return EmsembleLoss(torch.nn.CrossEntropyLoss())
