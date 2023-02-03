import torch
import torch.nn as nn
from torch import Tensor
import math


def layer(in_dim, out_dim, no_relu=False):
    if no_relu:
        return nn.Linear(in_dim, out_dim)
    else:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

class MultiLayerPerceptron2Path(nn.Module):
    r"""Parameters
    ----------
    input_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    layer_params : list
        List of the feature size for each layer.
    """

    def __init__(self, preinput_dims, input_dims, num_classes,
                 prelayer_params=(32, 32), layer_params=(),
                 **kwargs):

        self.neurons_in_preprocess = kwargs.pop('neurons_in_preprocess', False)
        self.no_last_relu = kwargs.pop('no_last_relu', False)
        super(MultiLayerPerceptron2Path, self).__init__(**kwargs)

        if self.neurons_in_preprocess:
            preinput_dims += input_dims
        prechannels = [preinput_dims] + list(prelayer_params) + [num_classes]
        self.premlp = nn.Sequential(
            *[layer(in_dim, out_dim, no_relu=(self.no_last_relu and i==len(prechannels)-2)) for i, (in_dim, out_dim) in enumerate(zip(prechannels[:-1], prechannels[1:]))]
        )
        channels = [input_dims] + list(layer_params) + [num_classes]
        self.mlp = nn.Sequential(
            *[layer(in_dim, out_dim, no_relu=(self.no_last_relu and i==len(channels)-2)) for i, (in_dim, out_dim) in enumerate(zip(channels[:-1], channels[1:]))]
        )

    def forward(self, xp, x):
        # x: the feature vector initally read from the data structure, in dimension (N, C) (no last dimension P as we set length = None)
        if self.neurons_in_preprocess:
            xp = torch.cat((xp, x), dim=1)
        return self.mlp(x) + self.premlp(xp)


def get_model(data_config, **kwargs):
    cfg = dict(
        prelayer_params=(32, 32),
        layer_params=(),
    )
    cfg.update(**kwargs)
    preinput_dims = len(data_config.input_dicts['basic'])
    input_dims = len(data_config.input_dicts['highlevel'])
    num_classes = 1
    model = MultiLayerPerceptron2Path(preinput_dims, input_dims, num_classes, **cfg)

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    print(model, model_info)
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


def get_loss(data_config, **kwargs):
    return LogCoshLoss()
