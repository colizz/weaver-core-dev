import torch
import torch.nn as nn
from torch import Tensor
import math


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

class MultiLayerPerceptron(nn.Module):
    r"""Parameters
    ----------
    input_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    layer_params : list
        List of the feature size for each layer.
    """

    def __init__(self, ft_dims, aux_dims, num_classes,
                 ft_layer_params=[(32, 0)], aux_layer_params=[],
                 **kwargs):

        self.concat_ft_layer = kwargs.pop('concat_ft_layer', -1) # -1: merge the beginning input ft vars to aux (i.e. layer index = -1) by default
        super(MultiLayerPerceptron, self).__init__(**kwargs)

        if isinstance(self.concat_ft_layer, int) and aux_dims is not None:
            # merge the ft vars into the aux vars
            if self.concat_ft_layer == -1:
                aux_dims += ft_dims
            elif self.concat_ft_layer >= 0:
                aux_dims += ft_layer_params[self.concat_ft_layer][0]
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

        # MLP in the aux path
        if aux_dims is not None:
            in_dim = aux_dims
            self.aux_mlp = nn.ModuleList()
            for out_dim, drop_rate in aux_layer_params:
                self.aux_mlp.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            self.aux_mlp.append(nn.Linear(in_dim, num_classes))
        else:
            self.aux_mlp = None

    def forward(self, *x):
        # x[0]: ft vars in 2D branch, dim: (N, 1, C)
        # x[1]: (if valid) aux vars read from length=None, dim: (N, C)

        x_ft = x[0].squeeze(1)

        if self.aux_mlp is not None:
            # has an additional aux path.
            # note that the ft path will be merge to this path, after passing "concat_ft_layer" ft layers
            x_aux = x[1]
            if self.concat_ft_layer == -1:
                return apply_seq(self.ft_mlp, x_ft) + apply_seq(self.aux_mlp, torch.cat((x_aux, x_ft), dim=-1))
            elif self.concat_ft_layer >= 0:
                x_ft = apply_seq(self.ft_mlp[:(self.concat_ft_layer+1)], x_ft)
                return apply_seq(self.ft_mlp[(self.concat_ft_layer+1):], x_ft) + apply_seq(self.aux_mlp, torch.cat((x_aux, x_ft), dim=-1))
        else:
            return apply_seq(self.ft_mlp, x_ft)


def get_model(data_config, **kwargs):
    cfg = dict(
        ft_layer_params=[(32, 0.)],
        aux_layer_params=[],
    )
    kwargs.pop('loss_type', None)
    cfg.update(**kwargs)
    print(cfg)
    ft_dims = data_config.input_shapes['ft_vars'][-1] # ft vars are given as a single 2D branch
    if 'aux_vars' in data_config.input_dicts:
        aux_dims = len(data_config.input_dicts['aux_vars'])
    else:
        aux_dims = None

    num_classes = (data_config.label_value_cls_num if 'label_value_cls_num' in dir(data_config) else 0) \
                  + (data_config.label_value_reg_num if 'label_value_reg_num' in dir(data_config) else 0)

    model = MultiLayerPerceptron(ft_dims, aux_dims, num_classes, **cfg)

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

class ClsMSELoss(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(ClsMSELoss, self).__init__(None, None, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # expend target to one-hot
        target = torch.nn.functional.one_hot(target, num_classes=input.shape[-1])
        x = input - target
        loss = x * x
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

def get_loss(data_config, **kwargs):
    loss_type = kwargs.pop('loss_type', 'cls')
    if loss_type == 'cls':
        return torch.nn.CrossEntropyLoss()
    if loss_type == 'cls:mse':
        return ClsMSELoss()
    elif loss_type == 'reg':
        return LogCoshLoss()
    else:
        raise NotImplementedError
