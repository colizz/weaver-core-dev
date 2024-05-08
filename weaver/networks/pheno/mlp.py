import torch
import torch.nn as nn
from torch import Tensor
import math
import re


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


class MultiLayerPerceptronAdvanced(nn.Module):
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
        super(MultiLayerPerceptronAdvanced, self).__init__(**kwargs)

        if isinstance(self.concat_ft_layer, int) and aux_dims is not None:
            # merge the ft vars into the aux vars
            if self.concat_ft_layer == -1:
                aux_dims += ft_dims
            elif self.concat_ft_layer >= 0:
                aux_dims += ft_layer_params[self.concat_ft_layer][0]
        # MLP in the ft path
        in_dim = ft_dims
        self.ft_mlp_list = nn.ModuleList()
        self.resid_indices = []
        for out_dim, drop_rate, *ext_params in ft_layer_params:
            self.ft_mlp_list.append(nn.ModuleList())
            self.resid_indices.append(-1)
            self.ft_mlp_list[-1].append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim

            for param in ext_params:
                if param.startswith('sm:'):
                    # softmax layer
                    start, end = param[3:].split(':')
                    self.ft_mlp_list[-1].append(SlicedSoftmax(dim=-1, start=int(start), end=int(end)))

                if param.startswith('idx:'):
                    # indexing layer
                    indices = []
                    for ind in param[4:].split(','):
                        # if match with an integer
                        if re.match(r'^\d+$', ind):
                            indices.append(int(ind))
                        elif re.match(r'^\d+:\d+$', ind):
                            start, end = ind.split(':')
                            indices.extend([e for e in range(int(start), int(end))])
                    self.ft_mlp_list[-1].append(Indexing(dim=-1, indices=indices))
                    in_dim = len(indices) # update in_dim
                
                if param.startswith('resid:'):
                    # add residue path from layer N to this layer
                    self.resid_indices[-1] = int(param.replace('resid:', ''))

        self.ft_mlp_list.append(nn.ModuleList())
        self.ft_mlp_list[-1].append(nn.Linear(in_dim, num_classes))

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

    def forward(self, *args):
        # x[0]: ft vars in 2D branch, dim: (N, 1, C) or (N, C)
        # x[1]: (if valid) aux vars read from length=None, dim: (N, C)

        if len(args[0].shape) == 3:
            assert args[0].shape[1] == 1
            x_ft = args[0].squeeze(1)
        else:
            x_ft = args[0]

        if self.aux_mlp is not None:
            x_aux = args[1]
        
        x_out = {}
        x_out[-1] = x_ft # regarded as the index=-1 layer
        for i, ft_mlp in enumerate(self.ft_mlp_list):
            # pass the i-th layer
            x_out[i] = apply_seq(ft_mlp, x_out[i-1])

            # add residue path
            if i < len(self.resid_indices) and self.resid_indices[i] >= 0:
                x_out[i] = x_out[self.resid_indices[i]] + x_out[i]

        # add auxilary path
        if self.aux_mlp is not None:
            # if concat_ft_layer = -1, concatenate with original x_ft with x_aux; if larger, then take the corresponding x_out[i]
            x_out[i] = apply_seq(self.aux_mlp, torch.cat((x_aux, x_out[self.concat_ft_layer]), dim=-1)) + x_out[i]

        return x_out[i]


class Emsemble(nn.Module):
    # an emsemble of provided modules for n times, stacking their output nodes
    def __init__(self, model_ensemble):
        super(Emsemble, self).__init__()
        self.model_ensemble = model_ensemble
        self.num_ensemble = len(model_ensemble)
    
    def forward(self, *args):
        x_out = []
        for i in range(self.num_ensemble):
            x_out.append(self.model_ensemble[i](*args))
        # convert labels
        return (x_out, *args[1:]) # (output, label), should include the labels in the customised training routine


def get_model(data_config, **kwargs):
    cfg = dict(
        ft_layer_params=[(32, 0.)],
        aux_layer_params=[],
    )
    kwargs.pop('loss_type', None)
    num_ensemble = kwargs.pop('num_ensemble', None)
    cfg.update(**kwargs)
    print(cfg)
    ft_dims = data_config.input_shapes['ft_vars'][-1] # ft vars are given as a single 2D branch
    if 'aux_vars' in data_config.input_dicts:
        aux_dims = len(data_config.input_dicts['aux_vars'])
    else:
        aux_dims = None

    num_classes = (data_config.label_value_cls_num if 'label_value_cls_num' in dir(data_config) else 0) \
                  + (data_config.label_value_reg_num if 'label_value_reg_num' in dir(data_config) else 0)

    if num_ensemble is None:
        model = MultiLayerPerceptron(ft_dims, aux_dims, num_classes, **cfg)
    else:
        model = Emsemble(nn.ModuleList([MultiLayerPerceptron(ft_dims, aux_dims, num_classes, **cfg) for _ in range(num_ensemble)]))

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

class EmsembleLoss(torch.nn.Module):
    def __init__(self, loss_fn):
        super(EmsembleLoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, logit_stack, target, *args):
        loss_tot = 0.
        losses = [self.loss_fn(logit, target.long()) for logit in logit_stack]
        loss_tot = sum(losses)

        return {'loss': loss_tot, **{f'loss_em{ind}': loss.detach() for ind, loss in enumerate(losses)}}


def get_loss(data_config, **kwargs):
    loss_type = kwargs.pop('loss_type', 'cls')
    num_ensemble = kwargs.pop('num_ensemble', None)

    if loss_type == 'cls':
        fn = torch.nn.CrossEntropyLoss()
    elif loss_type == 'cls:mse':
        fn = ClsMSELoss()
    elif loss_type == 'reg':
        fn = LogCoshLoss()
    else:
        raise NotImplementedError

    if num_ensemble is None:
        return fn
    else:
        return EmsembleLoss(fn)
