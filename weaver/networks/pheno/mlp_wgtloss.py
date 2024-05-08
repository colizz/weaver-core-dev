import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import os
import copy
from weaver.utils.import_tools import import_module

_mod = import_module(os.path.join(os.path.dirname(__file__), 'mlp_shared.py'), '_')
MLPWeightSharing = _mod.MLPWeightSharing

_mod = import_module(os.path.join(os.path.dirname(__file__), 'mlp.py'), '_')
MultiLayerPerceptron = _mod.MultiLayerPerceptron
Emsemble = _mod.Emsemble

class ModelWithAuxWeight(nn.Module):

    def __init__(self, m_main, m_weight):
        super(ModelWithAuxWeight, self).__init__()
        self.m_main = m_main
        self.m_weight = m_weight
    
    def forward(self, *args):

        # process weight
        with torch.no_grad():
            logit_weight_stack, *_ = self.m_weight(*args)
            weight = sum([F.softmax(logit, dim=1) for logit in logit_weight_stack]) / len(logit_weight_stack)
        
        logit_stack, label, *args = self.m_main(*args)

        return logit_stack, label, weight, *args


def get_model(data_config, **kwargs):
    cfg = dict(
        ft_layer_params=[(32, 0.)],
    )
    num_ensemble = kwargs.pop('num_ensemble', None)
    use_mlp_shared = kwargs.pop('use_mlp_shared', False)
    assert num_ensemble

    # weight model params
    num_ensemble_weight_model = kwargs.pop('num_ensemble_weight_model', None)
    assert num_ensemble_weight_model

    ft_layer_params_weight_model = kwargs.pop('ft_layer_params_weight_model', None)
    cfg.update(**kwargs)
    print(f"{cfg=}")

    cfg_w = copy.deepcopy(cfg)
    cfg_w.update({
        'ft_layer_params': ft_layer_params_weight_model,
    })
    print(f"{cfg_w=}")

    # initialize params
    ft_dims = data_config.input_shapes['ft_vars'][-1] # ft vars are given as a single 2D branch

    num_classes = 2

    # get two models
    if use_mlp_shared:
        model_weight = Emsemble(nn.ModuleList([MLPWeightSharing(ft_dims, num_classes, **cfg_w) for _ in range(num_ensemble_weight_model)]))
        model_main = Emsemble(nn.ModuleList([MLPWeightSharing(ft_dims, num_classes, **cfg) for _ in range(num_ensemble)]))
    else:
        model_weight = Emsemble(nn.ModuleList([MultiLayerPerceptron(ft_dims, None, num_classes, **cfg_w) for _ in range(num_ensemble_weight_model)]))
        model_main = Emsemble(nn.ModuleList([MultiLayerPerceptron(ft_dims, None, num_classes, **cfg) for _ in range(num_ensemble)]))

    model = ModelWithAuxWeight(model_main, model_weight)

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    print(model, model_info)
    return model, model_info


class WeightedLoss(torch.nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none') # keep the batch dim

    def forward(self, logit_stack, target, weight, *args):
        '''
            logit_stack: ensemble of output logit tensor (N, out_dim)
            target: label tensor (N)
            weight: weight tensor (N, out_dim)
        '''

        target = target.long()

        ## attempts to multiply by (1-w) for both classes
        # weight_gather = weight.gather(1, target.unsqueeze(1)).squeeze(1) # (N)
        # losses = [((1 - weight_gather) * self.loss_fn(logit, target)).mean(dim=0) for logit in logit_stack]

        # multiply by w1/w2 for the 2nd class
        weight = torch.clamp(weight / weight[:, 1:], min=1e-5, max=10) # in format of N x (w1/w2, 1)
        weight_gather = weight.gather(1, (1 - target).unsqueeze(1)).squeeze(1) # if target = 0 (data), take 1, if target = 1 (bkg), take w1/w2
        losses = [(weight_gather * self.loss_fn(logit, target)).mean(dim=0) for logit in logit_stack]

        loss_tot = sum(losses)

        return {'loss': loss_tot, **{f'loss_em{ind}': loss.detach() for ind, loss in enumerate(losses)}}


def get_loss(data_config, **kwargs):
    return WeightedLoss()
