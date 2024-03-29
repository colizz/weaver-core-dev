import torch
import torch.nn as nn


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

    def __init__(self, input_dims, num_classes,
                 layer_params=(1024, 256, 256),
                 **kwargs):

        self.prep_log = kwargs.pop('prep_log', False)
        self.prep_norm = kwargs.pop('prep_norm', False)
        super(MultiLayerPerceptron, self).__init__(**kwargs)
        channels = [input_dims] + list(layer_params) + [num_classes]
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1]))
            if i < len(channels) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: the feature vector initally read from the data structure, in dimension (N, C) (no last dimension P as we set length = None)
        if self.prep_norm:
            x = x / x.sum(axis=1, keepdims=True)
        if self.prep_log:
            x = torch.log(x.clamp(min=1e-10))
        return self.mlp(x)


def get_model(data_config, **kwargs):
    cfg = dict(
        layer_params=(),
    )
    cfg.update(**kwargs)
    input_dims = len(data_config.input_dicts['highlevel'])
    num_classes = len(data_config.label_value)
    model = MultiLayerPerceptron(input_dims, num_classes, **cfg)

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