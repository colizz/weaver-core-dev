import torch
import torch.nn as nn

def layer(in_dim, out_dim):
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
        self.skip_mlp = kwargs.pop('skip_mlp', False)
        super(MultiLayerPerceptron2Path, self).__init__(**kwargs)

        if self.neurons_in_preprocess:
            preinput_dims += input_dims
        prechannels = [preinput_dims] + list(prelayer_params) + [num_classes]
        self.premlp = nn.Sequential(
            *[layer(in_dim, out_dim) for in_dim, out_dim in zip(prechannels[:-1], prechannels[1:])]
        )
        channels = [input_dims] + list(layer_params) + [num_classes]
        if not self.skip_mlp:
            self.mlp = nn.Sequential(
                *[layer(in_dim, out_dim) for in_dim, out_dim in zip(channels[:-1], channels[1:])]
            )

    def forward(self, xp, x):
        # x: the feature vector initally read from the data structure, in dimension (N, C) (no last dimension P as we set length = None)
        x = torch.log(x.clamp(min=1e-10)) ## resume the pre-softmax score
        if self.neurons_in_preprocess:
            xp = torch.cat((xp, x), dim=1)
        if self.skip_mlp:
            x_upd = torch.cat((x[:, 28:], x[:, 23:28]), dim=1)
        else:
            x_upd = self.mlp(x)
        return x_upd + self.premlp(xp)


def get_model(data_config, **kwargs):
    prelayer_params = (32, 32)
    layer_params = ()
    preinput_dims = len(data_config.input_dicts['basic'])
    input_dims = len(data_config.input_dicts['highlevel'])
    num_classes = len(data_config.label_value)
    model = MultiLayerPerceptron2Path(preinput_dims, input_dims, num_classes, prelayer_params=prelayer_params, layer_params=layer_params, **kwargs)

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