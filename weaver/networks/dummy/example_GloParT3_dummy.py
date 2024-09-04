import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, **kwargs):
        super(DummyModel, self).__init__()
        self.kwargs = kwargs
        self.param = nn.Parameter(torch.zeros(1, 280))

    def forward(self, *inputs):
        bsz, *_ = inputs[0].shape
        # output dimension (bsz, x), x = hidden layer (256) + converted output nodes (24)
        return torch.arange(280).unsqueeze(0).repeat(bsz, 1) + self.param

def get_model(data_config, **kwargs):

    model = DummyModel(**kwargs)
    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['output'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output': {0: 'N'}}},
    }

    return model, model_info
