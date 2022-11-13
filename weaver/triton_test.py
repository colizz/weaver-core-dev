from importlib import import_module
from utils.dataset import SimpleIterDataset
import torch
import argparse
from typing import Optional, List, Dict

import numpy as np

batch_size = 10
pfs = 100
#svs = 7                                                                                                                                                                                                    
svs = 10

input_dict = {
    "pf_points__0": np.random.rand(batch_size, 2, pfs).astype("float32"),
    #"pf_features__1": np.random.rand(batch_size, 19, pfs).astype("float32"),
    "pf_features__1": np.random.rand(batch_size, 25, pfs).astype("float32"),
    "pf_mask__2": (np.random.rand(batch_size, 1, pfs) > 0.2).astype("float32"),
    "sv_points__3": np.random.rand(batch_size, 2, svs).astype("float32"),
    "sv_features__4": np.random.rand(batch_size, 11, svs).astype("float32"),
    "sv_mask__5": (np.random.rand(batch_size, 1, svs) > 0.2).astype("float32"),
}

"""
import tritonclient.grpc as triton_grpc
import tritonclient.http as triton_http

# from https://github.com/lgray/hgg-coffea/blob/triton-bdts/src/hgg_coffea/tools/chained_quantile.py
class wrapped_triton:
    def __init__(
        self,
        model_url: str,
    ) -> None:
        fullprotocol, location = model_url.split("://")
        _, protocol = fullprotocol.split("+")
        address, model, version = location.split("/")

        self._protocol = protocol
        self._address = address
        self._model = model
        self._version = version

    def __call__(self, input_dict: Dict[str, np.ndarray]) -> np.ndarray:
        if self._protocol == "grpc":
            client = triton_grpc.InferenceServerClient(url=self._address, verbose=False)
            triton_protocol = triton_grpc
        elif self._protocol == "http":
            client = triton_http.InferenceServerClient(
                url=self._address,
                verbose=False,
                concurrency=12,
            )
            triton_protocol = triton_http
        else:
            raise ValueError(f"{self._protocol} does not encode a valid protocol (grpc or http)")

        # Infer
        inputs = []

        for key in input_dict:
            input = triton_protocol.InferInput(key, input_dict[key].shape, "FP32")
            input.set_data_from_numpy(input_dict[key])
            inputs.append(input)

        output = triton_protocol.InferRequestedOutput("softmax__0")

        request = client.infer(
            self._model,
            model_version=self._version,
            inputs=inputs,
            outputs=[output],
        )

        out = request.as_numpy("softmax__0")

        return out

# model_url = "triton+grpc://ailab01.fnal.gov:8001/particlenet_hww/1"
# model_url = "triton+grpc://67.58.49.52:8001/particlenet_hww_ttbarwjets/1"
model_url = "triton+grpc://67.58.49.52:8001/particlenet_hww_inclv2_pre2/1"
#model_url = "triton+grpc://prp-gpu-1.t2.ucsd.edu:8001/particlenet_hww_ttbarwjets/1"
#model_url = "triton+grpc://prp-gpu-1.t2.ucsd.edu:8001/particlenet_hww_ttbarwjets/1"

triton_model = wrapped_triton(model_url)

print("running inference using the jitted model on triton")
output = triton_model(input_dict)
print(output)

"""
print('running inference using the non-jitted model locally')
# network_config = 'particle_net_pf_sv_4_layers_pyg_ef.py'
network_config = 'networks/particle_net_pf_sv_hybrid.py'

# data_config = SimpleIterDataset([], 'melissa_dataconfig_semilep_ttbarwjets.yaml', for_training=False).config
# data_config.export_json(f"model.json")

data_config = SimpleIterDataset([], 'models/particlenet_hww_inclv2_pre2/data/ak8_MD_vminclv2_pre2.yaml', for_training=False).config
network_module = import_module(network_config.replace(".py", "").replace("/", "."))
model, model_info = network_module.get_model(data_config, jittable=True, for_inference=True)
model.load_state_dict(
    torch.load(f"models/particlenet_hww_inclv2_pre2/data/net_best_epoch_state.pt", map_location=torch.device("cpu")),
    strict=False,
)
_ = model.eval()
jit_model = torch.jit.script(model)
_ = jit_model.eval()

print(model(torch.tensor(input_dict['pf_points__0']),
            torch.tensor(input_dict['pf_features__1']),
            torch.tensor(input_dict['pf_mask__2']),
            torch.tensor(input_dict['sv_points__3']),
            torch.tensor(input_dict['sv_features__4']),
            torch.tensor(input_dict['sv_mask__5'])))

print(jit_model(torch.tensor(input_dict['pf_points__0']),
            torch.tensor(input_dict['pf_features__1']),
            torch.tensor(input_dict['pf_mask__2']),
            torch.tensor(input_dict['sv_points__3']),
            torch.tensor(input_dict['sv_features__4']),
            torch.tensor(input_dict['sv_mask__5'])))

jit_model  = torch.load("models/particlenet_hww_inclv2_pre2/1/model.pt")
_ = jit_model.eval()

print(jit_model(torch.tensor(input_dict['pf_points__0']),
            torch.tensor(input_dict['pf_features__1']),
            torch.tensor(input_dict['pf_mask__2']),
            torch.tensor(input_dict['sv_points__3']),
            torch.tensor(input_dict['sv_features__4']),
            torch.tensor(input_dict['sv_mask__5'])))
