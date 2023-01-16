import argparse

import torch

from utils.dataset import SimpleIterDataset

from importlib import import_module

import numpy as np

batch_size = 10
pfs = 100
svs = 10

input_dict = {
    "pf_points__0": np.random.rand(batch_size, 2, pfs).astype("float32"),
    "pf_features__1": np.random.rand(batch_size, 25, pfs).astype("float32"),
    "pf_mask__2": (np.random.rand(batch_size, 1, pfs) > 0.2).astype("float32"),
    "sv_points__3": np.random.rand(batch_size, 2, svs).astype("float32"),
    "sv_features__4": np.random.rand(batch_size, 11, svs).astype("float32"),
    "sv_mask__5": (np.random.rand(batch_size, 1, svs) > 0.2).astype("float32"),
}

def main(args):
    data_config = SimpleIterDataset([], args.data_config, for_training=False).config
    data_config.export_json(f"{args.model_prefix}.json")

    network_module = import_module(args.network_config.replace(".py", "").replace("/", "."))
    model, model_info = network_module.get_model(data_config, jittable=True, for_inference=True)
    model.load_state_dict(
        torch.load(f"{args.model_prefix}_best_epoch_state.pt", map_location=torch.device("cpu")),
        strict=False,
    )
    _ = model.eval()

    model_output = model(
        torch.tensor(input_dict["pf_points__0"]),
        torch.tensor(input_dict["pf_features__1"]),
        torch.tensor(input_dict["pf_mask__2"]),
        torch.tensor(input_dict["sv_points__3"]),
        torch.tensor(input_dict["sv_features__4"]),
        torch.tensor(input_dict["sv_mask__5"]),
    )

    jit_model = torch.jit.script(model)

    jitted_model_output = jit_model(
        torch.tensor(input_dict["pf_points__0"]),
        torch.tensor(input_dict["pf_features__1"]),
        torch.tensor(input_dict["pf_mask__2"]),
        torch.tensor(input_dict["sv_points__3"]),
        torch.tensor(input_dict["sv_features__4"]),
        torch.tensor(input_dict["sv_mask__5"]),
    )

    if (model_output != jitted_model_output).sum() == 0:
        print(f"Saved in {args.model_prefix}_jitted.pt")
        #jit_model.save(f"{args.model_prefix}_jitted.pt")
        torch.jit.save(jit_model, f"{args.model_prefix}_jitted.pt")
    else:
        print("OOPS: jitted and unjitted model output DON'T match")
        sys.exit(0
)
if __name__ == "__main__":
    # e.g.
    # inside a condor job: python run.py --year 2017 --processor trigger --condor --starti 0 --endi 1
    # inside a dask job:  python run.py --year 2017 --processor trigger --dask

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--data-config",
        type=str,
        default="data/ak15_points_pf_sv_v0.yaml",
        help="data config YAML file",
    )
    parser.add_argument(
        "-n",
        "--network-config",
        type=str,
        default="networks/particle_net_pfcand_sv.py",
        help="network architecture configuration file; the path must be relative to the current dir",
    )
    parser.add_argument(
        "-m",
        "--model-prefix",
        type=str,
        default="models/{auto}/network",
        help="path to save or load the model; for training, this will be used as a prefix, so model snapshots "
        "will saved to `{model_prefix}_epoch-%d_state.pt` after each epoch, and the one with the best "
        "validation metric to `{model_prefix}_best_epoch_state.pt`; for testing, this should be the full path "
        "including the suffix, otherwise the one with the best validation metric will be used; "
        "for training, `{auto}` can be used as part of the path to auto-generate a name, "
        "based on the timestamp and network configuration",
    )
    args = parser.parse_args()

    main(args)
