"""Read-only environment inspection; does not download weights or start training."""
import argparse
import json
import importlib.util
from .environment import environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Also query local CUDA availability.")
    args = parser.parse_args(argv)
    info = environment()
    info["optional_fast_kernels"] = {name: importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d")}
    if args.cuda:
        import torch
        info["cuda"] = {"available": torch.cuda.is_available(), "build": torch.version.cuda,
                        "devices": torch.cuda.device_count()}
    print(json.dumps(info, indent=2))
