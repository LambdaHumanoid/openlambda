"""Reproduction metadata independent of Git commits or machine-specific paths."""
import importlib.metadata
import platform


def environment():
    packages = {}
    for name in ("torch", "torchvision", "transformers", "accelerate", "numpy", "av", "pyarrow",
                 "safetensors", "flash-linear-attention", "causal-conv1d"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages}
