"""Explicit path resolution; never depend on a hardcoded host or working directory."""
import copy
import os
from pathlib import Path
from string import Template


def expand(value):
    try:
        return os.path.expanduser(Template(str(value)).substitute(os.environ))
    except KeyError as exc:
        raise ValueError(f"Required environment variable is not set: {exc.args[0]}") from exc


def resolve_path(value, base=None):
    path = Path(expand(value))
    if not path.is_absolute():
        path = Path.cwd() / path if base is None else Path(base) / path
    return path.resolve()


def model_reference(value, base=None):
    raw = expand(value)
    candidate = resolve_path(raw, base)
    if candidate.exists() or Path(raw).is_absolute() or raw.startswith(("/", ".", "~")) or "$" in str(value):
        if not candidate.is_dir():
            raise FileNotFoundError(f"Local model directory does not exist: {candidate}")
        return str(candidate)
    return raw  # Hugging Face repository ID; require a revision for training.


def metadata_paths(meta, metadata_file, data_root=None):
    result = copy.deepcopy(meta)
    parent = Path(metadata_file).resolve().parent
    root = result["root_path"]
    result["root_path"] = str(resolve_path(root, data_root if data_root is not None else parent))
    if "normalization_path" in result:
        result["normalization_path"] = str(resolve_path(result["normalization_path"], parent))
    if "episode_index_path" in result:
        result["episode_index_path"] = str(resolve_path(result["episode_index_path"], result["root_path"]))
    return result
