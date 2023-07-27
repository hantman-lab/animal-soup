"""General utility functions."""

import torch
from typing import *
from pathlib import Path
from .dataframe import validate_path


def get_gpu_options() -> Dict[int, str]:
    """Returns a dictionary of {gpu_id: gpu name}"""
    gpu_options = dict()

    device_count = torch.cuda.device_count()

    for gpu_id in range(device_count):
        gpu_options[gpu_id] = torch.cuda.get_device_properties(gpu_id).name

    return gpu_options


def validate_checkpoint_path(model_checkpoint_path: Union[str, Path]) -> Path:
    """Utility function for validating a user-passed model checkpoint path."""
    # validate path
    model_checkpoint_path = validate_path(model_checkpoint_path)
    # check if path exists
    if not Path.is_file(model_checkpoint_path):
        raise ValueError(f"No checkpoint file exists at: {model_checkpoint_path}")
    # check valid suffix
    if model_checkpoint_path.suffix not in [".pt", ".ckpt"]:
        raise ValueError(
            "PyTorch model checkpoints should end in '.pt' or '.ckpt'. "
            "Please make sure the file you are trying to use is a model checkpoint."
        )
    return model_checkpoint_path


