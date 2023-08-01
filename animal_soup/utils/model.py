"""General utility functions."""
import pandas as pd
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


def validate_exp_type(df: pd.DataFrame) -> str:
    """Takes in a dataframe and returns the experiment type."""
    # validate that an exp_type has been set
    if None in set(df["exp_type"].values):
        raise ValueError("The experiment type for trials in your dataframe has not been set. Please"
                         "set the `exp_type` column in your dataframe before attempting training.")
    # validate only one experiment type being used in training
    if len(set(df["exp_type"].values)) > 1:
        raise ValueError("Training can only be completed with experiments of same type. "
                         f"The current experiments in your dataframe are: {set(list(df['exp_type']))} "
                         "Take a subset of your dataframe to train with one kind of experiment.")
    # set the experiment type
    exp_type = list(df["exp_type"])[0]

    return exp_type


