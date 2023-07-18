from typing import *
from pathlib import Path
import torch


def get_gpu_options() -> Dict[int, str]:
    """Returns a dictionary of {gpu_id: gpu name}"""
    gpu_options = dict()

    device_count = torch.cuda.device_count()

    for gpu_id in range(device_count):
        gpu_options[gpu_id] = torch.cuda.get_device_properties(gpu_id).name

    return gpu_options

def generate_flow_dataloader(videos: List[Path],
                             augs: Dict[str, Any],
                             batch_size: int,
                             conv_mode: str = Union["2d", "3d"]) -> Tuple[Dict[str], torch.utils.Dataloader]:
    """
    Creates a dataset for training based on the available trials in the current dataframe.

    Parameters
    ----------
    videos: List[Path]
        List of video paths available for training.
    conv_mode: str
        One of '2d', '3d'.
        If 2D, batch will be of shape [N, C*T, H, W].
        If 3D, batch will be of shape [N, C, T, H, W]
    augs: Dict[str, Any]
        Dictionary containing the image augmentations applied to the dataset.
    batch_size: int
        Batch size, number of training samples to go through before updating model params.

    Returns
    -------
    Info about the dataloader as well as a dataloader for training.

    """


    pass



