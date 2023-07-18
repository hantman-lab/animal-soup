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





