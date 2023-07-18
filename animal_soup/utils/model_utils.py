from typing import *
from pathlib import Path
import torch
import torchvision
import numpy as np

class Transpose:
    """Module to transpose image stacks.
    """
    def __call__(self, images: np.ndarray) -> np.ndarray:
        shape = images.shape
        if len(shape) == 4:
            # F x H x W x C -> C x F x H x W
            return images.transpose(3, 0, 1, 2)
        elif len(shape) == 3:
            # H x W x C -> C x H x W
            return images.transpose(2, 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + '()'

def get_gpu_options() -> Dict[int, str]:
    """Returns a dictionary of {gpu_id: gpu name}"""
    gpu_options = dict()

    device_count = torch.cuda.device_count()

    for gpu_id in range(device_count):
        gpu_options[gpu_id] = torch.cuda.get_device_properties(gpu_id).name

    return gpu_options

def get_cpu_transforms(augs: Dict[str, Any]) -> torchvision.transforms:
    """
    Takes in a dictionary of augmentations to be applied to each frame and returns
    a TorchVision transform object to augment frames.
    """
    # order matters!!!!

    transforms = list()

    if "crop_size" in augs.keys() and augs["crop_size"] is not None:
        transforms.append(torchvision.transforms.RandomCrop(augs["cro_size"]))
    if "resize" in augs.keys() and augs["resize"] is not None:
        transforms.append(torchvision.transforms.Resize(augs["resize"]))
    if "pad" in augs.keys() and augs["pad"] is not None:
        transforms.append(torchvision.transforms.Pad(augs["pad"]))

    transforms.append(Transpose())

    transforms = torchvision.transforms.Compose(transforms)

    return transforms





