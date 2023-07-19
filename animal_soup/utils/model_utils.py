from typing import *
import torch
import torchvision
import numpy as np
from kornia import augmentation as K
from kornia.augmentation.container import VideoSequential


class Transpose:
    """Module to transpose image stacks."""
    def __call__(self, images: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(images, torch.Tensor):
            images = images.numpy()
        shape = images.shape
        if len(shape) == 4:
            # F x H x W x C -> C x F x H x W
            return images.transpose(3, 0, 1, 2)
        elif len(shape) == 3:
            # H x W x C -> C x H x W
            return images.transpose(2, 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class DenormalizeVideo(torch.nn.Module):
    """Un-z-scores input video sequences
    """
    def __init__(self, mean, std):
        super().__init__()

        mean = np.asarray(mean)
        std = np.asarray(std)
        mean = torch.from_numpy(mean).float()
        std = torch.from_numpy(std).float()

        self.mean = mean.reshape(1, -1, 1, 1, 1)
        self.std = std.reshape(1, -1, 1, 1, 1)

        # self.normalize = K.Denormalize(mean=mean, std=std)

    def normalize(self, tensor):
        if self.mean.device != tensor.device:
            self.mean = self.mean.to(tensor.device)
        if self.std.device != tensor.device:
            self.std = self.std.to(tensor.device)

        return torch.clamp( tensor*self.std + self.mean , 0, 1)

    def forward(self, tensor):
        return self.normalize(tensor)

class NormalizeVideo(torch.nn.Module):
    """Z-scores input video sequences
    """

    def __init__(self, mean, std):
        super().__init__()

        mean = np.asarray(mean)
        std = np.asarray(std)
        mean = torch.from_numpy(mean).float()
        std = torch.from_numpy(std).float()

        self.mean = mean.reshape(1, -1, 1, 1, 1)
        self.std = std.reshape(1, -1, 1, 1, 1)
        self.ndim = self.mean.ndim

        assert self.ndim == self.std.ndim

    def normalize(self, tensor):
        assert tensor.ndim == self.ndim
        if self.mean.device != tensor.device:
            self.mean = self.mean.to(tensor.device)
        if self.std.device != tensor.device:
            self.std = self.std.to(tensor.device)
        return (tensor.float() - self.mean) / self.std

    def forward(self, tensor):
        return self.normalize(tensor)

class ToFloat(torch.nn.Module):
    """Module for converting input uint8 tensors to floats, dividing by 255
    """
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float().div(255)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class UnstackClip(torch.nn.Module):
    """Module to convert image from N,C*T,H,W -> N,C,T,H,W
    """
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        N, C, H, W = tensor.shape
        T = C // 3

        return torch.stack(torch.chunk(tensor, T, dim=1), dim=2)

class StackClipInChannels(torch.nn.Module):
    """Module to convert image from N,C,T,H,W -> N,C*T,H,W
    """
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        N, C, T, H, W = tensor.shape
        tensor = tensor.transpose(1, 2)
        stacked = torch.cat([tensor[:, i, ...] for i in range(T)], dim=1)
        return stacked

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


def get_gpu_transforms(augs: Dict[str, Any], conv_mode: str = '2d') -> torch.nn.Sequential:
    """
    Makes GPU augmentations.

    Parameters
    ----------
    augs : Dict[str, Any]
        Augmentation parameters
    conv_mode : str, optional
        If '2d', stacks clip in channels. If 3d, returns 5-D tensor, by default '2d'

    Returns
    -------
     A dict of nn.Sequential with Kornia augmentations.
    """

    kornia_transforms = []

    if "LR" in augs.keys() and augs["LR"] > 0:
        kornia_transforms.append(K.RandomHorizontalFlip(p=augs["LR"],
                                                        same_on_batch=False,
                                                        return_transform=False))
    if "UD" in augs.keys() and augs["UD"] > 0:
        kornia_transforms.append(K.RandomVerticalFlip(p=augs["UD"],
                                                      same_on_batch=False, return_transform=False))
    if "degrees" in augs.keys() and augs["degrees"] > 0:
        kornia_transforms.append(K.RandomRotation(augs["degrees"]))

    if ("brightness" in augs.keys() and augs["brightness"] > 0) or \
            ("contrast" in augs.keys() and augs["contrast"] > 0) or \
            ("saturation" in augs.keys() and augs["saturation"] > 0) or \
            ("hue" in augs.keys() and augs["hue"] > 0):
        kornia_transforms.append(K.ColorJitter(brightness=augs["brightness"],
                                               contrast=augs["contrast"],
                                               saturation=augs["saturation"],
                                               hue=augs["hue"],
                                               p=augs["color_p"],
                                               same_on_batch=False,
                                               return_transform=False))
    if "grayscale" in augs.keys() and augs["grayscale"] > 0:
        kornia_transforms.append(K.RandomGrayscale(p=augs["grayscale"]))

    norm = NormalizeVideo(mean=augs["normalization"]["mean"],
                          std=augs["normalization"]["std"])

    kornia_transforms = VideoSequential(*kornia_transforms,
                                        data_format='BCTHW',
                                        same_on_frame=True)

    train_transforms = [ToFloat(),
                        kornia_transforms,
                        norm]

    denormalize = []
    if conv_mode == '2d':
        train_transforms.append(StackClipInChannels())
        denormalize.append(UnstackClip())
    denormalize.append(DenormalizeVideo(mean=augs["normalization"]["mean"],
                                        std=augs["normalization"]["std"]))

    train_transforms = torch.nn.Sequential(*train_transforms)
    denormalize = torch.nn.Sequential(*denormalize)

    gpu_transforms = dict(train=train_transforms,
                          denormalize=denormalize)

    print(f'GPU transforms: {gpu_transforms}')

    return gpu_transforms





