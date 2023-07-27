"""
Utility functions and classes for getting transformations or applying them.

Taken from https://github.com/jbohnslav/deepethogram/blob/master/deepethogram/data/augs.py
with some slight modifications.
"""

from typing import *
import torch
import torchvision
import numpy as np
from kornia import augmentation as K
from kornia.augmentation.container import VideoSequential


class Normalizer:
    """Allows for easy z-scoring of tensors on the GPU.
    Example usage: You have a tensor of images of shape [N, C, H, W] or [N, T, C, H, W] in range [0,1]. You want to
        z-score this tensor.

    Methods:
        process_inputs: converts input mean, std into a torch tensor
        no_conversion: dummy method if you don't actually want to standardize the data
        handle_tensor: deals with self.mean and self.std depending on inputs. Example: your Tensor arrives on the GPU
            but your self.mean and self.std are still on the CPU. This method will move it appropriately.
        denormalize: converts standardized arrays back to their original range
        __call__: z-scores input data

    Instance variables:
        mean: mean of input data. For images, should have 2 or 3 channels
        std: standard deviation of input data
    """

    def __init__(
        self,
        mean: Union[list, np.ndarray, torch.Tensor] = None,
        std: Union[list, np.ndarray, torch.Tensor] = None,
        clamp: bool = True,
    ):
        """Constructor for Normalizer class.
        Args:
            mean: mean of input data. Should have 3 channels (for R,G,B) or 2 (for X,Y) in the optical flow case
            std: standard deviation of input data.
            clamp: if True, clips the output of a denormalized Tensor to between 0 and 1 (for images)
        """
        # make sure that if you have a mean, you also have a std
        # XOR
        has_mean, has_std = mean is None, std is None
        assert not has_mean ^ has_std

        self.mean = self.process_inputs(mean)
        self.std = self.process_inputs(std)
        # prevent divide by zero, but only change values if it's close to 0 already
        if self.std is not None:
            assert self.std.min() > 0
            self.std[self.std < 1e-8] += 1e-8
        self.clamp = clamp

    def process_inputs(self, inputs: Union[torch.Tensor, np.ndarray]):
        """Deals with input mean and std.
        Converts to tensor if necessary. Reshapes to [length, 1, 1] for pytorch broadcasting.
        """
        if inputs is None:
            return inputs
        if type(inputs) == list:
            inputs = np.array(inputs).astype(np.float32)
        if type(inputs) == np.ndarray:
            inputs = torch.from_numpy(inputs)
        assert type(inputs) == torch.Tensor
        inputs = inputs.float()
        C = inputs.shape[0]
        inputs = inputs.reshape(C, 1, 1)
        inputs.requires_grad = False
        return inputs

    def no_conversion(self, inputs):
        """Dummy function. Allows for normalizer to be called when you don't actually want to normalize.
        That way we can leave normalize in the training loop and only optionally call it.
        """
        return inputs

    def handle_tensor(self, tensor: torch.Tensor):
        """Reshapes std and mean to deal with the dimensions of the input tensor.
        Args:
            tensor: PyTorch tensor of shapes NCHW or NCTHW, depending on if your CNN is 2D or 3D
        Moves mean and std to the tensor's device if necessary
        If you've stacked the C dimension to have multiple images, e.g. 10 optic flows stacked has dim C=20,
            repeats self.mean and self.std to match
        """
        if tensor.ndim == 4:
            N, C, H, W = tensor.shape
        elif tensor.ndim == 5:
            N, C, T, H, W = tensor.shape
        else:
            raise ValueError(
                "Tensor input to normalizer of unknown shape: {}".format(tensor.shape)
            )

        t_d = tensor.device
        if t_d != self.mean.device:
            self.mean = self.mean.to(t_d)
        if t_d != self.std.device:
            self.std = self.std.to(t_d)

        c = self.mean.shape[0]
        if c < C:
            # handles the case where instead of N, C, T, H, W inputs, we have concatenated
            # multiple images along the channel dimension, so it's
            # N, C*T, H, W
            # this code simply repeats the mean T times, so it's
            # [R_mean, G_mean, B_mean, R_mean, G_mean, ... etc]
            n_repeats = C / c
            assert int(n_repeats) == n_repeats
            n_repeats = int(n_repeats)
            repeats = tuple([n_repeats] + [1 for i in range(self.mean.ndim - 1)])
            self.mean = self.mean.repeat((repeats))
            self.std = self.std.repeat((repeats))

        if tensor.ndim - self.mean.ndim > 1:
            # handles the case where our inputs are NCTHW
            self.mean = self.mean.unsqueeze(-1)
        if tensor.ndim - self.std.ndim > 1:
            self.std = self.std.unsqueeze(-1)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Converts normalized data back into its original distribution.
        If self.clamp: limits output tensor to the range (0,1). For images
        """
        if self.mean is None:
            return tensor

        # handles dealing with unexpected shape of inputs, wrong devices, etc.
        self.handle_tensor(tensor)
        tensor = (tensor * self.std) + self.mean
        if self.clamp:
            tensor = tensor.clamp(min=0.0, max=1.0)
        return tensor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalizes input data"""
        if self.mean is None:
            return tensor

        # handles dealing with unexpected shape of inputs, wrong devices, etc.
        self.handle_tensor(tensor)

        tensor = (tensor - self.mean) / (self.std)
        return tensor


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
        return self.__class__.__name__ + "()"


class DenormalizeVideo(torch.nn.Module):
    """Un-z-scores input video sequence"""

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

        return torch.clamp(tensor * self.std + self.mean, 0, 1)

    def forward(self, tensor):
        return self.normalize(tensor)


class NormalizeVideo(torch.nn.Module):
    """Z-scores input video sequences"""

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
    """Module for converting input uint8 tensors to floats, dividing by 255"""

    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float().div(255)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class UnstackClip(torch.nn.Module):
    """Module to convert image from N,C*T,H,W -> N,C,T,H,W"""

    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        N, C, H, W = tensor.shape
        T = C // 3

        return torch.stack(torch.chunk(tensor, T, dim=1), dim=2)


class StackClipInChannels(torch.nn.Module):
    """Module to convert image from N,C,T,H,W -> N,C*T,H,W"""

    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        N, C, T, H, W = tensor.shape
        tensor = tensor.transpose(1, 2)
        stacked = torch.cat([tensor[:, i, ...] for i in range(T)], dim=1)
        return stacked


def get_cpu_transforms(augs: Dict[str, Any]) -> torchvision.transforms:
    """
    Takes in a dictionary of augmentations to be applied to each frame and returns
    a TorchVision transform object to augment frames.
    """
    # order matters!!!!

    transforms = list()

    transforms.append(torchvision.transforms.ToTensor())

    if "crop_size" in augs.keys() and augs["crop_size"] is not None:
        transforms.append(
            torchvision.transforms.RandomCrop(augs["cro_size"], antialias=True)
        )
    if "resize" in augs.keys() and augs["resize"] is not None:
        transforms.append(torchvision.transforms.Resize(augs["resize"], antialias=True))
    if "pad" in augs.keys() and augs["pad"] is not None:
        transforms.append(torchvision.transforms.Pad(augs["pad"], antialias=True))

    transforms.append(Transpose())

    transforms = torchvision.transforms.Compose(transforms)

    return transforms


def get_gpu_transforms(
    augs: Dict[str, Any], conv_mode: str = "2d"
) -> torch.nn.Sequential:
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
        kornia_transforms.append(K.RandomHorizontalFlip(p=augs["LR"]))
    if "UD" in augs.keys() and augs["UD"] > 0:
        kornia_transforms.append(K.RandomVerticalFlip(p=augs["UD"]))
    if "degrees" in augs.keys() and augs["degrees"] > 0:
        kornia_transforms.append(K.RandomRotation(augs["degrees"]))

    if "grayscale" in augs.keys() and augs["grayscale"] > 0:
        kornia_transforms.append(K.RandomGrayscale(p=augs["grayscale"]))

    if (
        ("brightness" in augs.keys() and augs["brightness"] > 0)
        or ("contrast" in augs.keys() and augs["contrast"] > 0)
        or ("saturation" in augs.keys() and augs["saturation"] > 0)
        or ("hue" in augs.keys() and augs["hue"] > 0)
    ):
        kornia_transforms.append(
            K.ColorJitter(
                brightness=augs["brightness"],
                contrast=augs["contrast"],
                saturation=augs["saturation"],
                hue=augs["hue"],
                p=augs["color_p"],
                same_on_batch=False,
            )
        )

    norm = NormalizeVideo(
        mean=augs["normalization"]["mean"], std=augs["normalization"]["std"]
    )

    kornia_transforms = VideoSequential(
        *kornia_transforms, data_format="BCTHW", same_on_frame=True
    )

    train_transforms = [ToFloat(), kornia_transforms, norm]

    denormalize = []
    if conv_mode == "2d":
        train_transforms.append(StackClipInChannels())
        denormalize.append(UnstackClip())
    denormalize.append(
        DenormalizeVideo(
            mean=augs["normalization"]["mean"], std=augs["normalization"]["std"]
        )
    )

    train_transforms = torch.nn.Sequential(*train_transforms)
    denormalize = torch.nn.Sequential(*denormalize)

    gpu_transforms = dict(train=train_transforms, denormalize=denormalize)

    return gpu_transforms


def get_gpu_inference_transforms(augs: Dict[str, Any], conv_mode: str = '2d') -> torch.nn.Sequential:
    """Inference gpu transforms."""
    kornia_transforms = [
        ToFloat(),
        NormalizeVideo(mean=augs["normalization"]["mean"], std=augs["normalization"]["std"])
    ]

    if conv_mode == '2d':
        kornia_transforms.append(StackClipInChannels())

    inference_transforms = torch.nn.Sequential(*kornia_transforms)

    return inference_transforms

