import torch
import torchvision
from torch.utils import data
from typing import *
import numpy as np
from pathlib import Path
import os
from vidio import VideoReader
import random


class SingleVideoDataset(data.Dataset):
    """
    PyTorch Dataset for loading a set of sequential frames and applying pre-defined augmentations to each frame.
    """

    def __init__(self,
                 vid_path: Path,
                 mean_by_channels: Union[list, np.ndarray] = [0, 0, 0],
                 transform: torchvision.transforms = None,
                 conv_mode: str = '2d',
                 frames_per_clip: int = 1
                 ):
        """
        Initializes a VideoDataset object. Reads in a video, applies the CPU augmentations to every frame in the clip,
        and stacks all channels together for input to a CNN.

        Parameters
        ----------
        vid_path: Path
            Path object to video being read in.
        mean_by_channels: Union[list, np.ndarray], default [0, 0, 0]
            Mean for each channel input, initially 0 for all channels.
        transform: torchvision Transform object, default None
            Image transformations to be applied to each frame.
        conv_mode: str, default '2d'
            Indicates convolution mode.
        """

        self.vid_path = vid_path
        self.mean_by_channels = self.parse_mean_by_channels(mean_by_channels)
        self.frames_per_clip = 1
        self.transform = transform
        self.conv_mode = conv_mode

        # validate path
        if not os.path.exists(vid_path):
            raise ValueError(f"No video found at this path: {vid_path}")

        with VideoReader(str(self.vid_path)) as reader:
            self.metadata = dict()

            self.metadata['vid_path'] = vid_path
            self.metadata['width'] = reader.next().shape[1]
            self.metadata['height'] = reader.next().shape[0]
            self.metadata['framecount'] = reader.nframes
            self.metadata['fps'] = reader.fps

        self._zeros_image = None

    def get_zeros_image(self, c, h, w):
        """Zero image frames to be added to front or back of image stack."""
        if self._zeros_image is None:
            # ALWAYS ASSUME OUTPUT IS TRANSPOSED
            self._zeros_image = np.zeros((c, h, w), dtype=np.uint8)
            for i in range(3):
                self._zeros_image[i, ...] = self.mean_by_channels[i]
        return self._zeros_image

    def parse_mean_by_channels(self, mean_by_channels):
        """Editing of mean_by_channels arg."""
        if isinstance(mean_by_channels[0], (float, np.floating)):
            return np.clip(np.array(mean_by_channels) * 255, 0, 255).astype(np.uint8)
        elif isinstance(mean_by_channels[0], (int, np.integer)):
            assert np.array_equal(np.clip(mean_by_channels, 0, 255), np.array(mean_by_channels))
            return np.array(mean_by_channels).astype(np.uint8)
        else:
            raise ValueError('unexpected type for input channel mean: {}'.format(mean_by_channels))

    def __len__(self):
        return self.metadata['framecount']

    def _prepend_with_zeros(self, stack: List[np.ndarray], blank_start_frames: int):
        """
        For frames at beginning of video, for flow generator must create dummy frames to 
        get optic flow features.
        
        Parameters
        ----------
        stack: List[np.ndarray]
            List of frames.
        blank_start_frames: int
            Number of frames that need to be prepended to stack. 
        """
        if blank_start_frames == 0:
            return stack
        for i in range(blank_start_frames):
            stack.insert(0, self.get_zeros_image(*stack[0].shape))
        return stack

    def _append_with_zeros(self, stack: List[np.ndarray], blank_end_frames: int):
        """
        For frames at end of video, for flow generator must create dummy frames to 
        get optic flow features.

        Parameters
        ----------
        stack: List[np.ndarray]
            List of frames.
        blank_end_frames: int
            Number of frames that need to be prepended to stack. 
        """
        if blank_end_frames == 0:
            return stack
        for i in range(blank_end_frames):
            stack.append(self.get_zeros_image(*stack[0].shape))
        return stack

    def __getitem__(self, index: int):
        """
        Used for reading frames from disk.

        Args:
            index: integer from 0 to number of total clips in dataset
        Returns:
            np.ndarray of shape (H,W,C), where C is 3* frames_per_clip
                Could also be torch.Tensor of shape (C,H,W), depending on the augmentation applied
        """

        images = list()
        # if frames per clip is 11, dataset[0] would have 5 blank frames preceding, with the 6th-11th being real frames
        blank_start_frames = max(self.frames_per_clip // 2 - index, 0)

        framecount = self.metadata['framecount']

        start_frame = index - self.frames_per_clip // 2 + blank_start_frames
        blank_end_frames = max(index - framecount + self.frames_per_clip // 2 + 1, 0)
        real_frames = self.frames_per_clip - blank_start_frames - blank_end_frames

        seed = np.random.randint(2147483647)

        with VideoReader(str(self.vid_path)) as reader:
            for i in range(real_frames):
                try:
                    image = reader[i + start_frame]
                except Exception as e:
                    image = self._zeros_image.copy().transpose(1, 2, 0)
                if self.transform:
                    random.seed(seed)
                    image = self.transform(image)
                    images.append(image)

        images = self._prepend_with_zeros(images, blank_start_frames)
        images = self._append_with_zeros(images, blank_end_frames)

        # images are now numpy arrays of shape 3, H, W
        # stacking in the first dimension changes to 3, T, H, W, compatible with Conv3D
        images = np.stack(images, axis=1)

        outputs = {'images': images}

        return outputs


class VideoDataset(data.Dataset):
    """ Simple wrapper around SingleVideoDataset for smoothly loading multiple videos """

    def __init__(self,
                 vid_paths: List[Path],
                 transform: torchvision.transforms = None,
                 conv_mode: str = '2d',
                 mean_by_channels: Union[list, np.ndarray] = [0, 0, 0],
                 frames_per_clip: int = 11
                 ):
        """
        Parameters
        ----------
        vid_paths: List[Path]
            List of video paths from current dataframe.
        transform: TorchVision transform object
            CPU transforms to be applied to the frames of videos as they are loaded in.
        conv_mode: str, default '2d'
            Convolution mode. Depends on the model being used, determines the number of convolution channels.
        mean_by_channels: Union[list, np.ndarray], default [0,0,0]
            Mean of normalization aug.
        frames_per_clip: int, default 11
            Number of rgb frames in each training sample. Based on flow window.
        """
        datasets = list()
        dataset_info = list()
        for i in range(len(vid_paths)):
            dataset = SingleVideoDataset(
                vid_paths[i],
                transform=transform,
                conv_mode=conv_mode,
                mean_by_channels=mean_by_channels,
                frames_per_clip=frames_per_clip
            )
            datasets.append(dataset)
            dataset_info.append(dataset.metadata)

        self.dataset = data.ConcatDataset(datasets)
        self.dataset_info = dataset_info

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]
