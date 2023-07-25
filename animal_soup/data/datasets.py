import torchvision
from torch.utils import data
from typing import *
import numpy as np
from pathlib import Path
from vidio import VideoReader
import random
from .utils import *


class SingleVideoDataset(data.Dataset):
    """PyTorch Dataset for loading a set of sequential frames and applying pre-defined augmentations to each frame."""

    def __init__(
        self,
        vid_path: Path,
        label: np.ndarray = None,
        mean_by_channels: Union[list, np.ndarray] = [0, 0, 0],
        transform: torchvision.transforms = None,
        conv_mode: str = "2d",
        frames_per_clip: int = 1,
    ):
        """
        Initializes a VideoDataset object. Reads in a video, applies the CPU augmentations to every frame in the clip,
        and stacks all channels together for input to a CNN.

        Parameters
        ----------
        vid_path: Path
            Path object to video being read in.
        labels: np.ndarray, default None
            If supervised training, ethogram labels of associated video.
        mean_by_channels: Union[list, np.ndarray], default [0, 0, 0]
            Mean for each channel input. Will either be the channel means given by the normalization augmentation
            or will be the default.
        transform: torchvision Transform object, default None
            Image transformations to be applied to each frame.
        conv_mode: str, default '2d'
            Indicates convolution mode
        frames_per_clip: int, default 1
            How many sequential frames in a clip
        """

        self.vid_path = vid_path

        if isinstance(mean_by_channels[0], (float, np.floating)):
            self.mean_by_channels = np.clip(
                np.array(mean_by_channels) * 255, 0, 255
            ).astype(np.uint8)
        elif isinstance(mean_by_channels[0], (int, np.integer)):
            assert np.array_equal(
                np.clip(mean_by_channels, 0, 255), np.array(mean_by_channels)
            )
            self.mean_by_channels = np.array(mean_by_channels).astype(np.uint8)

        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.conv_mode = conv_mode

        # validate path
        if not Path.is_file(vid_path):
            raise ValueError(f"No video found at this path: {vid_path}")

        self.metadata = dict()
        with VideoReader(str(self.vid_path)) as reader:
            data_metadata = dict()

            data_metadata["vid_path"] = vid_path
            data_metadata["width"] = reader.next().shape[1]
            data_metadata["height"] = reader.next().shape[0]
            data_metadata["framecount"] = reader.nframes
            data_metadata["fps"] = reader.fps

        self.metadata["data_metadata"] = data_metadata

        if label is not None:
            label_metadata = dict()

            self.label = prepare_label(label)
            self.class_counts = self.label.sum(axis=0).astype(int)
            self.num_pos = (self.label == 1).sum(axis=0).astype(int)
            self.num_neg = (self.label == 0).sum(axis=0).astype(int)
            self.label_shape = self.label.shape

            label_metadata["class_counts"] = self.class_counts
            label_metadata["label_shape"] = self.label_shape

            self.metadata["label_metadata"] = label_metadata
        else:
            self.label = None

        self._zeros_image = None

    def get_zeros_image(self, c, h, w):
        """
        Zero image frames to be added to front or back of image stack.

        Parameters
        ----------
        c: int,
            colors dims, will be 2 or 3
        h: int
            image height
        w: int
            image width
        """
        if self._zeros_image is None:
            # ALWAYS ASSUME OUTPUT IS TRANSPOSED
            self._zeros_image = np.zeros((c, h, w), dtype=np.uint8)
            for i in range(3):
                self._zeros_image[i, ...] = self.mean_by_channels[i]
        return self._zeros_image

    def __len__(self):
        return self.metadata["data_metadata"]["framecount"]

    def _prepend_with_zeros(self, stack: List[np.ndarray], blank_start_frames: int):
        """
        For frames at beginning of video, flow generator must create dummy frames to
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
        For frames at end of video, flow generator must create dummy frames to
        get optic flow features.

        Parameters
        ----------
        stack: List[np.ndarray]
            List of frames.
        blank_end_frames: int
            Number of frames that need to be appended to stack.
        """
        if blank_end_frames == 0:
            return stack
        for i in range(blank_end_frames):
            stack.append(self.get_zeros_image(*stack[0].shape))
        return stack

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Used for reading frames from disk.

        Parameters
        ----------
        index: int
            Index to get given frame in video.

        Returns
        -------
        frame: np.ndarray
            np.ndarray of shape (H,W,C), where C is 3* frames_per_clip
            Could also be a torch.Tensor of shape (C,H,W), depending on the augmentation applied
        """

        images = list()
        # if frames per clip is 11, dataset[0] would have 5 blank frames preceding, with the 6th-11th being real frames
        blank_start_frames = max(self.frames_per_clip // 2 - index, 0)

        framecount = self.metadata["data_metadata"]["framecount"]

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
        images = images.transpose(2, 1, 0, 3)

        outputs = {"images": images}

        if self.label is not None:
            outputs["labels"] = self.label[index]

        reader.close()

        return outputs


class VideoDataset(data.Dataset):
    """Simple wrapper around SingleVideoDataset for smoothly loading multiple videos."""

    def __init__(
        self,
        vid_paths: List[Path],
        transform: torchvision.transforms = None,
        conv_mode: str = "2d",
        mean_by_channels: Union[list, np.ndarray] = [0, 0, 0],
        frames_per_clip: int = 11,
        labels: List[np.ndarray] = None
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
        labels: List[np.ndarray], default None
            List of ethogram labels, used in supervised training (flow generator, sequence model).
        """
        datasets = list()
        dataset_info = list()

        if labels is None: # non supervised training
            for i in range(len(vid_paths)):
                dataset = SingleVideoDataset(
                    vid_paths[i],
                    label=None,
                    transform=transform,
                    conv_mode=conv_mode,
                    mean_by_channels=mean_by_channels,
                    frames_per_clip=frames_per_clip,
                )
                datasets.append(dataset)
                dataset_info.append(dataset.metadata)
        else: # supervised training
            final_labels = list()
            self.class_counts = 0
            self.num_pos = 0
            self.num_neg = 0
            self.num_labels = len(labels)
            for i in range(len(vid_paths)):
                dataset = SingleVideoDataset(
                    vid_paths[i],
                    label=labels[i],
                    transform=transform,
                    conv_mode=conv_mode,
                    mean_by_channels=mean_by_channels,
                    frames_per_clip=frames_per_clip,
                )
                datasets.append(dataset)
                dataset_info.append(dataset.metadata)
                final_labels.append(dataset.label)

                self.class_counts += dataset.class_counts
                self.num_neg += dataset.num_neg
                self.num_pos += dataset.num_pos

                self.labels = np.concatenate(final_labels)

        self.dataset = data.ConcatDataset(datasets)
        self.dataset_info = dataset_info

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]
