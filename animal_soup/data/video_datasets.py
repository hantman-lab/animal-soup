import torchvision
from torch.utils import data
from typing import *
import numpy as np
from pathlib import Path
from vidio import VideoReader
import random
from .utils import *
from collections import deque
from ..utils import resolve_path


class SingleVideoDataset(data.Dataset):
    """PyTorch Dataset for loading a set of sequential frames and applying pre-defined augmentations to each frame."""

    def __init__(
        self,
        vid_path: Dict[str, Path],
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
        vid_path: Dict[str, Path]
            Dictionary containing the relative front and side video paths for a single trial.
        label: np.ndarray, default None
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

        self.front_path = resolve_path(vid_path["front"])
        self.side_path = resolve_path(vid_path["side"])

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
        if not Path.is_file(self.front_path):
            raise ValueError(f"No front video found at this path: {vid_path}")
        if not Path.is_file(self.side_path):
            raise ValueError(f"No side video found at this path: {vid_path}")

        self.metadata = dict()

        left_reader = VideoReader(str(self.side_path))
        right_reader = VideoReader(str(self.front_path))

        data_metadata = dict()

        data_metadata["vid_path"] = vid_path
        # check to make sure front and side videos are same side
        side_width = left_reader.next().shape[1]
        front_width = right_reader.next().shape[1]
        data_metadata["width"] = min(side_width, front_width)

        side_height = left_reader.next().shape[0]
        front_height = right_reader.next().shape[0]
        if side_height != front_height:
            raise ValueError(f"side video height: {side_height} and "
                             f"front video height: {front_height} do not match")
        else:
            data_metadata["height"] = side_height

        data_metadata["framecount"] = min(left_reader.nframes, right_reader.nframes)

        # check same fps
        side_fps = left_reader.fps
        front_fps = right_reader.fps
        if side_fps != front_fps:
            raise ValueError(f"side video fps: {side_fps} does not match "
                             f"front video fps: {front_fps}")
        else:
            data_metadata["fps"] = side_fps

        self.metadata["data_metadata"] = data_metadata

        left_reader.close()
        right_reader.close()

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

        left_reader = VideoReader(str(self.side_path))
        right_reader = VideoReader(str(self.front_path))

        for i in range(real_frames):
            try:
                left_image = left_reader[i + start_frame]
                right_image = right_reader[i + start_frame]
                image = np.hstack((left_image, right_image))
            except Exception as e:
                left_image = self._zeros_image.copy().transpose(1, 2, 0)
                right_image = self._zeros_image.copy().transpose(1, 2, 0)
                image = np.hstack((left_image, right_image))
            if self.transform:
                random.seed(seed)
                image = self.transform(image)
                images.append(image)

        images = self._prepend_with_zeros(images, blank_start_frames)
        images = self._append_with_zeros(images, blank_end_frames)

        # images are now numpy arrays of shape 3, H, W
        # stacking in the first dimension changes to 3, T, H, W, compatible with Conv3D
        images = np.stack(images, axis=1)
        images = images.transpose(2, 1, 3, 0)

        outputs = {"images": images}

        if self.label is not None:
            outputs["labels"] = self.label[index]

        left_reader.close()
        right_reader.close()

        return outputs


class VideoDataset(data.Dataset):
    """Simple wrapper around SingleVideoDataset for smoothly loading multiple videos."""

    def __init__(
        self,
        vid_paths: List[Dict[str, Path]],
        transform: torchvision.transforms = None,
        conv_mode: str = "2d",
        mean_by_channels: Union[list, np.ndarray] = [0, 0, 0],
        frames_per_clip: int = 11,
        labels: List[np.ndarray] = None
    ):
        """
        Parameters
        ----------
        vid_paths: List[Dict[str, Path]]
            List of dictionaries containing the relative front and side video paths for trials in the
            current dataframe.
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

                self.pos_weight = make_loss_weight(self.num_pos, self.num_neg)

        self.dataset = data.ConcatDataset(datasets)
        self.dataset_info = dataset_info

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]


class VideoIterable(data.IterableDataset):
    """Highly optimized Dataset for running inference on videos."""
    def __init__(self,
                 vid_path: Dict[str, Path],
                 cpu_transform,
                 sequence_length: int = 11,
                 mean_by_channels: Union[list, np.ndarray] = [0, 0, 0]):
        """
        Parameters
        ----------
        vid_path: Dict[str, Path]
            Dictionary of relative path to front and side video of a trial.
        cpu_transform:
            CPU transforms (cropping, resizing)
        sequence_length: int, optional
            Number of images in one clip, by default 11
        mean_by_channels : Union[list, np.ndarray], optional
            [description], by default [0, 0, 0]
        """
        super().__init__()

        # currently not supporting parallelized reading
        # until I figure it out
        num_workers = 0

        self.side_readers = {i: 0 for i in range(num_workers)}
        self.front_readers = {i: 0 for i in range(num_workers)}
        self.side_path = resolve_path(vid_path["side"])
        self.front_path = resolve_path(vid_path["front"])
        self.transform = cpu_transform

        self.start = 0
        self.sequence_length = sequence_length

        left_reader = VideoReader(str(self.side_path))
        right_reader = VideoReader(str(self.front_path))

        self.n_frames = min(len(left_reader), len(right_reader))

        left_reader.close()
        right_reader.close()

        self.blank_start_frames = self.sequence_length // 2
        self.cnt = 0

        self.mean_by_channels = self.parse_mean_by_channels(mean_by_channels)
        self.num_workers = num_workers
        self.buffer = deque([], maxlen=self.sequence_length)

        self._zeros_image = None
        self._image_shape = None
        self.get_image_shape()

    def __len__(self):
        return self.n_frames

    def get_image_shape(self):
        """Get image shape after CPU augmentations applied"""
        left_reader = VideoReader(str(self.side_path))
        right_reader = VideoReader(str(self.front_path))

        side_im = left_reader[0]
        front_im = right_reader[0]

        im = np.hstack((side_im, front_im))

        im = self.transform(im)
        self._image_shape = im.shape

    def get_zeros_image(self, ):
        if self._zeros_image is None:
            if self._image_shape is None:
                raise ValueError('must set shape before getting zeros image')
            # ALWAYS ASSUME OUTPUT IS TRANSPOSED
            self._zeros_image = np.zeros(self._image_shape, dtype=np.uint8)
            for i in range(3):
                self._zeros_image[i, ...] = self.mean_by_channels[i]
        return self._zeros_image.copy()

    def parse_mean_by_channels(self, mean_by_channels):
        if isinstance(mean_by_channels[0], (float, np.floating)):
            return np.clip(np.array(mean_by_channels) * 255, 0, 255).astype(np.uint8)
        elif isinstance(mean_by_channels[0], (int, np.integer)):
            assert np.array_equal(np.clip(mean_by_channels, 0, 255), np.array(mean_by_channels))
            return np.array(mean_by_channels).astype(np.uint8)
        else:
            raise ValueError('unexpected type for input channel mean: {}'.format(mean_by_channels))

    def my_iter_func(self, start, end):
        for i in range(start, end):
            self.buffer.append(self.get_current_item())

            yield {'images': np.stack(self.buffer, axis=1).transpose(2, 1, 3, 0), 'framenum': self.cnt - 1 - self.sequence_length // 2}

    def get_current_item(self):
        """
        Returns frames based on the frame number. Will return blank frames based on sequence size if frame count
        is at beginning or end of video.
        """
        worker_info = data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if self.cnt < 0:
            im = self.get_zeros_image()
        elif self.cnt >= self.n_frames:
            im = self.get_zeros_image()
        else:
            try:
                side_im = self.side_readers[worker_id][self.cnt]
                front_im = self.front_readers[worker_id][self.cnt]
                im = np.hstack((side_im, front_im))
            except Exception as e:
                print(f'problem reading frame {self.cnt}')
                raise
            im = self.transform(im)

        self.cnt += 1
        return im

    def fill_buffer_init(self, iter_start):
        self.cnt = iter_start
        # hack for the first one: don't quite fill it up
        for i in range(iter_start, iter_start + self.sequence_length - 1):
            self.buffer.append(self.get_current_item())

    def __iter__(self):
        """
        Gets the current worker info and if reading frames has not started, initializes. Otherwise,
        determines how much work is left.
        """
        worker_info = data.get_worker_info()
        iter_end = self.n_frames - self.sequence_length // 2
        if worker_info is None:
            iter_start = -self.blank_start_frames
            self.side_readers[0] = VideoReader(str(self.side_path))
            self.front_readers[0] = VideoReader(str(self.front_path))
        else:
            per_worker = self.n_frames // self.num_workers
            remaining = self.n_frames % per_worker
            nums = [per_worker for i in range(self.num_workers)]
            nums = [nums[i] + 1 if i < remaining else nums[i] for i in range(self.num_workers)]
            # print(nums)
            nums.insert(0, 0)
            starts = np.cumsum(nums[:-1])  # - self.blank_start_frames
            starts = starts.tolist()
            ends = starts[1:] + [iter_end]
            starts[0] = -self.blank_start_frames

            #print(starts, ends)

            iter_start = starts[worker_info.id]
            iter_end = min(ends[worker_info.id], self.n_frames)
            # print(f'worker: {worker_info.id}, start: {iter_start} end: {iter_end}')
            self.side_readers[worker_info.id] = VideoReader(str(self.side_path))
            self.front_readers[worker_info.id] = VideoReader(str(self.front_path))
        # FILL THE BUFFER TO START
        # print('iter start: {}'.format(iter_start))
        self.fill_buffer_init(iter_start)
        return self.my_iter_func(iter_start, iter_end)

    def close(self):
        """Close front and side readers."""
        # if value in reader dict is not a video reader instance continue
        for k, v in self.side_readers.items():
            if isinstance(v, int):
                continue
            # else value of key will be video reader, try to close
            try:
                v.close()
            except Exception as e:
                print(f'error destroying reader {k}')
        # if value in reader dict is not a video reader instance continue
        for k, v in self.front_readers.items():
            if isinstance(v, int):
                continue
            # else value of key will be video reader, try to close
            try:
                v.close()
            except Exception as e:
                print(f'error destroying reader {k}')

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
