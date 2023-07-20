"""File for calculating the normalization augmentations based on the data in the current dataframe."""

from typing import *
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from vidio import VideoReader


# https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
class StatsRecorder:
    def __init__(self,
                 mean: np.ndarray = None,
                 std: np.ndarray = None,
                 n_observations: int = None):
        """
        Class for calculating the normalization statistics used in video augmentation. 
        
        Parameters
        ----------
        mean: np.ndarray, default None
            Mean of video being added.
        std: np.ndarray, default None
            Standard deviation of video being added.
        n_observations: int, default None
            Number of videos used to calculate the overall mean and standard deviation thus far. 
        """
        # initially no videos have been seen
        self.nobservations = 0
        
        if mean is not None:
            assert std is not None
            assert n_observations is not None
            assert mean.shape == std.shape
            self.mean = mean
            self.std = std
            self.n_observations = n_observations

    def first_batch(self, data: Union[np.ndarray, torch.Tensor]):
        """
        Used for first video that stats are calculated for. Need to initialize the mean, standard deviation, 
        number of observations, and number of dimensions.
        
        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            Array of shape (nobservations, ndimensions)
        """
        if isinstance(data, np.ndarray):
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
        else: # type is torch.Tensor
            data = data.detach()  # don't accumulate gradients
            if data.ndim == 1:
                # assume it's one observation, not multiple observations with 1 dimension
                data = data.unsqueeze(0)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
            
        self.nobservations = data.shape[0]
        self.ndimensions = data.shape[1]

    def update(self, data: Union[np.ndarray, torch.Tensor]):
        """
        Update the overall mean and standard deviation for the data being used for training/inference.
        
        Parameters
        ----------
        data: np.ndarray or torch.Tensor
            Array of shape (nobservations, ndimensions)
        """
        # if no observations have been seen yet, initialize
        if self.nobservations == 0:
            self.first_batch(data)
            return  
        
        if data.shape[1] != self.ndimensions:
            raise ValueError("Data dims don't match prev observations.")
        
        if isinstance(data, np.ndarray):
            data = np.atleast_2d(data)
            new_mean = data.mean(axis=0)
            new_std = data.std(axis=0)
        else: # data is torch.Tensor
            data = data.detach()  # don't accumulate gradients
            if data.ndim == 1:
                # assume it's one observation, not multiple observations with 1 dimension
                data = data.unsqueeze(0)
            new_mean = data.mean(dim=0)
            new_std = data.std(dim=0)

        m = self.nobservations * 1.0
        n = data.shape[0]

        tmp = self.mean

        self.mean = m / (m + n) * tmp + n / (m + n) * new_mean
        self.std = m / (m + n) * self.std ** 2 + n / (m + n) * new_std ** 2 + \
                   m * n / (m + n) ** 2 * (tmp - new_mean) ** 2

        self.std = self.std ** 0.5

        self.nobservations += n


def get_video_statistics(video_path: Path, stride: int = 10) -> Dict[str, Union[float, int]]:
    """
    Calculates the channel-wise mean and standard deviation for a given input video.

    Parameters
    ----------
    video_path: Path
        Path object to video to calculate statistics for.
    stride: int, default 10
        Will only calculate statistics for every stride so that entire videos do not have to be loaded into memory.
        If you want to calculate image statistics for every frame, use stride=1.

    Returns
    -------
    imdata: Dict[str, Union[float, int]]
        Video stats for a given video returns as a dictionary where keys correspond to mean, standard deviation,
        and the number of frames taken from.
    """
    # use stats recorder to easily keep track of and update mean, std, and N across a single video
    image_stats = StatsRecorder()
    with VideoReader(str(video_path)) as reader:
        for i in tqdm(range(0, len(reader), stride)):
            try:
                image = reader[i]
            except Exception as e:
                continue
            image = image.astype(float) / 255
            image = image.transpose(2, 1, 0)
            image = image.reshape(3, -1).transpose(1, 0)
            image_stats.update(image)

    imdata = {'mean': image_stats.mean, 'std': image_stats.std, 'N': image_stats.nobservations}
    for k, v in imdata.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        imdata[k] = v

    reader.close()

    return imdata


def get_normalization(vid_paths: List[Path]) -> Dict[str, Union[float, int]]:
    """
    Returns the mean and standard deviation of all videos in current dataframe to be used for training/inference.

    Parameters
    ----------
    vid_paths: List[Path]
        List of video paths to calculate normalization augmentation.

    Returns
    -------
    normalization: Dict[str, Union[float, int]]
        Dictionary containing the total mean, standard deviation, and number of frames sampled from in all
        videos that are going to be used in training.
    """

    # initialize as zero, will update as video means/stds/frame selections is calculated
    mean_total = 0
    std_total = 0
    N_frames = 0

    for vp in vid_paths:
        vid_stats = get_video_statistics(vp, 10)

        N_frames += vid_stats['N']
        mean_total = (mean_total + vid_stats['N'] * np.array(vid_stats['mean'])) / N_frames
        std_total = (std_total + vid_stats['N'] * np.array(vid_stats['std'])) / N_frames

    normalization = dict()
    normalization["N"] = N_frames
    normalization["mean"] = mean_total
    normalization["std"] = std_total

    return normalization



