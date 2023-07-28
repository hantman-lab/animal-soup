"""
Feature extractor inference.

Adopted from: https://github.com/jbohnslav/deepethogram/blob/master/deepethogram/feature_extractor/inference.py
"""

from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

from .models import HiddenTwoStream
from ..data import VideoIterable
from .utils import *


def predict_single_video(
        vid_path: Path,
        hidden_two_stream: HiddenTwoStream,
        mean_by_channels: np.ndarray = [0, 0, 0],
        gpu_id: int = 0,
        flow_window: int = 11,
        cpu_transform=None,
        gpu_transform=None,
        num_workers: int = 8,
        batch_size: int = 16):
    """
    Runs inference on a single trial, caching the output probabilities and image and flow feature vectors.

    Parameters
    ----------
    vid_path: Path
        Path to input video
    hidden_two_stream: HiddenTwoStream
        Hidden two-stream model, feature extractor
    gpu_id: int, default 0
        GPU to use for inference. Default 0, assuming there is only one GPU to use.
    flow_window: int, default 11
        Flow window size. Used to infer optic flow features to pass to the feature extractor.
    mean_by_channels: np.ndarray, default [0, 0, 0]
        Image channel mean for z-scoring
    cpu_transform: callable
        CPU transforms for inference
    gpu_transform: callable
        GPU augmentations for inference
    num_workers: int, default 8
        Number of workers to read the video in parallel
    batch_size: int, default 16
        Batch size for inference.

    Returns
    -------
    dict
        keys: values
        probabilities: torch.Tensor, T x K probabilities of each behavior
        logits: torch.Tensor, T x K outputs for each behavior, before activation function
        spatial_features: T x 512 feature vectors from images
        flow_features: T x 512 feature vectors from optic flow

    """
    # make sure using CUDNN
    torch.backends.cudnn.benchmark = True

    device = torch.device(gpu_id)

    hidden_two_stream = hidden_two_stream.to(device)

    # free model, and set mode to inference
    for param in hidden_two_stream.parameters():
        param.requires_grad = False
    hidden_two_stream.eval()
   
    activation_function = nn.Sigmoid()

    dataset = VideoIterable(str(vid_path),
                            cpu_transform=cpu_transform,
                            num_workers=num_workers,
                            sequence_length=flow_window,
                            mean_by_channels=mean_by_channels)
    
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
    
    video_frame_num = len(dataset)

    activation = unpack_penultimate_layer(hidden_two_stream)

    buffer = {}

    for i, batch in enumerate(tqdm(dataloader, leave=False)):
        if isinstance(batch, dict):
            images = batch['images']
        elif isinstance(batch, torch.Tensor):
            images = batch
        else:
            raise ValueError('unknown input type: {}'.format(type(batch)))

        if images.device != device:
            images = images.to(device)
        with torch.no_grad():
            images = gpu_transform(images)

            logits = hidden_two_stream(images)
            spatial_features = activation['spatial']
            flow_features = activation['flow']
        
        # because we are using iterable datasets, each batch will be a consecutive chunk of frames from one worker
        # but they might be from totally different chunks of the video. therefore, we return the frame numbers,
        # and use this to store into our buffer in the right location
        frame_numbers = batch['framenum'].detach().cpu()

        probabilities = activation_function(logits).detach().cpu()
        logits = logits.detach().cpu()
        spatial_features = spatial_features.detach().cpu()
        flow_features = flow_features.detach().cpu()

        if i == 0:
            buffer['probabilities'] = torch.zeros((video_frame_num, probabilities.shape[1]), dtype=probabilities.dtype)
            buffer['logits'] = torch.zeros((video_frame_num, logits.shape[1]), dtype=logits.dtype)
            buffer['spatial_features'] = torch.zeros((video_frame_num, spatial_features.shape[1]),
                                                     dtype=spatial_features.dtype)
            buffer['flow_features'] = torch.zeros((video_frame_num, flow_features.shape[1]), dtype=flow_features.dtype)
        buffer['probabilities'][frame_numbers, :] = probabilities
        buffer['logits'][frame_numbers] = logits

        buffer['spatial_features'][frame_numbers] = spatial_features
        buffer['flow_features'][frame_numbers] = flow_features

    return buffer
