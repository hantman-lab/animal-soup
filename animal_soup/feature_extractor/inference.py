from pathlib import Path
import torch
from torch import nn
import numpy as np

from .models import HiddenTwoStream


# def predict_single_video(
#         vid_path: Path,
#         hidden_two_stream: HiddenTwoStream,
#         activation_function: nn.Module,
#         fusion: str,
#         num_rgb: int,
#         mean_by_channels: np.ndarray,
#         device: str = 'cuda:0',
#         cpu_transform=None,
#         gpu_transform=None,
#         should_print: bool = False,
#         num_workers: int = 1,
#         batch_size: int = 16):
#     """
#     Runs inference on a single trial, caching the output probabilities and image and flow feature vectors.
#
#     Parameters
#     ----------
#     vid_path : Path
#         Path to input video
#     hidden_two_stream : nn.Module
#         Hidden two-stream model
#     activation_function : nn.Module
#         Sigmoid activation function
#     fusion : str
#         How features are fused. Needed for extracting them from the model architecture
#     num_rgb : int
#         How many images are input to the model
#     mean_by_channels : np.ndarray
#         Image channel mean for z-scoring
#     device : str, optional
#         Device on which to run inference, by default 'cuda:0'. Options: ['cuda:N', 'cpu']
#     cpu_transform : callable, optional
#         CPU transforms to perform, e.g. center cropping / resizing, by default None
#     gpu_transform : callable, optional
#         GPU augmentations. For inference, should just be conversion to float and z-scoring, by default None
#     should_print : bool, optional
#         If true, print more debug statements, by default False
#     num_workers : int, optional
#         Number of workers to read the video in parallel, by default 1
#     batch_size : int, optional
#         Batch size for inference. Values above 1 will be much faster. by default 16
#
#     Returns
#     -------
#     dict
#         keys: values
#         probabilities: torch.Tensor, T x K probabilities of each behavior
#         logits: torch.Tensor, T x K outputs for each behavior, before activation function
#         spatial_features: T x 512 feature vectors from images
#         flow_features: T x 512 feature vectors from optic flow
#         debug: T x 1 tensor storing the number of times each frame was read. Should be full of ones and only ones
#
#     Raises
#     ------
#     ValueError
#         If input from dataloader is not a dict or a Tensor, raises
#     """
#
#     hidden_two_stream.eval()
#     # model.set_mode('inference')
#
#     if type(device) != torch.device:
#         device = torch.device(device)
#
#     dataset = VideoIterable(vid_path,
#                             transform=cpu_transform,
#                             num_workers=num_workers,
#                             sequence_length=num_rgb,
#                             mean_by_channels=mean_by_channels)
#     dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
#     video_frame_num = len(dataset)
#
#     activation = unpack_penultimate_layer(hidden_two_stream, fusion)
#
#     buffer = {}
#
#     has_printed = False
#     # log.debug('model training mode: {}'.format(model.training))
#     for i, batch in enumerate(tqdm(dataloader, leave=False)):
#         if isinstance(batch, dict):
#             images = batch['images']
#         elif isinstance(batch, torch.Tensor):
#             images = batch
#         else:
#             raise ValueError('unknown input type: {}'.format(type(batch)))
#
#         if images.device != device:
#             images = images.to(device)
#         # images = batch['images']
#         with torch.no_grad():
#             images = gpu_transform(images)
#
#             logits = hidden_two_stream(images)
#             spatial_features = activation['spatial']
#             flow_features = activation['flow']
#         # because we are using iterable datasets, each batch will be a consecutive chunk of frames from one worker
#         # but they might be from totally different chunks of the video. therefore, we return the frame numbers,
#         # and use this to store into our buffer in the right location
#         frame_numbers = batch['framenum'].detach().cpu()
#
#         probabilities = activation_function(logits).detach().cpu()
#         logits = logits.detach().cpu()
#         spatial_features = spatial_features.detach().cpu()
#         flow_features = flow_features.detach().cpu()
#
#         if not has_printed and should_print:
#             print_debug_statement(images, logits, spatial_features, flow_features, probabilities)
#             has_printed = True
#         if i == 0:
#             # print(f'~~~ N: {N} ~~~')
#             buffer['probabilities'] = torch.zeros((video_frame_num, probabilities.shape[1]), dtype=probabilities.dtype)
#             buffer['logits'] = torch.zeros((video_frame_num, logits.shape[1]), dtype=logits.dtype)
#             buffer['spatial_features'] = torch.zeros((video_frame_num, spatial_features.shape[1]),
#                                                      dtype=spatial_features.dtype)
#             buffer['flow_features'] = torch.zeros((video_frame_num, flow_features.shape[1]), dtype=flow_features.dtype)
#             buffer['debug'] = torch.zeros((video_frame_num,)).float()
#         buffer['probabilities'][frame_numbers, :] = probabilities
#         buffer['logits'][frame_numbers] = logits
#
#         buffer['spatial_features'][frame_numbers] = spatial_features
#         buffer['flow_features'][frame_numbers] = flow_features
#         buffer['debug'][frame_numbers] += 1
#     return buffer
