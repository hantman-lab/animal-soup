from inspect import isfunction
import warnings

import numpy as np
import torch
import torch.nn as nn

from .classifiers import resnet18, resnet50, resnet3d_34
from .utils import pop


# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def get_cnn(
        model_name: str,
        in_channels: int = 3,
        num_classes: int = 1000,
        freeze: bool = False,
        pos: np.ndarray = None,
        neg: np.ndarray = None,
        final_bn: bool = False,
        **kwargs):
    """
    Initializes a pretrained CNN from Torchvision.

    Parameters
    ----------
    model_name: str,
        One of ["resnet18", "resnet50", "resnet34-3D"]. Which ResNet model to load.
    in_channels: int, default 3
        Number of input channels. If not 3, the per-channel weights will be averaged and replicated
        in_channels times.
    num_classes: int, default 1000
        Number of output classes (neurons in final FC layer). Corresponds to number of behaviors
        being classified.
    freeze: boolean, default False
        If True, model weights will be frozen
    pos: np.ndarray
        Number of positive examples in training set. Used for custom bias initialization in
        final layer
    neg: np.ndarray
        Number of negative examples in training set. Used for custom bias initialization in
        final layer
    **kwargs (): passed to model initialization function

    Returns:
        model: a pytorch CNN

    """
    if model_name == "ResNet18":
        model = resnet18(in_channels=in_channels,
                         num_classes=num_classes,
                         **kwargs
                         )
    elif model_name == "ResNet50":
        model = resnet50(
                        in_channels=in_channels,
                        num_classes=num_classes,
                        **kwargs
        )
    else: # model name must be "ResNet3D-34"
        model = resnet3d_34(
                        in_channels=in_channels,
                        **kwargs
        )

    # freeze weights
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # we have to use the pop function because the final layer in these models has different names
    model, num_features, final_layer = pop(model, 1)
    linear_layer = nn.Linear(num_features, num_classes, bias=not final_bn)
    modules = [model, linear_layer]
    if final_bn:
        bn_layer = nn.BatchNorm1d(num_classes)
        modules.append(bn_layer)
    # initialize bias to roughly approximate the probability of positive examples in the training set
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
    if pos is not None and neg is not None:
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                bias = np.nan_to_num(np.log(pos / neg), neginf=0.0, posinf=1.0)
            bias = torch.nn.Parameter(torch.from_numpy(bias).float())
            if final_bn:
                bn_layer.bias = bias
            else:
                linear_layer.bias = bias

    model = nn.Sequential(*modules)
    return model
