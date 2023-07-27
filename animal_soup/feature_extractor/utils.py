"""
Utility functions for feature extractor inference.

Taken from: https://github.com/jbohnslav/deepethogram/blob/master/deepethogram/feature_extractor/inference.py
"""
from typing import *
from torch import nn


def unpack_penultimate_layer(model: Type[nn.Module]):
    """ Adds the activations in the penulatimate layer of the given PyTorch module to a dictionary called 'activation'.

    Assumes the model has two subcomponents: spatial and flow models. Every time the forward pass of this network
    is run, the penultimate neural activations will be added to the activations dictionary.
    This function uses the register_forward_hook syntax in PyTorch:
    https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks

    Example:
        my_model = deg_f()
        activations = unpack_penultimate_layer(my_model)
        print(activations) # nothing in it
        outputs = my_model(some_data)
        print(activations)
        # activations = {'spatial': some 512-dimensional vector, 'flow': another 512 dimensional vector}

    Args:
        model (nn.Module): a two-stream model with subcomponents spatial and flow
        fusion (str): one of average or concatenate

    Returns:
        activations (dict): dictionary with keys ['spatial', 'flow']. After forward pass, will contain
        512-dimensional vector of neural activations (before the last fully connected layer)
    """
    activation = {}

    def get_inputs(name):
        # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
        def hook(model, inputs, output):
            if type(inputs) == tuple:
                if len(inputs) == 1:
                    inputs = inputs[0]
                else:
                    raise ValueError('unknown inputs: {}'.format(inputs))
            activation[name] = inputs.detach()

        return hook

    final_spatial_linear = get_linear_layers(model.spatial_classifier)[-1]
    final_spatial_linear.register_forward_hook(get_inputs('spatial'))
    final_flow_linear = get_linear_layers(model.flow_classifier)[-1]
    final_flow_linear.register_forward_hook(get_inputs('flow'))
    return activation


def get_linear_layers(model: nn.Module) -> list:
    """unpacks the linear layers from a nn.Module, including in all the sequentials

    Parameters
    ----------
    model : nn.Module
        CNN

    Returns
    -------
    linear_layers: list
        ordered list of all the linear layers
    """
    linear_layers = []
    children = model.children()
    for child in children:
        if isinstance(child, nn.Sequential):
            linear_layers.append(get_linear_layers(child))
        elif isinstance(child, nn.Linear):
            linear_layers.append(child)
    return linear_layers


def get_penultimate_layer(model: Type[nn.Module]):
    """ Function to unpack a linear layer from a nn sequential module """
    assert isinstance(model, nn.Module)
    children = list(model.children())
    return children[-1]
