from torch import nn


def pop(model, n_layers):
    """Remove the last layer of a resnet model."""

    if n_layers == 1:
        # use empty sequential module as an identity function
        num_features = model.fc.in_features
        final_layer = model.fc
        model.fc = nn.Identity()
    else:
        raise NotImplementedError('Can only pop off the final layer of a resnet')

    return model, num_features, final_layer
