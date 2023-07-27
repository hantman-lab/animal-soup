"""
Feature extractor model utility function/classes.

Taken from: https://github.com/jbohnslav/deepethogram/blob/master/deepethogram/feature_extractor/models/utils.py
"""

from torch import nn
import torch


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


def remove_cnn_classifier_layer(cnn):
    """Removes the final layer of a torchvision classification model, and figures out dimensionality of final layer"""
    # cnn should be a nn.Sequential(custom_model, nn.Linear)
    module_list = list(cnn.children())
    assert (len(module_list) == 2 or len(module_list) == 3) and isinstance(module_list[1], nn.Linear)
    in_features = module_list[1].in_features
    module_list[1] = nn.Identity()
    cnn = nn.Sequential(*module_list)
    return cnn, in_features


class Fusion(nn.Module):
    def __init__(self,
                 fusion_type,
                 num_spatial_features,
                 num_flow_features,
                 num_classes,
                 flow_fusion_weight=1.5,
                 activation=nn.Identity()):
        """
        Module for fusing spatial and flow features and passing through Linear layer.

        Parameters
        ----------
        fusion_type: str
            One of ['average', 'weighted_average', 'concatenate']. Indicates how spatial and
            flow classifier should be fused together.
        num_spatial_features: int
            Number of output features to spatial classification model. Last layer is Linear.
        num_flow_features: int
            Number of output features to flow classification model. Last layer is Linear.
        num_classes: int
            Number of behaviors being classified for plus background.
        flow_fusion_weight: float, default 1.5
            How much to up-weight the flow fusion.
        activation: default torch.nn.Identity()
            Forwarding activation layer.
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        self.activation = activation
        self.flow_fusion_weight = flow_fusion_weight

        if self.fusion_type == 'average':
            # self.spatial_fc = nn.Linear(num_spatial_features,num_classes)
            # self.flow_fc = nn.Linear(num_flow_features, num_classes)

            self.num_features_out = num_classes

        elif self.fusion_type == 'concatenate':
            self.num_features_out = num_classes
            self.fc = nn.Linear(num_spatial_features + num_flow_features, num_classes)

        elif self.fusion_type == 'weighted_average':
            self.flow_weight = nn.Parameter(torch.Tensor([0.5]).float(), requires_grad=True)
        else:
            raise NotImplementedError

    def forward(self, spatial_features, flow_features):
        if self.fusion_type == 'average':
            return (spatial_features + flow_features * self.flow_fusion_weight) / (1 + self.flow_fusion_weight)
        elif self.fusion_type == 'concatenate':
            # if we're concatenating, we want the model to learn nonlinear mappings from the spatial logits and flow
            # logits that means we should apply an activation function note: this won't work if you froze both
            # encoding models
            features = self.activation(torch.cat((spatial_features, flow_features), dim=1))
            return self.fc(features)
        elif self.fusion_type == 'weighted_average':
            return self.flow_weight * flow_features + (1 - self.flow_weight) * spatial_features
