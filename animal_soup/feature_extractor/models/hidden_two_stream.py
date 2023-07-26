"""
Hidden Two Stream model class.

Taken from: https://github.com/jbohnslav/deepethogram/blob/master/deepethogram/feature_extractor/models/hidden_two_stream.py
"""
import torch
import torch.nn as nn


class Viewer(nn.Module):
    """ PyTorch module for extracting the middle image of a concatenated stack.

    Example: you have 10 RGB images stacked in a channel of a tensor, so it has shape [N, 30, H, W].
        viewer = Viewer(10)
        middle = viewer(tensor)
        print(middle.shape) # [N, 3, H, W], taken from channels 15:18

    """

    def __init__(self, num_images, label_location):
        super().__init__()
        self.num_images = num_images
        if label_location == 'middle':
            self.start = int(num_images / 2 * 3)
        elif label_location == 'causal':
            self.start = int(num_images * 3 - 3)
        self.end = int(self.start + 3)

    def forward(self, x):
        x = x[:, self.start:self.end, :, :]
        return x


class HiddenTwoStream(nn.Module):
    """ Hidden Two-Stream Network model
    Paper: https://arxiv.org/abs/1704.00389

    Classifies video inputs using a spatial CNN, using RGB video frames as inputs; and using a flow CNN, which
    uses optic flow as inputs. Optic flow is generated on-the-fly by a flow generator network. This has distinct
    advantages, as optic flow loaded from disk is both more discrete and has compression artifacts.
    """

    def __init__(self, flow_generator, spatial_classifier, flow_classifier, fusion,
                 classifier_name: str, num_images: int = 11,
                 label_location: str = 'middle'):
        """ Hidden two-stream constructor.

        Args:
            flow_generator (nn.Module): CNN that generates optic flow from a stack of RGB frames
            spatial_classifier (nn.Module): CNN that classifies original RGB inputs
            flow_classifier (nn.Module): CNN that classifies optic flow inputs
            classifier_name (str): name of CNN (e.g. resnet18) used in both classifiers
            num_images (int): number of input images to the flow generator. Flow outputs will be num_images - 1
            label_location (str): either middle or causal. Middle: the label will be selected from the middle of a
                stack of image frames. Causal: the label will come from the last image in the stack (no look-ahead)

        """
        super().__init__()
        assert (isinstance(flow_generator, nn.Module) and isinstance(spatial_classifier, nn.Module)
                and isinstance(flow_classifier, nn.Module) and isinstance(fusion, nn.Module))

        self.spatial_classifier = spatial_classifier
        self.flow_generator = flow_generator
        self.flow_classifier = flow_classifier
        if '3d' in classifier_name.lower():
            self.viewer = nn.Identity()
        else:
            self.viewer = Viewer(num_images, label_location)
        self.fusion = fusion

        self.frozen_state = {}
        self.freeze('flow_generator')

    def freeze(self, submodel_to_freeze: str):
        """ Freezes a component of the model. Useful for curriculum training

        Args:
            submodel_to_freeze (str): one of flow_generator, spatial, flow, fusion
        """
        if submodel_to_freeze == 'flow_generator':
            self.flow_generator.eval()
            for param in self.flow_generator.parameters():
                param.requires_grad = False
        elif submodel_to_freeze == 'spatial':
            self.spatial_classifier.eval()
            for param in self.spatial_classifier.parameters():
                param.requires_grad = False
        elif submodel_to_freeze == 'flow':
            self.flow_classifier.eval()
            for param in self.flow_classifier.parameters():
                param.requires_grad = False
        elif submodel_to_freeze == 'fusion':
            self.fusion.eval()
            for param in self.fusion.parameters():
                param.requires_grad = False
        else:
            raise ValueError('submodel not found:%s' % submodel_to_freeze)
        self.frozen_state[submodel_to_freeze] = True

    def set_mode(self, mode: str):
        """ Freezes and unfreezes portions of the model, useful for curriculum training.

        Args:
            mode (str): one of spatial_only, flow_only, fusion_only, classifier, end_to_end, or inference
        """
        if mode == 'spatial_only':
            self.freeze('flow_generator')
            self.freeze('flow')
            self.freeze('fusion')
            self.unfreeze('spatial')
        elif mode == 'flow_only':
            self.freeze('flow_generator')
            self.freeze('spatial')
            self.unfreeze('flow')
            self.freeze('fusion')
        elif mode == 'fusion_only':
            self.freeze('flow_generator')
            self.freeze('spatial')
            self.freeze('flow')
            self.unfreeze('fusion')
        elif mode == 'classifier':
            self.freeze('flow_generator')
            self.unfreeze('spatial')
            self.unfreeze('flow')
            self.unfreeze('fusion')
        elif mode == 'end_to_end':
            self.unfreeze('flow_generator')
            self.unfreeze('spatial')
            self.unfreeze('flow')
            self.unfreeze('fusion')
        elif mode == 'inference':
            self.freeze('flow_generator')
            self.freeze('spatial')
            self.freeze('flow')
            self.freeze('fusion')
        else:
            raise ValueError('Unknown mode: %s' % mode)

    def unfreeze(self, submodel_to_unfreeze: str):
        """ Unfreezes portions of the model

        Args:
            submodel_to_unfreeze (str): one of flow_generator, spatial, flow, or fusion

        Returns:

        """
        if submodel_to_unfreeze == 'flow_generator':
            self.flow_generator.train()
            for param in self.flow_generator.parameters():
                param.requires_grad = True
        elif submodel_to_unfreeze == 'spatial':
            self.spatial_classifier.train()
            for param in self.spatial_classifier.parameters():
                param.requires_grad = True
        elif submodel_to_unfreeze == 'flow':
            self.flow_classifier.train()
            for param in self.flow_classifier.parameters():
                param.requires_grad = True
        elif submodel_to_unfreeze == 'fusion':
            self.fusion.train()
            for param in self.fusion.parameters():
                param.requires_grad = True
        else:
            raise ValueError('submodel not found:%s' % submodel_to_unfreeze)
        self.frozen_state[submodel_to_unfreeze] = False

    def get_param_groups(self):
        param_list = [{'params': self.flow_generator.parameters()},
                      {'params': self.spatial_classifier.parameters()},
                      {'params': self.flow_classifier.parameters()},
                      {'params': self.fusion.parameters()}]
        return (param_list)

    def forward(self, batch):
        with torch.no_grad():
            flows = self.flow_generator(batch)
        RGB = self.viewer(batch)
        spatial_features = self.spatial_classifier(RGB)
        # flows[0] because flow returns a pyramid of spatial resolutions, zero being the highest res
        flow_features = self.flow_classifier(flows[0])
        return self.fusion(spatial_features, flow_features)


