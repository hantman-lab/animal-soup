from torch.utils import data
from typing import *
from pathlib import Path
import numpy as np
import torch
from .utils import *


class SequenceDataset(data.Dataset):
    """ Simple wrapper around SingleSequenceDataset for smoothly loading multiple sequences """

    def __init__(self,
                 vid_paths: List[Path],
                 labels: List[np.ndarray],
                 features: List[dict],
                 nonoverlapping: bool = True,
                 sequence_length: int = 180,
                 ):
        """
        Parameters
        ----------
        vid_paths: List[Path]
            List of video paths corresponding to trials in the current dataframe.
        labels: List[np.ndarray]
            List of labels corresponding to each trial in the current dataframe.
        features: List[dict]
            List of extracted features from feature extractor inference corresponding to each trial in the current
            dataframe.
        nonoverlapping: bool, default True
            If True, indexing into dataset will return non-overlapping sequences. With a sequence length of 10,
            for example,
            if nonoverlapping:
                sequence[0] contains data from frames [0:10], sequence[1]: frames [11:20], etc...
            else:
                sequence[0] contains data from frames [0:10], sequence[1]: frames[1:11], etc...
        sequence_length: int, default 180
            Number of elements in sequence
        """

        datasets = list()
        dataset_info = list()

        self.class_counts = 0
        self.num_pos = 0
        self.num_neg = 0
        self.num_features = labels[0].shape[1]

        for i, (vid_path, label, feature) in enumerate(zip(vid_paths, labels, features)):
            dataset = SingleSequenceDataset(
                vid_path=vid_path,
                label=label,
                feature=feature,
                nonoverlapping=nonoverlapping,
                sequence_length=sequence_length)

            datasets.append(dataset)
            dataset_info.append(dataset.metadata)

            self.class_counts += dataset.class_counts
            self.num_pos += dataset.num_pos
            self.num_neg += dataset.num_neg

        self.num_features = dataset.num_features
        self.pos_weight = make_loss_weight(self.num_pos, self.num_neg)
        self.dataset = data.ConcatDataset(datasets)
        self.dataset_info = dataset_info

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]


class SingleSequenceDataset(data.Dataset):
    """PyTorch Dataset for loading a set of saved 1d features and one-hot labels for Action Detection."""

    def __init__(self,
                 vid_path: Path,
                 label: np.ndarray,
                 feature: dict,
                 sequence_length: int = 60,
                 nonoverlapping: bool = True,
                 ):
        """

        Parameters
        ----------
        vid_path: Path
            Location of trial video.
        label: np.ndarray
            Trial labels in shape [# time points, # behaviors]
        feature: dict
            Dictionary generated from running feature extraction that contains the flow features, spatial features,
            probabilities, and logits.
        sequence_length: int, default 60
            Sequence length.
        nonoverlapping: bool, default True
            Determines whether sequences overlap or not.
        """

        self.vid_path = vid_path
        self.metadata = dict()

        flow_feats = feature["flow_features"].T
        image_feats = feature["spatial_features"].T
        sequence = np.concatenate((image_feats, flow_feats), axis=0)
        logits = feature["logits"].T

        self.feature = dict(features=sequence, logits=logits)

        # after transpose, label will be of shape N_behaviors x T
        self.label = prepare_label(label).T
        self.class_counts = (self.label == 1).sum(axis=1)
        self.num_pos = (self.label == 1).sum(axis=1)
        self.num_neg = np.logical_not((self.label == 1)).sum(axis=1)

        self.metadata["vid_path"] = self.vid_path
        self.metadata["class_counts"] = self.class_counts
        self.metadata["num_pos"] = self.num_pos
        self.metadata["num_neg"] = self.num_neg

        self.sequence_length = sequence_length
        self.nonoverlapping = nonoverlapping

        self.starts = None
        self.ends = None

        self.shape = self.feature['features'].shape
        self.N = self.shape[1]

        self.compute_starts_ends()

        tmp_sequence = self.__getitem__(0)  # self.read_sequence([0, 1])
        self.num_features = tmp_sequence['features'].shape[0]

    def read_sequence(self, indices):
        """Returns slice of feature data based on indices"""

        data = {
            'features': self.feature['features'][:, indices],
            'logits': self.feature['logits'][:, indices]
        }

        return data

    def compute_starts_ends(self):
        if self.nonoverlapping:
            self.starts = []
            self.ends = []

            starts = np.arange(self.N, step=self.sequence_length)
            ends = np.roll(np.copy(starts), -1)
            ends[-1] = starts[-1] + self.sequence_length

            self.starts = starts
            self.ends = ends
        else:
            inds = np.arange(self.N)
            self.starts = inds - self.sequence_length // 2
            self.ends = inds + self.sequence_length // 2 + self.sequence_length % 2

    def __len__(self):
        return len(self.starts)

    def compute_indices_and_padding(self, index):
        start = self.starts[index]
        end = self.ends[index]

        # sequences close to the 0th frame are padded on the left
        if start < 0:
            pad_left = np.abs(start)
            pad_right = 0
            start = 0
        elif end > self.N:
            pad_left = 0
            pad_right = end - self.N
            end = self.N
        else:
            pad_left = 0
            pad_right = 0

        indices = np.arange(start, end)

        pad = (pad_left, pad_right)

        label_indices = indices
        label_pad = pad

        assert (len(indices) + pad_left + pad_right) == self.sequence_length, \
            'indices: {} + pad_left: {} + pad_right: {} should equal seq len: {}'.format(
                len(indices), pad_left, pad_right, self.sequence_length)
        # if we are stacking in time, label indices should not be the sequence length
        assert (len(label_indices) + label_pad[0] + label_pad[1]) == self.sequence_length, \
            'label indices: {} + pad_left: {} + pad_right: {} should equal seq len: {}'.format(
                len(label_indices), label_pad[0], label_pad[1], self.sequence_length)

        return indices, label_indices, pad, label_pad

    def __getitem__(self, index: int) -> dict:
        indices, label_indices, pad, label_pad = self.compute_indices_and_padding(index)

        data = self.read_sequence(indices)

        # can be multiple things in "data", like "image features" and "logits" from feature extractors
        # all will be converted to float32
        output = {}
        pad_left, pad_right = pad
        for key, value in data.items():
            value = np.pad(value, ((0, 0), (pad_left, pad_right)), mode='constant')
            value = torch.from_numpy(value).float()
            output[key] = value

        pad_left, pad_right = label_pad

        labels = self.label[:, label_indices].astype(np.int64)
        if labels.ndim == 1:
            labels = labels[:, np.newaxis]
        labels = np.pad(labels, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=-1)
        labels = labels.squeeze()
        labels = torch.from_numpy(labels).to(torch.long)
        output['labels'] = labels

        if labels.ndim > 1 and labels.shape[1] != output['features'].shape[1]:
            import pdb
            pdb.set_trace()

        return output
