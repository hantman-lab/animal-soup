"""Utility functions for creating datasets."""

import numpy as np
import warnings


def prepare_label(label: np.ndarray) -> np.ndarray:
    """Function to take original ethograms and add background column."""
    # transpose
    label = label.T
    # add background row
    rows = list()
    for row in label:
        # sum across all behaviors at a given time point
        # if the sum is zero (no behavior labeled at this time point),
        # set background = 1
        if sum(row) > 0:
            background = 0
        else:
            background = 1
        row = np.insert(row, 0, background)
        rows.append(row)

    # returns array of shape (time points, # behaviors)
    return np.array(rows)


def make_loss_weight(
        num_pos: np.ndarray,
        num_neg: np.ndarray) -> np.ndarray:
    """
    Makes weight for sigmoid loss function.

    Parameters
    ----------
    class_counts: np.ndarray, shape (K, )
        Number of positive examples in dataset
    num_pos: np.ndarray, shape (K, )
        number of positive examples in dataset
    num_neg: np.ndarray, shape (K, )
        number of negative examples in dataset

    Returns
    -------
    pos_weight_transformed: np.ndarray, shape (K, )
        Amount to weight each class. Used with sigmoid activation, BCE loss

        Note: pos_weight is the relative amount of positive labels for each behavior.
        Want this to be calculated in order to factor into the loss function for
        the feature extractor.

    """

    # if there are zero positive examples, we don't want the pos weight to be a large number
    # we want it to be infinity, then we will manually set it to zero
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        pos_weight = num_neg / num_pos
    # if there are zero negative examples, loss should be 1
    pos_weight[pos_weight == 0] = 1
    pos_weight_transformed = pos_weight.astype(np.float32)
    # if all examples positive: will be 1
    # if zero examples positive: will be 0
    pos_weight_transformed = np.nan_to_num(pos_weight_transformed, nan=0.0, posinf=0.0, neginf=0)

    return pos_weight_transformed

