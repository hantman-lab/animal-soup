"""Utility functions for creating datasets."""

import numpy as np


def prepare_label(label: np.ndarray) -> np.ndarray:
    """Function to take original ethograms and add background column."""
    # transpose
    label = label.T
    # add background row
    rows = list()
    for row in label:
        # sum across all behaviors for a time point
        # insert in first position the background value
        # for that time point
        if sum(row) > 0:
            background = 0
        else:
            background = 1
        row = np.insert(row, 0, background)
        rows.append(row)

    # returns array of shape (time points, # behaviors)
    return np.array(rows)
