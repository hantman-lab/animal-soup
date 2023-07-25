"""Utility functions for creating datasets."""

import numpy as np


def prepare_label(label: np.ndarray) -> np.ndarray:
    """Function to take original ethograms and add background column."""
    # transpose
    label = label.T
    # add background row
    rows = list()
    for r in label:
        if sum(r) > 0:
            background = 0
        else:
            background = 1
        r = np.insert(r, 0, background)
        rows.append(r)

    return np.array(rows)
