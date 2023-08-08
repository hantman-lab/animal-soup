"""Ethogram utility functions."""

from typing import *
from pathlib import Path
import os
import h5py
import numpy as np
import pandas as pd
from ..utils import get_parent_raw_data_path


# from scipy.io import loadmat
# import numpy as np


def get_ethogram_from_disk(row: pd.Series):
    """Returns whether ethogram exists for trial at given output_path."""

    output_path = get_parent_raw_data_path().joinpath(row["output_path"])

    if not os.path.exists(Path(output_path)):
        raise ValueError("No output path exists for this session. Please run inference for "
                         "trials in the current dataframe to generate ethograms for viewing.")

    with h5py.File(output_path, "r") as f:
        # check if trial in keys
        if str(row["trial_id"]) not in f.keys():
            raise ValueError("Inference has not been run for this trial yet. Please run "
                             "inference on ALL trials in the dataframe before trying to "
                             "view them.")

        # check if sequence inference has been run
        elif "sequence" not in f[str(row["trial_id"])].keys():
            raise ValueError("Sequence inference has not been run for this trial yet. Please "
                             "make sure that feature extraction and sequence inference has been "
                             "completed for ALL trials in the dataframe before trying to view generated "
                             "ethograms.")

        # ethogram exists! want to return cleaned ethogram if possible
        if "cleaned_ethogram" in f[str(row["trial_id"])]["ethograms"].keys():
            return f[str(row["trial_id"])]["ethograms"]["cleaned_ethogram"][:]
        else:
            return f[str(row["trial_id"])]["ethograms"]["ethogram"][:]


def save_ethogram_to_disk(row: pd.Series, cleaned_ethogram: np.ndarray):
    """Saves a cleaned ethogram to disk in session output file."""

    output_path = get_parent_raw_data_path().joinpath(row["output_path"])

    # update h5 file with cleaned_ethogram
    with h5py.File(output_path, "r+") as f:
        # if exists, delete and regenerate, else just create
        if "cleaned_ethogram" in f[str(row["trial_id"])]["ethograms"].keys():
            del f[str(row["trial_id"])]["ethograms"]["cleaned_ethogram"]

        ethogram_group = f[str(row["trial_id"])]["ethograms"]

        ethogram_group.create_dataset("cleaned_ethogram",
                                      data=cleaned_ethogram)


# for now, a place to stor the method for getting a jaaba ethogram (hand_label or jaaba pred)


# def _get_ethogram(trial_index: int, mat_path, ethogram_type: str):
#         """
#         Returns the ethogram for a given trial in a session.
#         """
#         m = loadmat(mat_path)
#         behaviors = sorted([b.split('_')[0] for b in m['data'].dtype.names if 'scores' in b])
#
#         all_behaviors = [
#             "Lift",
#             "Handopen",
#             "Grab",
#             "Sup",
#             "Atmouth",
#             "Chew"
#         ]
#
#         sorted_behaviors = [b for b in all_behaviors if b in behaviors]
#
#         ethograms = []
#
#         mat_trial_index = np.argwhere(m["data"]["trial"].ravel() == (trial_index + 1))
#         # Trial not found in JAABA data
#         if mat_trial_index.size == 0:
#             return False
#
#         mat_trial_index = mat_trial_index.item()
#
#         if ethogram_type == 'hand-labels':
#             for b in sorted_behaviors:
#                 behavior_index = m['data'].dtype.names.index(f'{b}_labl_label')
#                 row = m['data'][mat_trial_index][0][behavior_index]
#                 row[row == -1] = 0
#                 ethograms.append(row)
#         else:
#             for b in sorted_behaviors:
#                 behavior_index = m['data'].dtype.names.index(f'{b}_postprocessed')
#                 row = m['data'][mat_trial_index][0][behavior_index]
#                 row[row == -1] = 0
#                 ethograms.append(row)
#
#         sorted_behaviors = [b.lower() for b in sorted_behaviors]
#
#         return np.hstack(ethograms).T, sorted_behaviors
