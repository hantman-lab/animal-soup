"""
Post-processing utility functions.

Adopted from: https://github.com/jbohnslav/deepethogram/blob/master/deepethogram/postprocessing.py
"""

import numpy as np


def _find_bout_indices(predictions_trace: np.ndarray,
                       bout_length: int,
                       positive: bool = True,
                       eps: float = 1e-6) -> np.ndarray:
    """Utility function for finding where bouts of each behavior occur so they can be removed if too short."""
    # make a filter for convolution that will be 1 at that bout center
    center = np.ones(bout_length) / bout_length
    filt = np.concatenate([[-bout_length / 2], center, [-bout_length / 2]])
    if not positive:
        predictions_trace = np.logical_not(predictions_trace.copy()).astype(int)
    out = np.convolve(predictions_trace, filt, mode='same')
    # precision issues: using == 1 here has false negatives in case where out = 0.99999999998 or something
    indices = np.where(np.abs(out - 1) < eps)[0]
    if len(indices) == 0:
        return np.array([]).astype(int)
    # if even, this corresponds to the center + 0.5 frame in the bout
    # if odd, this corresponds to the center frame of the bout
    # we want indices to contain the entire bout, not just the center frame
    if bout_length % 2:
        expanded = np.concatenate([np.array(range(i - bout_length // 2, i + bout_length // 2 + 1)) for i in indices])
    else:
        expanded = np.concatenate([np.array(range(i - bout_length // 2, i + bout_length // 2)) for i in indices])
    return expanded


def min_bout_post_process(
        predictions: dict,
        thresholds: np.ndarray,
        min_bout_length: int = 1
):
    """
    Post-processing technique to remove detected behaviors that occur in bouts less than the
    specified length.

    Parameters
    ----------
    thresholds: np.ndarray
        A 1D array representing the thresholds required to label a single time point as a certain behavior.
        For now, using default thresholds for the Hantman reach-to-grab task but should go back and
        implement calculating the thresholds during training of the feature extractor.
    predictions: dict
        Dictionary containing the probabilities and logits from sequence inference.
    min_bout_length: int, default 1
        The minimum default bout length a behavior must occur across consecutive time-points to be
        classified as occurring.

    Returns
    -------
    final_preds: np.ndarray
        An array of classified behaviors in the shape [# behaviors, # time points]. The "background"
        behavior used in training/inference is removed.
    """
    predictions = (predictions["probabilities"] > thresholds).astype(int)

    T, K = predictions.shape

    for k in range(K):
        predictions_trace = predictions[:, k]
        for bout_len in range(1, min_bout_length + 1):
            # first, remove "false negatives", like filling in gaps in true behavior bouts
            short_neg_indices = _find_bout_indices(predictions_trace, bout_len, positive=False)
            predictions_trace[short_neg_indices] = 1
            # then remove "false positives", very short "1" bouts
            short_pos_indices = _find_bout_indices(predictions_trace, bout_len)
            predictions_trace[short_pos_indices] = 0
        predictions[:, k] = predictions_trace

    # remove background
    final_preds = np.delete(predictions.T, 0, 0)

    return final_preds
