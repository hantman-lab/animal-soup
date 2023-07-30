import torch.nn
from .models import TGMJ
from ..data import SingleSequenceDataset
from pathlib import Path
import numpy as np


def predict_single_video(
    vid_path: Path,
    sequence_model: TGMJ,
    features: dict,
    nonoverlapping: bool = True,
    sequence_length: int = 60,

):
    """
    Predict sequence probabilities and logits for a single video.

    Parameters
    ----------
    vid_path: Path
        Path to current video trial.
    sequence_model: TGMJ
        Sequence model being used for inference
    features: dict
        Dictionary output of sequence inference.
    nonoverlapping: bool, default True
        Indicates whether sequences should overlap or not.
    sequence_length: int, default 60
        Sequence length.

    Returns
    -------
        dictionary of probabilities and logits that can be used for post-processing and
        finalizing ethograms

    """
    torch.backends.cudnn.benchmark = True
    device = torch.device("cpu")

    activation_function = torch.nn.Sigmoid()

    if next(sequence_model.parameters()).device != device:
        sequence_model = sequence_model.to(device)

    if next(sequence_model.parameters()).requires_grad:
        for parameter in sequence_model.parameters():
            parameter.requires_grad = False

    if sequence_model.training:
        sequence_model = sequence_model.eval()

    gen = SingleSequenceDataset(
            vid_path=vid_path,
            label=None,
            feature=features,
            sequence_length=sequence_length,
            nonoverlapping=nonoverlapping
    )
    n_datapoints = gen.shape[1]
    gen = torch.utils.data.DataLoader(gen, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    gen = iter(gen)

    all_logits = list()
    all_probs = list()

    for i in range(len(gen)):

        with torch.no_grad():
            batch = next(gen)
            features = batch['features'].to(device)
            logits = sequence_model(features)

            probabilities = activation_function(logits).detach().cpu().numpy().squeeze().T
            logits = logits.detach().cpu().numpy().squeeze().T

            end = min(i * sequence_length + sequence_length, n_datapoints)
            indices = range(i * sequence_length, end)
            # get rid of padding in final batch
            if len(indices) < logits.shape[0]:
                logits = logits[:len(indices), :]
                probabilities = probabilities[:len(indices), :]

            all_logits.append(logits)
            all_probs.append(probabilities)

    logits = np.concatenate(all_logits)
    probabilities = np.concatenate(all_probs)

    return {"logits": logits, "probabilities": probabilities}
