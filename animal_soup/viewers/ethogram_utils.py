from scipy.io import loadmat
import numpy as np

# for now, a place to stor the method for getting a jaaba ethogram (hand_label or jaaba pred)

# TODO: should really just make this function return the merged ethogram if possible,
#  merged of hand and jaaba pred will always be better than jaaba by itself


def _get_ethogram(trial_index: int, mat_path, ethogram_type: str):
        """
        Returns the ethogram for a given trial in a session.
        """
        m = loadmat(mat_path)
        behaviors = sorted([b.split('_')[0] for b in m['data'].dtype.names if 'scores' in b])

        all_behaviors = [
            "Lift",
            "Handopen",
            "Grab",
            "Sup",
            "Atmouth",
            "Chew"
        ]

        sorted_behaviors = [b for b in all_behaviors if b in behaviors]

        ethograms = []

        mat_trial_index = np.argwhere(m["data"]["trial"].ravel() == (trial_index + 1))
        # Trial not found in JAABA data
        if mat_trial_index.size == 0:
            return False

        mat_trial_index = mat_trial_index.item()

        if ethogram_type == 'hand-labels':
            for b in sorted_behaviors:
                behavior_index = m['data'].dtype.names.index(f'{b}_labl_label')
                row = m['data'][mat_trial_index][0][behavior_index]
                row[row == -1] = 0
                ethograms.append(row)
        else:
            for b in sorted_behaviors:
                behavior_index = m['data'].dtype.names.index(f'{b}_postprocessed')
                row = m['data'][mat_trial_index][0][behavior_index]
                row[row == -1] = 0
                ethograms.append(row)

        sorted_behaviors = [b.lower() for b in sorted_behaviors]

        return np.hstack(ethograms).T, sorted_behaviors