import numpy as np

from .test_df import sample_data_dir, get_tmp_filename
from animal_soup.utils.dataframe import (
                                         set_parent_raw_data_path,
                                         get_parent_raw_data_path)
from animal_soup import create_df
from animal_soup.utils import get_normalization


def test_normalization():
    """Test zscore process for batch of videos using front/side concat on fly."""
    print("testing video normalization")
    # set parent raw data path to sample data dir
    set_parent_raw_data_path(sample_data_dir)

    # assert path is as expected
    assert (get_parent_raw_data_path(), sample_data_dir)

    # create empty dataframe, remove existing if True
    fname = get_tmp_filename("soup")
    df = create_df(fname, remove_existing=True)

    # get animal_ids in sample data
    animal_dirs = sorted(get_parent_raw_data_path().glob('M*'))

    # add all sessions for an animal
    animal_id1 = animal_dirs[0].stem
    df.behavior.add_item(animal_id=animal_id1)

    vid_paths = list(df["vid_paths"].values)

    norm = get_normalization(vid_paths)

    assert(norm["N"] == 299520000)

    ground_mean = np.array([0.02168639, 0.02168639, 0.02168639])
    assert(np.allclose(norm["mean"], ground_mean))

    ground_std = np.array([0.01692407, 0.01692407, 0.01692407])
    assert (np.allclose(norm["std"], ground_std))



