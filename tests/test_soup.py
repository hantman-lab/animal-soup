from pathlib import Path
import os
from animal_soup import create_df, load_df
import shutil
import pandas as pd
from typing import *
from animal_soup.utils.dataframe import (DATAFRAME_COLUMNS,
                                         set_parent_raw_data_path,
                                         get_parent_raw_data_path)
import pytest
from tqdm import tqdm
from zipfile import ZipFile
import requests

tmp_dir = Path(os.path.dirname(os.path.abspath(__file__)), "tmp")
sample_data_dir = Path(os.path.dirname(os.path.abspath(__file__)), "sample_data")
ground_truth_dir = Path(os.path.dirname(os.path.abspath(__file__)), "ground_truth")
sample_data_file = Path(
    os.path.dirname(os.path.abspath(__file__)), "sample_data.zip"
)
ground_truth_file = Path(
    os.path.dirname(os.path.abspath(__file__)), "ground_truth.zip"
)

os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(sample_data_dir, exist_ok=True)
os.makedirs(ground_truth_dir, exist_ok=True)


def _download_ground_truth():
    print(f"Downloading ground truth data")
    url = f"https://zenodo.org/record/8172852/files/ground_truth.zip"

    # basically from https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(ground_truth_file, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    ZipFile(ground_truth_file).extractall(ground_truth_dir.parent)


def _download_sample_data():
    print(f"Downloading sample data")
    url = f"https://zenodo.org/record/8172852/files/sample_data.zip"

    # basically from https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(sample_data_file, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    ZipFile(sample_data_file).extractall(sample_data_dir.parent)


if len(list(sample_data_dir.iterdir())) == 0:
    _download_sample_data()

elif "DOWNLOAD_SAMPLE_DATA" in os.environ.keys():
    if os.environ["DOWNLOAD_SAMPLE_DATA"] == "1":
        _download_sample_data()

if len(list(ground_truth_dir.iterdir())) == 0:
    _download_ground_truth()

elif "DOWNLOAD_GROUND_TRUTH" in os.environ.keys():
    if os.environ["DOWNLOAD_GROUND_TRUTH"] == "1":
        _download_ground_truth()


def get_tmp_filename():
    return os.path.join(tmp_dir, f"test_df.hdf")


def clear_tmp():
    shutil.rmtree(tmp_dir)


def _create_tmp_df() -> Tuple[pd.DataFrame, str]:
    fname = get_tmp_filename()
    df = create_df(fname)

    return df, fname


def test_create_df() -> Tuple[pd.DataFrame, str]:
    fname = get_tmp_filename()
    if os.path.exists(fname):
        # assert attempting to create df with remove_existing=True
        df = create_df(fname, remove_existing=True)

        # make sure appropriate columns exist
        for c in DATAFRAME_COLUMNS:
            assert c in df.columns

        # assert dataframe is empty
        assert (len(df.index) == 0)
        # assert dataframe is empty animal_soup df
        empty_df = pd.DataFrame(columns=DATAFRAME_COLUMNS)
        pd.testing.assert_frame_equal(empty_df, df)

    else:  # tmp df does not exist
        # test creating a df
        df, fname = _create_tmp_df()

        # make sure appropriate columns exist
        for c in DATAFRAME_COLUMNS:
            assert c in df.columns

        # assert dataframe is empty
        assert (len(df.index) == 0)

        # assert dataframe is empty animal_soup df
        empty_df = pd.DataFrame(columns=DATAFRAME_COLUMNS)
        pd.testing.assert_frame_equal(empty_df, df)

    # assert attempting to create df at same path raises
    with pytest.raises(FileExistsError):
        create_df(fname)


def test_add_item():
    # set parent raw data path to sample data dir
    set_parent_raw_data_path(sample_data_dir)

    # assert path is as expected
    assert (get_parent_raw_data_path(), sample_data_dir)

    # create empty dataframe, remove existing if True
    fname = get_tmp_filename()
    df = create_df(fname, remove_existing=True)

    # get animal_ids in sample data
    animal_dirs = sorted(get_parent_raw_data_path().glob('M*'))

    # add all sessions for an animal
    animal_id1 = animal_dirs[0].stem
    df.behavior.add_item(animal_id=animal_id1)

    # assert df is as expected
    ground_df = pd.read_hdf(ground_truth_dir.joinpath('add_all_sessions.hdf'))
    pd.testing.assert_frame_equal(ground_df, df)

    # add all trials for a given session
    animal_id2 = animal_dirs[1].stem
    session_dirs = sorted(animal_dirs[1].glob('*'))
    df.behavior.add_item(animal_id=animal_id2, session_id=session_dirs[0].stem)

    # assert df is as expected
    ground_df = pd.read_hdf(ground_truth_dir.joinpath('add_single_session.hdf'))
    pd.testing.assert_frame_equal(ground_df, df)

    # add a single trial for a given animal/session
    trials = sorted(session_dirs[1].glob('*'))
    for trial in trials:
        df.behavior.add_item(animal_id=animal_id2, session_id=session_dirs[1].stem, trial_id=trial.stem)

    # assert df is as expected
    ground_df = pd.read_hdf(ground_truth_dir.joinpath('add_trials.hdf'))
    pd.testing.assert_frame_equal(ground_df, df)


def test_remove_item():
    # set parent raw data path to sample data dir
    set_parent_raw_data_path(sample_data_dir)

    # assert path is as expected
    assert (get_parent_raw_data_path(), sample_data_dir)

    # create empty dataframe, remove existing if True
    fname = get_tmp_filename()
    df = create_df(fname, remove_existing=True)

    # get animal_ids in sample data
    animal_dirs = sorted(get_parent_raw_data_path().glob('M*'))

    # add all sessions for all animals
    for animal in animal_dirs:
        df.behavior.add_item(animal_id=animal.stem)

    # assert full dataframe is as expected
    ground_df = pd.read_hdf(ground_truth_dir.joinpath('ground_df.hdf'))
    pd.testing.assert_frame_equal(df, ground_df)

    # remove all sessions for a given animal
    animal_id1 = animal_dirs[0].stem
    df.behavior.remove_item(animal_id=animal_id1)

    # assert dataframe as expected
    ground_df = pd.read_hdf(ground_truth_dir.joinpath('remove_all_sessions.hdf'))
    pd.testing.assert_frame_equal(ground_df, df)

    # remove by index
    df.behavior.remove_item(row_ix=0)

    # assert dataframe as expected
    ground_df = pd.read_hdf(ground_truth_dir.joinpath('remove_row.hdf'))
    pd.testing.assert_frame_equal(ground_df, df)

    # remove all items from a given session
    sessions = sorted(animal_dirs[1].glob('*'))
    df.behavior.remove_item(animal_id=animal_dirs[1].stem, session_id=sessions[0].stem)

    # assert dataframe as expected
    ground_df = pd.read_hdf(ground_truth_dir.joinpath('remove_single_session.hdf'))
    pd.testing.assert_frame_equal(ground_df, df)

    # remove a single trial
    trials = sorted(sessions[1].glob('*.avi'))
    df.behavior.remove_item(animal_id=animal_dirs[1].stem, session_id=sessions[1].stem, trial_id=trials[0].stem)

    # assert dataframe as expected
    ground_df = pd.read_hdf(ground_truth_dir.joinpath('remove_single_trial.hdf'))
    pd.testing.assert_frame_equal(ground_df, df)


def test_load_df():
    # add items to a df
    test_add_item()

    # load the df and assert it is as expected
    df = load_df(get_tmp_filename())

    ground_df = pd.read_hdf(ground_truth_dir.joinpath('ground_df.hdf'))
    pd.testing.assert_frame_equal(ground_df, df)
