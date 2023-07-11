# TODO: write tests to make sure things do not break
import datetime
from pathlib import Path
import os
from animal_soup import create_df, load_df
import shutil
import pandas as pd
from typing import *
from animal_soup.batch_utils import DATAFRAME_COLUMNS, set_parent_raw_data_path, get_parent_raw_data_path
import pytest
from datetime import date

tmp_dir = Path(os.path.dirname(os.path.abspath(__file__)), "tmp")
sample_data_dir = Path(os.path.dirname(os.path.abspath(__file__)), "sample_data")

os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(sample_data_dir, exist_ok=True)

# will need to have something similar to mescore
# where sample data gets downloaded from zenodo
# for now just have it locally

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

    else: # tmp df does not exist
        # test creating a df
        df, fname = _create_tmp_df()

        # make sure appropriate columns exist
        for c in DATAFRAME_COLUMNS:
            assert c in df.columns

        # assert dataframe is empty
        assert(len(df.index) == 0)

    # assert attempting to create df at same path raises
    with pytest.raises(FileExistsError):
        create_df(fname)

def test_add_item():
    # set parent raw data path to sample data dir
    set_parent_raw_data_path(sample_data_dir)

    # assert path is as expected
    assert(get_parent_raw_data_path(), sample_data_dir)

    # create empty dataframe, remove existing if True
    fname = get_tmp_filename()
    df = create_df(fname, remove_existing=True)

    # get animal_ids in sample data
    animal_ids = sorted(get_parent_raw_data_path().glob('*M'))




    fname = get_tmp_filename()

    if not os.path.exists(fname):
        test_create_df()

    df = load_df(fname)

    # test adding item when animal_id/session_id given
    df.behavior.add_item(animal_id=animal_id, session_id=session_id)

    assert(len(df.index) == 1)

    # test adding item when only animal_id given
    df.behavior.add_item(animal_id=animal_id)

    assert(len(df.index) == 2)


