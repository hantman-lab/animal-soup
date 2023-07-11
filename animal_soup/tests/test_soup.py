# TODO: write tests to make sure things do not break

from pathlib import Path
import os
from animal_soup import create_df
import shutil
import pandas as pd
from typing import *
from animal_soup.batch_utils import DATAFRAME_COLUMNS
import pytest

tmp_dir = Path(os.path.dirname(os.path.abspath(__file__)), "tmp")

os.makedirs(tmp_dir, exist_ok=True)

def get_tmp_filename():
    return os.path.join(tmp_dir, f"test_df.hdf")

def clear_tmp():
    shutil.rmtree(tmp_dir)

def _create_tmp_df() -> Tuple[pd.DataFrame, str]:
    fname = get_tmp_filename()
    df = create_df(fname)

    return df, fname

def test_create_df() -> Tuple[pd.DataFrame, str]:
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

    # assert attempting to create df with remove_existing=True
    df = create_df(fname, remove_existing=True)

    # make sure appropriate columns exist
    for c in DATAFRAME_COLUMNS:
        assert c in df.columns

    # assert dataframe is empty
    assert (len(df.index) == 0)






    # test add item

    # test remove item

    # test load df

if __name__ == "__main__":
    clear_tmp()