from typing import Union
from ..batch_utils import get_parent_raw_data_path, validate_path, load_df
import os
import pandas as pd
from pathlib import Path
import shutil
import time


@pd.api.extensions.register_dataframe_accessor("behavior")
class BehaviorDataFrameExtension:
    def __init__(self, df):
        self._df = df

    def add_item(
            self,
            animal_id: str,
            session_id: Union[str, None] = None):
        """
        Add item to dataframe, if session_id is None will add all available sessions for that animal.
        """

        if get_parent_raw_data_path() is None:
            raise ValueError(
                "parent raw data path is not set, you must set it using:\n"
                "`set_parent_raw_data_path()`"
            )

        PARENT_DATA_PATH = get_parent_raw_data_path()

        animal_dir = PARENT_DATA_PATH.joinpath(animal_id)

        validate_path(animal_dir)

        if not os.path.exists(animal_dir):
            raise ValueError(
                f"animal_id path is not valid at: {animal_dir}"
            )


        if session_id is None:
            # gets all sessions under an animal_id
            session_dirs = sorted(animal_dir.glob('*'))
            if len(session_dirs) == 0:
                raise ValueError("no sessions found in this animal dir")
            for session_dir in session_dirs:
                # check if already in dataframe # if so raise error
                if len(self._df.index[
                           (self._df['animal_id'] == animal_id) & (self._df['session_id'] == session_dir.stem)].tolist()) > 0:
                    raise ValueError(
                        f"item already exists with animal_id={animal_id} and session_id={session_dir.stem} \n"
                        f"must remove item before trying to add"
                    )
                session_id = session_dir.stem
                mat_path = session_dir.joinpath('jaaba.mat')
                mat_path = self._df.paths.split(mat_path)[-1]
                if os.path.exists(session_dir.joinpath('deg_labels.npy')):
                    deg_path = session_dir.joinpath('deg_labels.npy')
                    deg_path = self._df.paths.split(deg_path)[-1]
                else:
                    deg_path = None
                session_vids = sorted(session_dir.glob('*.avi'))
                session_vids = [self._df.paths.split(vid_path)[-1] for vid_path in session_vids]

                s = pd.Series(
                    {
                        "animal_id": animal_id,
                        "session_id": session_id,
                        "mat_path": mat_path,
                        "deg_path": deg_path,
                        "session_vids": session_vids,
                        "training_trials": list(),
                        "notes": None
                    }
                )

                self._df.loc[self._df.index.size] = s

                self._df.to_hdf(self._df.paths.get_df_path(), key='df')
        else:
            # check if already in dataframe # if so raise error
            if len(self._df.index[
                       (self._df['animal_id'] == animal_id) & (self._df['session_id'] == session_id)].tolist()) > 0:
                raise ValueError(
                    f"item already exists with animal_id={animal_id} and session_id{session_id} \n"
                    f"must remove item before trying to add"
                )
            session_dir = animal_dir.joinpath(session_id)
            mat_path = session_dir.joinpath('jaaba.mat')
            mat_path = self._df.paths.split(mat_path)[-1]
            if os.path.exists(session_dir.joinpath('deg_labels.npy')):
                deg_path = session_dir.joinpath('deg_labels.npy')
                deg_path = self._df.paths.split(deg_path)[-1]
            else:
                deg_path = None
            session_vids = sorted(session_dir.glob('*.avi'))
            session_vids = [self._df.paths.split(vid_path)[-1] for vid_path in session_vids]
            s = pd.Series(
                {
                    "animal_id": animal_id,
                    "session_id": session_id,
                    "mat_path": mat_path,
                    "deg_path": deg_path,
                    "session_vids": session_vids,
                    "training_trials": list(),
                    "notes": None
                }
            )

            self._df.loc[self._df.index.size] = s

            self._df.to_hdf(self._df.paths.get_df_path(), key='df')

    def save_to_disk(self, max_index_diff: int = 0):
        """
        Saves DataFrame to disk, copies to a backup before overwriting existing file.
        """
        path: Path = self._df.paths.get_df_path()

        disk_df = load_df(path)

        # check that max_index_diff is not exceeded
        if abs(disk_df.index.size - self._df.index.size) > max_index_diff:
            raise IndexError(
                f"The number of rows in the DataFrame on disk differs more "
                f"than has been allowed by the `max_index_diff` kwarg which "
                f"is set to <{max_index_diff}>. This is to prevent overwriting "
                f"the full DataFrame with a sub-DataFrame. If you still wish "
                f"to save the smaller DataFrame, use `caiman.save_to_disk()` "
                f"with `max_index_diff` set to the highest allowable difference "
                f"in row number."
            )

        bak = path.with_suffix(".bak")

        shutil.copyfile(path, bak)
        try:
            self._df.to_hdf(path, key='df')
            os.remove(bak)
        except:
            shutil.copyfile(bak, path)
            raise IOError(f"Could not save dataframe to disk.")

    def remove_item(self, animal_id: str,  session_id: Union[str, None] = None):
        """Remove item from dataframe."""

        if session_id is None:
            # remove all items for animal
            index = self._df.index[self._df['animal_id'] == animal_id].tolist()
        else:
            # remove single item that matches animal_id/session_id
            index = self._df.index[(self._df['animal_id'] == animal_id) & (self._df['session_id'] == session_id)].tolist()

        # Drop selected index
        for ix in index:
            self._df.drop([ix], inplace=True)
            # Reset indices so there are no 'jumps'
        self._df.reset_index(drop=True, inplace=True)
        # Save new df to disc
        self._df.to_hdf(self._df.paths.get_df_path(), key='df')