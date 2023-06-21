from typing import Union
from ..batch_utils import get_parent_raw_data_path, validate_path, load_df
from .._behavior import BehaviorVizContainer
from .._ethogram import EthogramVizContainer
from .._ethogram_cleaner import EthogramCleaner
import os
import pandas as pd
from pathlib import Path
import shutil
import time
import warnings


@pd.api.extensions.register_dataframe_accessor("behavior")
class BehaviorDataFrameExtension:
    def __init__(self, df):
        self._df = df

    def view(
            self,
            start_index: int = 0,
            ethogram_view: bool = True,
    ):
        if ethogram_view:
            container = EthogramVizContainer(
                dataframe=self._df,
                start_index=start_index
            )
        else:
            container = BehaviorVizContainer(
                dataframe=self._df,
                start_index=start_index,
            )

        return container

    def clean_ethograms(self,
                start_index: int = 0,
                        ):
        container = EthogramCleaner(
            dataframe=self._df,
            start_index=start_index,
        )

        return container

    def add_item(
            self,
            animal_id: str,
            session_id: Union[str, None] = None):
        """
        Add item to dataframe. If `animal_id` already exists, will try to add any new videos to
        `ethograms` dictionary for given session(s).

        Parameters
        ----------
        animal_id: str
            Animal to add to the dataframe.
        session_id: str or None, default None
            Session to be added for a given animal. If `None`, will attempt to add all sessions for
            the given animal.
        """
        # simple fix for now to supress PerformanceWarning when saving dataframe
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

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

        # session_id = None, get all sessions under that animal_id
        if session_id is None:
            # gets all sessions under an animal_id
            session_dirs = sorted(animal_dir.glob('*'))
            if len(session_dirs) == 0:
                raise ValueError(f"no sessions found in this animal dir: {animal_dir}")
            for session_dir in session_dirs:
                # check if already in dataframe
                # add new trials to ethograms if so
                df_ix = self._df.index[
                           (self._df['animal_id'] == animal_id) & (self._df['session_id'] == session_dir.stem)].tolist()
                if len(df_ix) > 0:
                    warnings.warn(
                        f"\n Item already exists with animal_id={animal_id} and session_id={session_dir.stem}. \n"
                        f"Will try to add any new videos. "
                    )
                    session_vids = sorted(session_dir.glob('*.avi'))
                    session_vids = [vid_path.stem for vid_path in session_vids]

                    for sv in session_vids:
                        if sv not in self._df.loc[:, "ethograms"].loc[df_ix].keys():
                            self._df.loc[:, "ethograms"].loc[df_ix][sv] = None
                else: # want to add all sessions
                    session_id = session_dir.stem
                    session_vids = sorted(session_dir.glob('*.avi'))
                    session_vids = [vid_path.stem for vid_path in session_vids]

                    ethograms = dict()

                    # add the keys for the available trial videos in a given session
                    for sv in session_vids:
                        ethograms[sv] = None

                    if "cleaned_ethograms" in self._df.columns:
                        s = pd.Series(
                            {
                                "animal_id": animal_id,
                                "session_id": session_id,
                                "ethograms": ethograms,
                                "cleaned_ethograms": ethograms,
                                "notes": None
                            }
                        )
                    else:
                        s = pd.Series(
                            {
                                "animal_id": animal_id,
                                "session_id": session_id,
                                "ethograms": ethograms,
                                "notes": None
                            }
                        )

                    self._df.loc[self._df.index.size] = s

                self._df.to_hdf(self._df.paths.get_df_path(), key='df')
        else: # session_id is provided in args
            # check if already in dataframe # if so raise error
            if len(self._df.index[
                       (self._df['animal_id'] == animal_id) & (self._df['session_id'] == session_id)].tolist()) > 0:
                warnings.warn(
                    f"\n Item already exists with animal_id={animal_id} and session_id{session_id}. \n"
                    f"Will try to add any new trial videos."
                )
                session_dir = animal_dir.joinpath(session_id)

                session_vids = sorted(session_dir.glob('*.avi'))
                session_vids = [vid_path.stem for vid_path in session_vids]

                for sv in session_vids:
                    if sv not in self._df.loc[:, "ethograms"].loc[df_ix].keys():
                        self._df.loc[:, "ethograms"].loc[df_ix][sv] = None

            else:
                session_dir = animal_dir.joinpath(session_id)
                session_vids = sorted(session_dir.glob('*.avi'))
                session_vids = [vid_path.stem for vid_path in session_vids]

                ethograms = dict()

                for sv in session_vids:
                    ethograms[sv] = None

                s = pd.Series(
                    {
                        "animal_id": animal_id,
                        "session_id": session_id,
                        "ethograms": ethograms,
                        "notes": None
                    }
                )

                self._df.loc[self._df.index.size] = s

            self._df.to_hdf(self._df.paths.get_df_path(), key='df')

    def save_to_disk(self, max_index_diff: int = 0):
        """
        Saves DataFrame to disk, copies to a backup before overwriting existing file.

        Parameters
        ----------
        max_index_diff: int, default 0
            Max difference in the number of indices between the current dataframe and the dataframe
            stored on disk. If index difference exceeds max, will raise `IndexError`.

        """
        # simple fix for now to supress PerformanceWarning when saving dataframe
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

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
        """
        Remove item(s) from dataframe.

        Parameters
        ----------

        animal_id: str
            Animal to remove from dataframe.
        session_id: str or None, default None
            Session to remove from dataframe. If `None`, will remove all sessions for the provided
            animal.
        """
        # simple fix for now to supress PerformanceWarning when saving dataframe
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

        if session_id is None:
            warnings.warn("No `session_id` provided, will remove all items for provided "
                          f"animal_id: {animal_id}")
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