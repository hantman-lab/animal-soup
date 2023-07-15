from typing import Union
from ..df_utils import get_parent_raw_data_path, validate_path, load_df
from ..viewers import *
import os
import pandas as pd
from pathlib import Path
import shutil
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

    def compare_ethograms(self,
                          start_index: int = 0
                          ):
        container = EthogramComparison(
            dataframe=self._df,
            start_index=start_index
        )

        return container

    def add_item(
            self,
            animal_id: str,
            session_id: str = None,
            trial_id: str = None,
            exp_type: str = None
    ):
        """
        Add item to dataframe. If `animal_id`/'session_id'/'trial_id' already exists, will raise ValueError.
        Can add all sessions for an animal, a single session, or a single trial within a given session.

        Parameters
        ----------
        animal_id: str
            Animal to add to the dataframe.
        session_id: str, default None
            Session to be added for a given animal. If `None`, will attempt to add all sessions for the given animal.
        trial_id: str, default None
            Trial to be added for given animal/session. If `None`, will add all trials for the given session.
        exp_type: str, default None
            Type of experiment, either 'table' or 'pez'. Default as None, can be set later.

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
                f"animal_id path does not exist at: {animal_dir}"
            )

        # session_id = None, get all sessions under that animal_id
        if session_id is None:

            # if session_id is None, trial_id must also be None
            if trial_id is not None:
                raise ValueError("If session_id is None, trial_id must also be None.")

            # get all sessions under an animal_id
            session_dirs = sorted(animal_dir.glob('*'))

            # no sessions found, raise
            if len(session_dirs) == 0:
                raise ValueError(f"no sessions found in this animal dir: {animal_dir}")

            for session_dir in session_dirs:
                # get trials
                trials = sorted(session_dir.glob('*'))

                # no trials found
                if len(trials) == 0:
                    raise ValueError(f'no trials found in this session: {session_dir}')

                # add trials to dataframe
                for trial in trials:
                    # check if trial already in dataframe
                    if len(self._df[(self._df['animal_id'] == animal_id) &
                                    (self._df['session_id'] == session_dir.stem) &
                                    (self._df['trial_id'] == trial.stem)].index) > 0:
                        raise ValueError(f"Item already exists with animal_id={animal_id}, "
                                         f"session_id={session_dir.stem}, and trial_id={trial.stem}. "
                                         f"Please remove the item before attempting to add again.")

                    # add item to dataframe
                    s = pd.Series(
                        {
                            "animal_id": animal_id,
                            "session_id": session_dir.stem,
                            "trial_id": trial.stem,
                            "ethograms": None,
                            "exp_type": exp_type,
                            "notes": None
                        }
                    )

                    self._df.loc[self._df.index.size] = s

        else: # session_id is provided in args
            session_dir = animal_dir.joinpath(session_id)

            # if trial_id is None, add all
            if trial_id is None:
                trials = sorted(session_dir.glob('*'))

                if len(trials) == 0:
                    raise ValueError(f'no trials found in this session: {session_dir}')

                for trial in trials:
                    # check if trial already in dataframe
                    if len(self._df[(self._df['animal_id'] == animal_id) &
                                    (self._df['session_id'] == session_id) &
                                    (self._df['trial_id'] == trial.stem)].index) > 0:
                        raise ValueError(f"Item already exists with animal_id={animal_id}, "
                                         f"session_id={session_id}, and trial_id={trial.stem}. "
                                         f"Please remove the item before attempting to add again.")

                    # add item to dataframe
                    s = pd.Series(
                        {
                            "animal_id": animal_id,
                            "session_id": session_id,
                            "trial_id": trial.stem,
                            "ethograms": None,
                            "exp_type": exp_type,
                            "notes": None
                        }
                    )

                    self._df.loc[self._df.index.size] = s
            # trial id is not None, only adding one item
            else:
                # check if trial already in dataframe
                if len(self._df[(self._df['animal_id'] == animal_id) &
                                (self._df['session_id'] == session_id) &
                                (self._df['trial_id'] == trial_id)].index) > 0:
                    raise ValueError(f"Item already exists with animal_id={animal_id}, "
                                     f"session_id={session_id}, and trial_id={trial_id}. "
                                     f"Please remove the item before attempting to add again.")

                # add item to dataframe
                s = pd.Series(
                    {
                        "animal_id": animal_id,
                        "session_id": session_id,
                        "trial_id": trial_id,
                        "ethograms": None,
                        "exp_type": exp_type,
                        "notes": None
                    }
                )

                self._df.loc[self._df.index.size] = s

        # save df to disk
        self._df.to_hdf(self._df.paths.get_df_path(), key='df')

    def remove_item(self,
                    row_ix: int = None,
                    animal_id: str = None,
                    session_id: str = None,
                    trial_id: str = None):
        """
        Remove item(s) from dataframe.

        Parameters
        ----------
        row_ix: int, default None
            Row index to remove from dataframe, takes precedence over other arguments passed in.
        animal_id: str, default None
            Animal to remove from dataframe.
        session_id: str, default None
            Session to remove from dataframe. If `None`, will remove all sessions for the provided
            animal.
        trial_id: str, default None
            Trial to remove from dataframe
        """
        # simple fix for now to supress PerformanceWarning when saving dataframe
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

        # row_ix takes precedence over other args
        if row_ix is not None:
            index = [row_ix]
        elif session_id is None:
            warnings.warn("No `session_id` provided, will remove all items for provided "
                          f"animal_id: {animal_id}")
            # remove all items for animal
            index = self._df.index[self._df['animal_id'] == animal_id].tolist()
        else:
            if trial_id is not None:
                # remove single item that matches animal_id/session_id/trial_id
                index = self._df.index[(self._df['animal_id'] == animal_id) &
                                       (self._df['session_id'] == session_id) &
                                       (self._df['trial_id'] == trial_id)].tolist()
            else: # remove all item for given animal_id/session_id
                warnings.warn("No `trial_id` provided, will remove all items for provided "
                              f"animal_id: {animal_id} and session_id: {session_id}")
                index = self._df.index[(self._df['animal_id'] == animal_id) &
                                       (self._df['session_id'] == session_id)].tolist()
        # Drop selected index
        for ix in index:
            self._df.drop([ix], inplace=True)
            # Reset indices so there are no 'jumps'
        self._df.reset_index(drop=True, inplace=True)
        # Save new df to disc
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

    def _unsafe_save(self):
        path = self._df.paths.get_df_path()

        self._df.to_hdf(path, key='df')

