from ..utils import get_parent_raw_data_path, validate_path, split_path, load_df
from ..viewers import *
import os
import pandas as pd
from pathlib import Path
import shutil
import warnings

ACCEPTED_CODECS = ['.avi', '.mp4', '.mov', '.mpeg']


@pd.api.extensions.register_dataframe_accessor("behavior")
class BehaviorDataFrameExtension:
    def __init__(self, df):
        self._df = df

    def view(
        self,
        start_index: int = 0,
        ethogram_view: bool = True,
        ethogram_mode: str = "inference"
    ):
        """
        View behavior with or without ethograms.

        Parameters
        ----------
        start_index: int, default 0
            Row index to start visualization from.
        ethogram_view: bool, default True
            Indicates if ethograms should be viewed along with behavior or not. Set to false if you just want to
            view your behavior videos in the current dataframe.
        ethogram_mode: str, default 'inference'
            One of ['ground', 'inference']. Indicates where you want to load ethograms from. In 'ground' mode,
             ethograms will be looked for in the current dataframe. In 'inference' mode, ethograms will be
             looked for in output files saved on disk.
        Returns
        -------
        container
            Object that contains the datagrid of trials in the dataframe as well as behavior viewer and, if applicable,
            an ethogram viewer plot.
        """
        if ethogram_view:
            container = EthogramVizContainer(
                dataframe=self._df, start_index=start_index, mode=ethogram_mode
            )
        else:
            container = BehaviorVizContainer(
                dataframe=self._df,
                start_index=start_index,
            )

        return container

    def clean_ethograms(
        self,
        start_index: int = 0,
        ethogram_mode: str = "inference"
    ):
        """
        Clean ethograms.

        Parameters
        ----------
        start_index: int, default 0
            Row index to start visualization from.
        ethogram_mode: str, default 'inference'
            One of ['ground', 'inference']. Indicates where you want to load ethograms from. In 'ground' mode,
             ethograms will be looked for in the current dataframe. In 'inference' mode, ethograms will be
             looked for in output files saved on disk.

        Returns
        -------
        container
            Object that contains the datagrid of trials in the dataframe as well as behavior viewer and ethogram
            cleaner plot.
        """
        container = EthogramCleanerVizContainer(
            dataframe=self._df,
            start_index=start_index,
            mode=ethogram_mode
        )

        return container

    def compare_ethograms(self, start_index: int = 0):
        container = EthogramComparisonVizContainer(
            dataframe=self._df, start_index=start_index
        )

        return container

    def add_item(
        self,
        animal_id: str,
        session_id: str = None,
        trial_id: int = None,
        exp_type: str = None,
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
        trial_id: int, default None
            Trial to be added for given animal/session. If `None`, will add all trials for the given session.
            Should be an integer representing the trial #.
        exp_type: str, default None
            Type of experiment, either 'table' or 'pez'. Default as None, can be set later.

        """
        # simple fix for now to supress PerformanceWarning when saving dataframe
        warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

        if get_parent_raw_data_path() is None:
            raise ValueError(
                "parent raw data path is not set, you must set it using:\n"
                "`set_parent_raw_data_path()`"
            )

        PARENT_DATA_PATH = get_parent_raw_data_path()

        animal_dir = PARENT_DATA_PATH.joinpath(animal_id)

        validate_path(animal_dir)

        # validate exp_type
        if exp_type is not None:
            if exp_type not in ["table", "pez"]:
                raise ValueError(f"{exp_type} is not currently supported. Please pass in a valid exp_type "
                                 f"such as 'table' or 'pez'.")

        if not os.path.exists(animal_dir):
            raise ValueError(f"animal_id path does not exist at: {animal_dir}")

        # session_id = None, get all sessions under that animal_id
        if session_id is None:
            # if session_id is None, trial_id must also be None
            if trial_id is not None:
                raise ValueError("If session_id is None, trial_id must also be None.")

            # get all sessions under an animal_id
            session_dirs = sorted(animal_dir.glob("*"))

            # no sessions found, raise
            if len(session_dirs) == 0:
                raise ValueError(f"no sessions found in this animal dir: {animal_dir}")

            for session_dir in session_dirs:
                # get trials
                trials = sorted(session_dir.glob("*"))

                # no trials found
                if len(trials) == 0:
                    raise ValueError(f"no trials found in this session: {session_dir}")

                # remove invalid files and add trial_id to list
                trial_ids = list()
                for t in trials:
                    if t.suffix not in ACCEPTED_CODECS:
                        trials.remove(t)
                    elif 'front' in t.stem:
                        trial_ids.append(t.stem.replace('_front', ''))
                    elif 'side' in t.stem:
                        trial_ids.append(t.stem.replace('_side', ''))

                trial_ids = sorted(set(trial_ids))

                # add trials to dataframe
                for trial in trial_ids:
                    # check if trial already in dataframe
                    if (
                        len(
                            self._df[
                                (self._df["animal_id"] == animal_id)
                                & (self._df["session_id"] == session_dir.stem)
                                & (self._df["trial_id"] == int(trial[-3:]))
                            ].index
                        )
                        > 0
                    ):
                        raise ValueError(
                            f"Item already exists with animal_id={animal_id}, "
                            f"session_id={session_dir.stem}, and trial_id={int(trial[-3:])}. "
                            f"Please remove the item before attempting to add again."
                        )

                    # get vid path regardless of codec
                    front_vid_path = sorted(session_dir.glob(f"*front*{trial[-3:]}*"))
                    if len(front_vid_path) == 0:
                        raise ValueError(f"front vid path does not exist for trial: {trial}")
                    side_vid_path = sorted(session_dir.glob(f"*side*{trial[-3:]}*"))
                    if len(side_vid_path) == 0:
                        raise ValueError(f"side vid path does not exist for trial: {trial}")

                    parent_path, relative_vid_path_front = split_path(front_vid_path[0])
                    parent_path, relative_vid_path_side = split_path(side_vid_path[0])

                    vid_paths = {
                        "front": relative_vid_path_front,
                        "side": relative_vid_path_side
                    }

                    # get output path
                    output_path = Path(f'{animal_id}/{session_dir.stem}/').joinpath(session_dir.stem).with_name(f'outputs.h5')

                    # add item to dataframe
                    s = pd.Series(
                        {
                            "animal_id": animal_id,
                            "session_id": session_dir.stem,
                            "trial_id": int(trial[-3:]),
                            "vid_paths": vid_paths,
                            "output_path": output_path,
                            "exp_type": exp_type,
                            "model_params": dict(),
                            "notes": None,
                        }
                    )

                    self._df.loc[self._df.index.size] = s

        else:  # session_id is provided in args
            session_dir = animal_dir.joinpath(session_id)

            # if trial_id is None, add all
            if trial_id is None:
                trials = sorted(session_dir.glob("*"))

                if len(trials) == 0:
                    raise ValueError(f"no trials found in this session: {session_dir}")

                # remove invalid files
                for t in trials:
                    if t.suffix not in ACCEPTED_CODECS:
                        trials.remove(t)

                # remove invalid files and add trial_id to list
                trial_ids = list()
                for t in trials:
                    if t.suffix not in ACCEPTED_CODECS:
                        trials.remove(t)
                    elif 'front' in t.stem:
                        trial_ids.append(t.stem.replace('_front', ''))
                    elif 'side' in t.stem:
                        trial_ids.append(t.stem.replace('_side', ''))

                trial_ids = sorted(set(trial_ids))

                for trial in trial_ids:
                    # check if trial already in dataframe
                    if (
                        len(
                            self._df[
                                (self._df["animal_id"] == animal_id)
                                & (self._df["session_id"] == session_id)
                                & (self._df["trial_id"] == int(trial[-3:]))
                            ].index
                        )
                        > 0
                    ):
                        raise ValueError(
                            f"Item already exists with animal_id={animal_id}, "
                            f"session_id={session_id}, and trial_id={int(trial[-3:])}. "
                            f"Please remove the item before attempting to add again."
                        )

                    # get vid path regardless of codec
                    front_vid_path = sorted(session_dir.glob(f"*front*{trial[-3:]}*"))
                    if len(front_vid_path) == 0:
                        raise ValueError(f"front vid path does not exist for trial: {trial}")
                    side_vid_path = sorted(session_dir.glob(f"*side*{trial[-3:]}*"))
                    if len(side_vid_path) == 0:
                        raise ValueError(f"side vid path does not exist for trial: {trial}")

                    parent_path, relative_vid_path_front = split_path(front_vid_path[0])
                    parent_path, relative_vid_path_side = split_path(side_vid_path[0])

                    vid_paths = {
                        "front": relative_vid_path_front,
                        "side": relative_vid_path_side
                    }

                    # get output path
                    output_path = Path(f'{animal_id}/{session_id}/').joinpath(session_id).with_name(f'outputs.h5')

                    # add item to dataframe
                    s = pd.Series(
                        {
                            "animal_id": animal_id,
                            "session_id": session_id,
                            "trial_id": int(trial[-3:]),
                            "vid_paths": vid_paths,
                            "output_path": output_path,
                            "exp_type": exp_type,
                            "model_params": dict(),
                            "notes": None,
                        }
                    )

                    self._df.loc[self._df.index.size] = s
            # trial id is not None, only adding one item
            else:
                # check if trial already in dataframe
                if (
                    len(
                        self._df[
                            (self._df["animal_id"] == animal_id)
                            & (self._df["session_id"] == session_id)
                            & (self._df["trial_id"] == trial_id)
                        ].index
                    )
                    > 0
                ):
                    raise ValueError(
                        f"Item already exists with animal_id={animal_id}, "
                        f"session_id={session_id}, and trial_id={trial_id} "
                        f"Please remove the item before attempting to add again."
                    )

                # get vid path regardless of codec
                front_vid_path = sorted(session_dir.glob(f"*front*{trial_id}*"))
                if len(front_vid_path) == 0:
                    raise ValueError(f"front vid path does not exist for trial: {trial_id}")
                side_vid_path = sorted(session_dir.glob(f"*side*{trial_id}*"))
                if len(side_vid_path) == 0:
                    raise ValueError(f"side vid path does not exist for trial: {trial_id}")

                parent_path, relative_vid_path_front = split_path(front_vid_path[0])
                parent_path, relative_vid_path_side = split_path(side_vid_path[0])

                vid_paths = {
                    "front": relative_vid_path_front,
                    "side": relative_vid_path_side
                }

                # get output path
                output_path = Path(f'{animal_id}/{session_id}/').joinpath(session_id).with_name(f'outputs.h5')

                # add item to dataframe
                s = pd.Series(
                    {
                        "animal_id": animal_id,
                        "session_id": session_id,
                        "trial_id": trial_id,
                        "vid_paths": vid_paths,
                        "output_path": output_path,
                        "exp_type": exp_type,
                        "model_params": dict(),
                        "notes": None,
                    }
                )

                self._df.loc[self._df.index.size] = s

        # save df to disk
        self._df.to_hdf(self._df.paths.get_df_path(), key="df")

    def remove_item(
        self,
        row_ix: int = None,
        animal_id: str = None,
        session_id: str = None,
        trial_id: str = None,
    ):
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
        warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

        # row_ix takes precedence over other args
        if row_ix is not None:
            index = [row_ix]
        elif session_id is None:
            warnings.warn(
                "No `session_id` provided, will remove all items for provided "
                f"animal_id: {animal_id}"
            )
            # remove all items for animal
            index = self._df.index[self._df["animal_id"] == animal_id].tolist()
        else:
            if trial_id is not None:
                # remove single item that matches animal_id/session_id/trial_id
                index = self._df.index[
                    (self._df["animal_id"] == animal_id)
                    & (self._df["session_id"] == session_id)
                    & (self._df["trial_id"] == trial_id)
                ].tolist()
            else:  # remove all item for given animal_id/session_id
                warnings.warn(
                    "No `trial_id` provided, will remove all items for provided "
                    f"animal_id: {animal_id} and session_id: {session_id}"
                )
                index = self._df.index[
                    (self._df["animal_id"] == animal_id)
                    & (self._df["session_id"] == session_id)
                ].tolist()
        # Drop selected index
        for ix in index:
            self._df.drop([ix], inplace=True)
            # Reset indices so there are no 'jumps'
        self._df.reset_index(drop=True, inplace=True)
        # Save new df to disc
        self._df.to_hdf(self._df.paths.get_df_path(), key="df")

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
        warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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
            self._df.to_hdf(path, key="df")
            os.remove(bak)
        except:
            shutil.copyfile(bak, path)
            raise IOError(f"Could not save dataframe to disk.")

    def _unsafe_save(self):
        path = self._df.paths.get_df_path()

        self._df.to_hdf(path, key="df")


@pd.api.extensions.register_series_accessor("behavior")
class BehaviorSeriesExtensions:
    """Pandas series extensions for inference."""

    def __init__(self, s: pd.Series):
        self._series = s

    def infer(
            self,
            mode: str = "fast",
            gpu_id: int = 0,
    ):
        """
        Run feature extractor inference and sequence inference together for a given ``mode``. This
        method will use all default model parameters. If you would like to specify other model weight
        paths for the flow generator, feature extractor, or sequence model than the pre-trained ones,
        please call feature extraction and sequence inference separately.

        See the User Guide for more details on model customization.

        Parameters
        ----------
        mode: str, default 'fast'
            One of ['slow', 'medium', 'fast']. Determines which pairing of models to use for the
            feature extractor and sequence model reconstruction. See table below for model details.

            +--------+-----------------+---------------+-----------------+
            | mode   | flow model      | feature model | sequence model  |
            +========+=================+===============+=================+
            | slow   | TinyMotionNet   | ResNet3D-34   | TGMJ            |
            +--------+-----------------+---------------+-----------------+
            | medium | MotionNet       | ResNet50      | TGMJ            |
            +--------+-----------------+---------------+-----------------+
            | fast   | TinyMotionNet3D | ResNet18      | TGMJ            |
            +--------+-----------------+---------------+-----------------+

        gpu_id: int, default 0
            Integer id of GPU to be used for inference. By default, assumes only one available GPU.
        """

        self._series.feature_extractor.infer(mode=mode, gpu_id=gpu_id)
        self._series.sequence.infer(mode=mode, gpu_id=gpu_id)

        print("Successfully ran feature extraction and sequence inference.")
