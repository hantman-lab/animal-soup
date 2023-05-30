import os
from pathlib import Path
from typing import Union

import pandas as pd
import re as regex

CURRENT_DF_PATH: Path = None  # only one df at a time
PARENT_DATA_PATH: Path = None

DATAFRAME_COLUMNS = ["animal_id", "session_id", "mat_path", "deg_path", "session_vids", "notes"]


def validate_path(path: Union[str, Path]):
    if not regex.match("^[A-Za-z0-9@\/\\\:._-]*$", str(path)):
        raise ValueError(
            "Paths must only contain alphanumeric characters, "
            "hyphens ( - ), underscores ( _ ) or periods ( . )"
        )
    return path


def set_parent_raw_data_path(path: Union[Path, str]) -> Path:
    """
    Set the global `PARENT_DATA_PATH`

    Parameters
    ----------
    path: Path or str
        Full parent data path
    """
    global PARENT_DATA_PATH
    path = Path(validate_path(path))
    if not path.is_dir():
        raise NotADirectoryError(
            "The directory passed to `set_parent_raw_data_path()` does not exist.\n"
        )
    PARENT_DATA_PATH = path

    return PARENT_DATA_PATH


def get_parent_raw_data_path() -> Path:
    """
    Get the global `PARENT_DATA_PATH`

    Returns
    -------
    Path
        global `PARENT_DATA_PATH` as a Path object

    """
    global PARENT_DATA_PATH
    return PARENT_DATA_PATH


class _BasePathExtensions:
    def __init__(self, data: Union[pd.DataFrame, pd.Series]):
        self._data = data

    def set_df_path(self, path: Union[str, Path]):
        self._data.attrs["df_path"] = Path(path)

    def get_df_path(self) -> Path:
        """
        Get the full path to the current dataframe
        """
        if "df_path" in self._data.attrs.keys():
            if self._data.attrs["df_path"] is not None:
                return self._data.attrs["df_path"]
        else:
            raise ValueError("df path is not set")

    def resolve(self, path: Union[str, Path]) -> Path:
        """
        Resolve the full path of the passed ``path`` if possible, first tries
        "df_dir" then "raw_data_dir".

        Parameters
        ----------
        path: str or Path
            The relative path to resolve

        Returns
        -------
        Path
            Full path with the batch path or raw data path appended

        """
        path = Path(path)
        if self.get_df_path().parent.joinpath(path).exists():
            return self.get_df_path().parent.joinpath(path)

        # else check if in parent raw data dir
        elif get_parent_raw_data_path() is not None:
            if get_parent_raw_data_path().joinpath(path).exists():
                return get_parent_raw_data_path().joinpath(path)

        raise FileNotFoundError(f"Could not resolve full path of:\n{path}")

    def split(self, path: Union[str, Path]):
        """
           Split a full path into (batch_dir, relative_path) or (raw_data_dir, relative_path)

           Parameters
           ----------
           path: str or Path
               Full path to split with respect to batch_dir or raw_data_dir

           Returns
           -------
           Tuple[Path, Path]
               (<batch_dir> or <raw_data_dir>, <relative_path>)

           """
        path = Path(path)
        # check if input movie is within batch dir
        if self.get_df_path().parent in path.parents:
            return self.get_df_path().parent, path.relative_to(
                self.get_df_path().parent
            )

        # else check if in parent raw data dir
        elif get_parent_raw_data_path() is not None:
            if get_parent_raw_data_path() in path.parents:
                return get_parent_raw_data_path(), path.relative_to(
                    get_parent_raw_data_path()
                )

        raise NotADirectoryError(
            f"Could not split `path`:\n{path}"
            f"\nnot relative to either batch path:\n{self.get_df_path()}"
            f"\nor parent raw data path:\n{get_parent_raw_data_path()}"
        )


@pd.api.extensions.register_dataframe_accessor("paths")
class PathsDataFrameExtension(_BasePathExtensions):
    pass


@pd.api.extensions.register_series_accessor("paths")
class PathsSeriesExtension(_BasePathExtensions):
    pass


def load_df(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the behavior dataframe hdf5 file

    Parameters
    ----------
    path: str or Path

    Returns
    -------
    pd.DataFrame
        behavior dataframe loaded from the specified path

    """

    path = validate_path(path)

    df = pd.read_hdf(Path(path))
    df.paths.set_df_path(path)

    return df


def create_df(path: Union[str, Path], remove_existing: bool = False) -> pd.DataFrame:
    """
    Create a new behavior DataFrame

    Parameters
    ----------
    path: str or Path
        path to save the new behavior DataFrame as a hdf5 file, should be located under PARENT_DATA_PATH

    remove_existing: bool
        If ``True``, remove an existing behavior DataFrame file if it exists at the given `path`, default ``False``

    Returns
    -------
    pd.DataFrame
        New empty behavior DataFrame

    """
    path = validate_path(path)

    if Path(path).is_file():
        if remove_existing:
            os.remove(path)
        else:
            raise FileExistsError(
                f"Behavior dataframe file already exists at specified location: {path}"
            )

    if not Path(path).parent.is_dir():
        os.makedirs(Path(path).parent)

    df = pd.DataFrame(columns=DATAFRAME_COLUMNS)
    df.paths.set_df_path(path)

    df.to_hdf(path, key='df')

    return df
