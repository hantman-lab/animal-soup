import os
from pathlib import Path
from typing import Union

import pandas as pd
import re as regex

CURRENT_DF_PATH: Path = None  # only one batch at a time
PARENT_DATA_PATH: Path = None

DATAFRAME_COLUMNS = ["animal_id", "session_id", "mat_file", "session_vids", "notes"]


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
        path = Path(path)

        if self.get_df_path().parent.joinpath(path).exists():
            return self.get_df_path().parent.joinpath(path)

        elif get_parent_raw_data_path() is not None:
            if get_parent_raw_data_path().joinpath(path).exists():
                return get_parent_raw_data_path().joinpath(path)

        raise FileNotFoundError(f"Could not resolve full path of:\n{path}")


@pd.api.extensions.register_dataframe_accessor("paths")
class PathsDataFrameExtension(_BasePathExtensions):
    pass


@pd.api.extensions.register_series_accessor("paths")
class PathsSeriesExtension(_BasePathExtensions):
    pass


def load_df(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the batch dataframe hdf5 file

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

    return df


def create_df(path: Union[str, Path], remove_existing: bool = False) -> pd.DataFrame:
    """
    Create a new behavior DataFrame

    Parameters
    ----------
    path: str or Path
        path to save the new batch DataFrame as a hdf5 file

    remove_existing: bool
        If ``True``, remove an existing batch DataFrame file if it exists at the given `path`, default ``False``

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
                f"Behavior file already exists at specified location: {path}"
            )

    if not Path(path).parent.is_dir():
        os.makedirs(Path(path).parent)

    df = pd.DataFrame(columns=DATAFRAME_COLUMNS)
    df.paths.set_df_path(path)

    df.to_hdf(path, key='df')

    return df

