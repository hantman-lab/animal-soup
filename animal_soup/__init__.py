

from animal_soup.utils.df_utils import (
    create_df,
    load_df,
    set_parent_raw_data_path,
    get_parent_raw_data_path,
)

from .arrays import *
from .extensions import *
from .viewers import *

from pathlib import Path

with open(Path(__file__).parent.joinpath("VERSION"), "r") as f:
    __version__ = f.read().split("\n")[0]

__all__ = [
    "set_parent_raw_data_path",
    "get_parent_raw_data_path",
    "load_df",
    "create_df",
    "BehaviorVizContainer",
    "BehaviorDataFrameExtension",
    "EthogramVizContainer",
    "EthogramCleanerVizContainer",
    "EthogramComparisonVizContainer"
]

