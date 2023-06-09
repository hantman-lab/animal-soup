from ._behavior import BehaviorVizContainer, DECORD_CONTEXT
from ._ethogram import EthogramVizContainer
from ._ethogram_cleaner import EthogramCleaner
from ._ethogram_comparison import EthogramComparison
from .extensions import *
from .batch_utils import (
    create_df,
    load_df,
    set_parent_raw_data_path,
    get_parent_raw_data_path,
)

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
    "EthogramCleaner",
    "DECORD_CONTEXT",
    "EthogramComparison"
]

