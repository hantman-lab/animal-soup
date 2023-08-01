from .extensions import *
from .utils import (
    load_df,
    create_df,
    set_parent_raw_data_path,
    get_parent_raw_data_path,
)
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
    "EthogramComparisonVizContainer",
    "FeatureExtractorSeriesExtensions",
    "FeatureExtractorDataframeExtension",
    "FlowGeneratorDataframeExtension",
    "SequenceModelSeriesExtensions",
    "SequenceModelDataframeExtension"
]
