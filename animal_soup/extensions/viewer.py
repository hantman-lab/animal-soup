from .._behavior import BehaviorVizContainer
from .._ethogram import EthogramVizContainer
from .._ethogram_cleaner import EthogramCleaner
import pandas as pd
from pathlib import Path
from typing import Union

@pd.api.extensions.register_dataframe_accessor("viewer")
class BehaviorDataFrameVizExtension:
    def __init__(self, df):
        self._df = df

    def behavior_view(
            self,
            start_index: int = 0,
    ):
        container = BehaviorVizContainer(
            dataframe=self._df,
            start_index=start_index,
        )

        return container

    def ethogram_view(self,
                      start_index: int=0):
        container = EthogramVizContainer(
            dataframe=self._df,
            start_index=start_index
        )

        return container

@pd.api.extensions.register_dataframe_accessor("cleaner")
class EthogramCleanerExtension:
    def __init__(self, df):
        self._df = df

    def clean(self,
                start_index: int = 0,
                clean_df: Union[str, Path] = None):
        container = EthogramCleaner(
            dataframe=self._df,
            start_index=start_index,
            clean_df=clean_df
        )

        return container


