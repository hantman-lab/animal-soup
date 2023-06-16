from .._behavior import BehaviorVizContainer
from .._ethogram_comparison import EthogramComparison
from .._ethogram import EthogramVizContainer
from .common import BehaviorDataFrameExtension
import pandas as pd

@pd.api.extensions.register_dataframe_accessor("viewer")
class BehaviorDataFrameVizExtension(BehaviorDataFrameExtension):
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

    def comparison_view(self,
                        start_index: int = 0):
        container = EthogramComparison(
            dataframe=self._df,
            start_index=start_index,
        )

        return container
