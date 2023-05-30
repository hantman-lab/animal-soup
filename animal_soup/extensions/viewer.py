from .._behavior import BehaviorVizContainer
from .common import BehaviorDataFrameExtension
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("behavior")
class BehaviorDataFrameVizExtension(BehaviorDataFrameExtension):
    def __init__(self, df):
        self._df = df

    def view(
            self,
            start_index: int = 0,
    ):
        container = BehaviorVizContainer(
            dataframe=self._df,
            start_index=start_index,
        )

        return container
