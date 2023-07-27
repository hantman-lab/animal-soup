import pandas as pd


@pd.api.extensions.register_dataframe_accessor("sequence")
class SequenceModelDataframeExtension:
    """
    Pandas dataframe extensions for training the sequence model.
    """

    def __init__(self, df):
        self._df = df

    def train(self):
        pass


@pd.api.extensions.register_series_accessor("sequence")
class SequenceModelSeriesExtensions:
    """Pandas series extensions for inference of the sequence model."""

    def __init__(self, s: pd.Series):
        self._series = s

    def infer(self):
        pass

