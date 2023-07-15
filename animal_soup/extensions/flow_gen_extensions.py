import pandas as pd

@pd.api.extensions.register_dataframe_accessor("flow_generator")
class FlowGeneratorDataframeExtension:
    def __init__(self, df):
        self._df = df

    def train(self, mode: str = "slow", batch_size: int = 16, gpu_id: int = 0, lr: float = 0.00001):
        pass