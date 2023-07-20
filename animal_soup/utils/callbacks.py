"""Module for defining callbacks classes for Pytorch Lightning Trainer"""

from pytorch_lightning.callbacks import Callback


class PlotLossCallback(Callback):
    def __init__(self):
        super().__init__()

        pass


class CheckpointCallback(Callback):
    def __init__(self):
        super().__init__()

        pass


class CheckStopCallback(Callback):
    def __init__(
            self,
            stop_method: str
    ):
        """
        Callback method added to training to check at the end of every epoch if training should be stopped.

        Parameters
        ----------
        stop_method: str
            One of ["early", "learning_rate", "num_epochs"].
        """
        super().__init__()

        self.stop_method = stop_method

        pass


