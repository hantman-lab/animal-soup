import pandas as pd
from typing import *
import torch
from pathlib import Path

TRAINING_OPTIONS = {"slow": "TinyMotionNet3D",
                    "medium": "MotionNet",
                    "fast": "TinyMotionNet"}

@pd.api.extensions.register_dataframe_accessor("flow_generator")
class FlowGeneratorDataframeExtension:
    def __init__(self, df):
        self._df = df

    def train(self, mode: str = "slow", batch_size: int = 16, gpu_id: int = 0, lr: float = 0.0001):
        """Train flow generator model.

        Parameters
        ==========
        mode: str, default 'slow'
            Options must be one of ["slow", "medium", "fast"]. Determines the model used for training the flow generator.

            | mode   | model           | # params |
            |--------|-----------------|----------|
            | fast   | TinyMotionNet   | 1.9M     |
            | medium | MotionNet       | 45.8M    |
            | slow   | TinyMotionNet3D | 0.4M     |

        batch_size: int, default 16
            Batch size.
        gpu_id: int, default 0
            Specify which gpu to use for training the model.
        lr: float, default 0.0001
            Default learning rate.
        """
        # check valid mode
        if mode not in TRAINING_OPTIONS.keys():
            raise ValueError(f"mode argument must be one of {TRAINING_OPTIONS.keys()}")

        # check gpu_id
        gpu_options = self._get_gpu_options()
        if gpu_id not in gpu_options.keys():
            raise ValueError(f"gpu_id: {gpu_id} not in {gpu_options}. "
                             f"Please select a valid gpu.")

        # reload weights from file, want to use pretrained weights
        weights_file = Path('/data/caitlin/')

        # create dataloader from dataframe

        # later, have a check if gpu is available?

        print("Starting training")
        print(f"Training Mode: {mode} \n"
              f"Model: {TRAINING_OPTIONS[mode]} \n"
              f"Params: ")

        pass

    def _get_gpu_options(self) -> Dict[int, str]:
        """Returns a dictionary of {gpu_id: gpu name}"""
        gpu_options = dict()

        device_count = torch.cuda.get_device_count()

        for gpu_id in device_count:
            gpu_options[gpu_id] = torch.cuda.get_device_properties(gpu_id).name

        return gpu_options