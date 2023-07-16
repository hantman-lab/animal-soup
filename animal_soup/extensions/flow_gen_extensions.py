import pandas as pd
from typing import *
import torch
from pathlib import Path
from ..df_utils import validate_path, get_parent_raw_data_path
from ..flow_generator import *

TRAINING_OPTIONS = {"slow": "TinyMotionNet3D",
                    "medium": "MotionNet",
                    "fast": "TinyMotionNet"}

STOP_METHODS = {"early": 5,
                "learning_rate": 5e-7,
                "num_epochs": 1000
                }


@pd.api.extensions.register_dataframe_accessor("flow_generator")
class FlowGeneratorDataframeExtension:
    def __init__(self, df):
        self._df = df

    def train(self, mode: str = "slow",
              batch_size: int = 16,
              gpu_id: int = 0,
              lr: float = 0.0001,
              stop_method: Dict[str, Union[int, float]] = {"learning_rate": STOP_METHODS["learning_rate"]}
             ):
        """Train flow generator model.

        Parameters
        ==========
        mode: str, default 'slow'
            Argument must be one of ["slow", "medium", "fast"]. Determines the model used for training the flow generator.

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
        stop_method: Dict[str, Union[int, float]], default {num_epochs: 10}
            Method for stopping training. Argument must be one of ["early", "learning_rate", "num_epochs"]

            | stop method   | description                                                                |
            |---------------|----------------------------------------------------------------------------|
            | early         | Stop training if there is no improvement after a certain number of events  |
            | learning_rate | Stop training when learning rate drops below a given threshold             |
            | num_epochs    | Stop training after a given number of epochs                               |

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
        model = self._load_pretrained_model(weight_path=Path('/home/clewis7/repos/animal-soup/pretrained_checkpoints/flow_gen.ckpt'), mode="slow")

        # create available dataset from items in df
        training_vids = list()
        parent_data_path = get_parent_raw_data_path()
        for ix, row in self._df.iterrows():
            # for now assuming codec is AVI, but in future will need to detect or update
            training_vids.append(parent_data_path.joinpath(row["animal_id"],
                                                           row["session_id"],
                                                           row["trial_id"]).with_suffix('.avi'))


        # parse stop method
        if "stop_method" not in STOP_METHODS.keys():
            raise ValueError(f"stop_method: {stop_method} must be one of {STOP_METHODS}")
        # if stop method == "learning rate", should be type float
        if stop_method.keys()[0] == "learning_rate" and not isinstance(stop_method["learning_rate"], float):
            raise ValueError("if stop_method is 'learning_rate', must be of type float")
        # else should be type int
        elif not isinstance(stop_method.values()[0], int):
            raise ValueError("if 'stop_method' is either 'early' or 'num_epochs', must be type int")

        # in notes column, add flow_gen_train params for model
        # or should store all in output file
        # flow_gen output should also get stored in hdf5 file in same place as df path
        # at end of training should also store new model checkpoint?


        print("Starting training")
        print(f"Training Mode: {mode} \n"
              f"Model: {TRAINING_OPTIONS[mode]} \n"
              f"Params: ")

        return model

    def _get_gpu_options(self) -> Dict[int, str]:
        """Returns a dictionary of {gpu_id: gpu name}"""
        gpu_options = dict()

        device_count = torch.cuda.device_count()

        for gpu_id in range(device_count):
            gpu_options[gpu_id] = torch.cuda.get_device_properties(gpu_id).name

        return gpu_options

    def _load_pretrained_model(self, weight_path: Union[str, Path], mode: str) -> Union[TinyMotionNet3D]:
        """Returns a model with the pretrained weights."""

        # check valid mode
        if mode not in TRAINING_OPTIONS.keys():
            raise ValueError(f"mode argument must be one of {TRAINING_OPTIONS.keys()}")

        if mode == "slow":
            model = TinyMotionNet3D()
        elif mode == "medium":
            pass
        else: # mode is fast
            pass

        if isinstance(weight_path, str):
            weight_path = Path(weight_path)

        # validate weight path
        validate_path(weight_path)

        # load model weights from checkpoint
        pretrained_model_state = torch.load(weight_path)["state_dict"]

        # remove "model." preprend if exists
        new_state_dict = OrderedDict()
        for k, v in pretrained_model_state.items():
            if k[:6] == 'model.':
                name = k[6:]
            else:
                name = k
            new_state_dict[name] = v
        pretrained_model_state = new_state_dict

        # update model dict with model state from checkpoint
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in pretrained_model_state.items():
            if 'criterion' in k:
                # we might have parameters from the loss function in our loaded weights. we don't want to reload these;
                # we will specify them for whatever we are currently training.
                continue
            if k not in model_dict:
                print('{} not found in model dictionary'.format(k))
            else:
                if model_dict[k].size() != v.size():
                    print('{} has different size: pretrained:{} model:{}'.format(k, v.size(), model_dict[k].size()))
                else:
                    print('Successfully loaded: {}'.format(k))
                    pretrained_dict[k] = v

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=True)

        return model






