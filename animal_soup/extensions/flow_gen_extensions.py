from animal_soup.utils import *
from ..flow_generator import *
from ..data import SingleVideoDataset, VideoDataset

# map the mode of training to the appropriate model
TRAINING_OPTIONS = {"slow": "TinyMotionNet3D",
                    "medium": "MotionNet",
                    "fast": "TinyMotionNet"}

# map the stop methods to default values
STOP_METHODS = {"early": 5,
                "learning_rate": 5e-7,
                "num_epochs": 15
                }

# max number of epochs before training will stop if stop method default value is never reached
MAX_EPOCHS = 1000

# default image augmentations
DEFAULT_AUGS = {
    "LR": 0.5,
    "UD": 0.0,
    "brightness": 0.25,
    "color_p": 0.5,
    "contrast": 0.1,
    "crop_size": None,
    "grayscale": 0.5,
    "hue": 0.1,
    "normalization": {
        "N": 149760000,
        "mean": [0.175888385730895,
                 0.175888385730895,
                 0.175888385730895],
        "std": [0.22989982574283255,
                0.22989982574283255,
                0.22989982574283255]
    },
    "pad": None,
    "random_resize": False,
    "resize": (224, 224),
    "saturation": 0.1
}


@pd.api.extensions.register_dataframe_accessor("flow_generator")
class FlowGeneratorDataframeExtension:
    """
    Pandas dataframe extensions for training the flow generator.
    """
    def __init__(self, df):
        self._df = df

    def train(self, mode: str = "slow",
              flow_window: int = 11,
              batch_size: int = 32,
              gpu_id: int = 0,
              initial_lr: float = 0.0001,
              stop_method: str = "learning_rate"
              ):
        """
        Train flow generator model.

        Parameters
        ----------
        mode: str, default 'slow'
            Argument must be one of ["slow", "medium", "fast"]. Determines the model used for training the flow generator.

            | mode   | model           |
            |--------|-----------------|
            | fast   | TinyMotionNet   |
            | medium | MotionNet       |
            | slow   | TinyMotionNet3D |

        flow_window: int, default 11
            Window size for computing optical flow features of a frame. Recommended to be the minimum number of frames
            any given behavior takes.
        batch_size: int, default 16
            Batch size.
        gpu_id: int, default 0
            Specify which gpu to use for training the model.
        initial_lr: float, default 0.0001
            Default learning rate.
        stop_method: str, default learning_rate
            Method for stopping training. Argument must be one of ["early", "learning_rate", "num_epochs"]

            | stop method   | description                                                                |
            |---------------|----------------------------------------------------------------------------|
            | early         | Stop training if there is no improvement after a certain number of events  |
            | learning_rate | Stop training when learning rate drops below a given threshold, means loss |
            |               | has stopped improving                                                      |
            | num_epochs    | Stop training after a given number of epochs                               |

        """
        # check valid mode
        if mode not in TRAINING_OPTIONS.keys():
            raise ValueError(f"mode argument must be one of {TRAINING_OPTIONS.keys()}")

        # check gpu_id
        gpu_options = get_gpu_options()
        if gpu_id not in gpu_options.keys():
            raise ValueError(f"gpu_id: {gpu_id} not in {gpu_options}. "
                             f"Please select a valid gpu.")

        # validate stop method
        if stop_method not in STOP_METHODS.keys():
            raise ValueError(f"stop_method argument must be one of {STOP_METHODS.keys()}")

        # reload weights from file, want to use pretrained weights
        model = self._load_pretrained_model(weight_path=Path('/home/clewis7/repos/animal-soup/pretrained_checkpoints/flow_gen.ckpt'),
                                            mode="slow",
                                            flow_window=flow_window)

        # create available dataset from items in df
        training_vids = list()
        parent_data_path = get_parent_raw_data_path()
        for ix, row in self._df.iterrows():
            # should there be a vid_path column in dataframe, would simplify a lot of code in a lot of places
            # would alleviate codec issue
            # for now assuming codec is AVI, but in future will need to detect or update
            training_vids.append(parent_data_path.joinpath(row["animal_id"],
                                                           row["session_id"],
                                                           row["trial_id"]).with_suffix('.avi'))

        # validate number of videos in training set
        if len(training_vids) < 3:
            raise ValueError("You need at least 3 trials to train the flow generator. Please "
                             "add more trials to the current dataframe!")

        # set the convolution mode
        conv_mode = "2d"
        if mode == "slow":
            conv_mode = "3d"

        # generate torchvision.transforms object from augs
        transforms = get_cpu_transforms(DEFAULT_AUGS)

        # create VideoDataset from available videos with augs
        dataset = VideoDataset(
                            vid_paths=training_vids,
                            transform=transforms,
                            conv_mode=conv_mode,
                            mean_by_channels=DEFAULT_AUGS["normalization"]["mean"],
                            frames_per_clip=flow_window
                            )

        dataset_metadata = dataset.metadata

        dataloader = torch.utils.data.Dataloader(
                                            dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True
                                            )

        # metrics

        # stopper

        # lightening module

        # trainer

        # train





        # in notes column, add flow_gen_train params for model
        # or should store all in output file
        # flow_gen output should also get stored in hdf5 file in same place as df path
        # at end of training should also store new model checkpoint?


        print("Starting training")
        print(f"Training Mode: {mode} \n"
              f"Model: {TRAINING_OPTIONS[mode]} \n"
              f"Params: "
              f"Data Info: ")

        return model

    def _load_pretrained_flow_model(self,
                               weight_path: Union[str, Path],
                               mode: str,
                               flow_window: int) -> Union[TinyMotionNet3D]:
        """Returns a model with the pretrained weights."""

        # check valid mode
        if mode not in TRAINING_OPTIONS.keys():
            raise ValueError(f"mode argument must be one of {TRAINING_OPTIONS.keys()}")

        if mode == "slow":
            model = TinyMotionNet3D(num_images=flow_window)
        elif mode == "medium":
            model = TinyMotionNet(num_images=flow_window)
        else: # mode is fast
            model = MotionNet(num_images=flow_window)

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







