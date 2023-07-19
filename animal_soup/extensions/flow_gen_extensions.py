import os.path

from animal_soup.utils import *
from ..flow_generator import *
from ..data import VideoDataset

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
MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 512

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
    # currently using the normalization in a previous config
    # but some kind of calculation is being done to calculate these based on videos?
    # need to look into how this is calculated
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

    def train(self,
              mode: str = "slow",
              flow_window: int = 11,
              batch_size: int = 32,
              gpu_id: int = 0,
              initial_lr: float = 0.0001,
              stop_method: str = "learning_rate",
              model_in: Union[str, Path] = None,
              model_out: Union[str, Path] = None
              ):
        """
        Train flow generator model.

        Parameters
        ----------
        mode: str, default 'slow'
            Argument must be one of ["slow", "medium", "fast"]. Determines the model used for training the flow
            generator.

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

        model_in: str or Path, default None
            If you want to retrain the model using different model weights than the default. User can
            provide a location to a different model checkpoint. For example, if you had retrained the flow generator
            previously and wanted to use those weights instead.
        model_out: str or Path, default None
            User provided location of where to store model output such as model checkpoint with updated weights,
            hdf5 file with model results, etc. By default, the model output will get stored in the same directory
            as the dataframe.
        """
        # check valid model_in if not None
        if model_in is not None:
            # validate path
            validate_path(model_in)
            # check if path exists
            if not os.path.exists(model_in):
                raise ValueError(f'No checkpoint file exists at: {model_in}')
            # if path is str, convert
            if isinstance(model_in, str):
                model_in = Path(model_in)
            # check if model_in suffix in ["pt", "ckpt"]
            if model_in.suffix not in [".pt", ".ckpt"]:
                raise ValueError("PyTorch model checkpoints should end in '.pt' or '.ckpt'. "
                                 "Please make sure the file you are trying to use is a model checkpoint.")

        # check if model_out is valid
        if model_out is not None:
            # validate path
            validate_path(model_out)
            # make sure path exists
            if not os.path.exists(model_out):
                raise ValueError(f"path does not exist at: {model_out}")
            # if model_out is str, convert to path
            if isinstance(model_out, str):
                model_out = Path(model_out)

        # check valid mode
        if mode not in TRAINING_OPTIONS.keys():
            raise ValueError(f"mode argument must be one of {TRAINING_OPTIONS.keys()}")

        # check gpu_id
        gpu_options = get_gpu_options()
        if gpu_id not in gpu_options.keys():
            raise ValueError(f"gpu_id: {gpu_id} not in {gpu_options}. "
                             f"Please select a valid gpu.")

        # check batch_size
        if batch_size < MIN_BATCH_SIZE or batch_size > MAX_BATCH_SIZE:
            raise ValueError(f'batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}')

        # validate stop method
        if stop_method not in STOP_METHODS.keys():
            raise ValueError(f"stop_method argument must be one of {STOP_METHODS.keys()}")

        # reload weights from file, want to use pretrained weights
        model = self._load_pretrained_flow_model(
            weight_path=model_in,
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
        datasets = VideoDataset(
            vid_paths=training_vids,
            transform=transforms,
            conv_mode=conv_mode,
            mean_by_channels=DEFAULT_AUGS["normalization"]["mean"],
            frames_per_clip=flow_window
        )

        dataset_metadata = datasets.dataset_info

        # metrics
        key_metric = "SSIM"
        #metrics = OpticalFlow(rundir, key_metric, num_parameters)
        # stopper

        # lightening module
        lightning_module = FlowLightningModule(
            model=model,
            datasets=datasets,
            initial_lr=initial_lr,
            batch_size=batch_size,
            augs=DEFAULT_AUGS
        )

        # trainer

        # train
        # trainer.fit(lightning_module)

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

        if mode == "slow":
            model = TinyMotionNet3D(num_images=flow_window)
        elif mode == "medium":
            model = TinyMotionNet(num_images=flow_window)
        else:  # mode is fast
            model = MotionNet(num_images=flow_window)

        # using default weight path
        if weight_path is None:
            weight_path = Path('/home/clewis7/repos/animal-soup/pretrained_checkpoints/flow_gen.ckpt')

        if isinstance(weight_path, str):
            weight_path = Path(weight_path)

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
                raise ValueError(f'{k} not found in model dictionary')
            elif model_dict[k].size() != v.size():
                raise ValueError(f'{k} has different size: pretrained:{v.size()} model:{model_dict[k].size}')
            else:
                pretrained_dict[k] = v

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=True)

        print("Successfully loaded model from checkpoint!")

        return model
