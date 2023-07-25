from ..utils import *
from ..data import VideoDataset
import pprint
from typing import *
from .flow_gen_extensions import _load_pretrained_flow_model

# map the mode of training to the appropriate model
TRAINING_OPTIONS = {
    "slow": "ResNet3D-34",
    "medium": "ResNet50",
    "fast": "ResNet18",
}

# class names
BEHAVIOR_NAMES = [
    "lift",
    "handopen",
    "grab",
    "sup",
    "atmouth",
    "chew"
]

# map the stop methods to default values
STOP_METHODS = {"learning_rate": 5e-7, "num_epochs": 15}

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
    "pad": None,
    "random_resize": False,
    "resize": (224, 224),
    "saturation": 0.1,
    "degrees": 10
}


@pd.api.extensions.register_dataframe_accessor("feature_extractor")
class FeatureExtractorDataframeExtension:
    """
    Pandas dataframe extensions for training and inference of the feature extractor.
    """

    def __init__(self, df):
        self._df = df

    def train(
            self,
            mode: str = "slow",
            batch_size: int = 32,
            gpu_id: int = 0,
            initial_lr: float = 0.0001,
            stop_method: str = "learning_rate",
            flow_model_in: Union[str, Path] = None,
            flow_window: int = 11,
            feature_model_in: Union[str, Path] = None,
            model_out: Union[str, Path] = None,
    ):
        """
        Train feature extractor model.

        Parameters
        ----------
        mode: str, default 'slow'
            Argument must be one of ["slow", "medium", "fast"]. Determines the model used for training the feature
            extractor.

            | mode   | model           |
            |--------|-----------------|
            | fast   | ResNet18        |
            | medium | ResNet50        |
            | slow   | ResNet3D-34     |

        batch_size: int, default 32
            Batch size.
        gpu_id: int, default 0
            Specify which gpu to use for training the model.
        initial_lr: float, default 0.0001
            Initial learning rate.
        stop_method: str, default learning_rate
            Method for stopping training. Argument must be one of ["learning_rate", "num_epochs"]

            | stop method   | description                                                                |
            |---------------|----------------------------------------------------------------------------|
            | learning_rate | Stop training when learning rate drops below a given threshold, means loss |
            |               | has stopped improving                                                      |
            | num_epochs    | Stop training after a given number of epochs                               |

        flow_model_in: str or Path, default None
            Location of checkpoint used for flow generator. If None, then will use default checkpoint of flow
            generator.
        flow_window: int, default 11
            Flow window size. Used to infer optic flow features to pass to the feature extractor.
        feature_model_in: str or Path, default None
            If you want to train the model using different model weights than the default. User can
            provide a location to a different model checkpoint. For example, if you had retrained the feature extractor
            previously and wanted to use those weights instead.
        model_out: str or Path, default None
            User provided location of where to store model output such as model checkpoint with updated weights,
            hdf5 file with model results/metrics, etc. Should be a directory. By default, the model output will get
            stored in the same directory as the dataframe.
        """
        # validate feature_model_in
        if feature_model_in is not None:
            feature_model_in = validate_checkpoint_path(feature_model_in)

        # validate flow_model_in
        if flow_model_in is not None:
            flow_model_in = validate_checkpoint_path(flow_model_in)

        # check if model_out is valid
        if model_out is not None:
            # validate path
            validate_path(model_out)
            # if model_out is not a directory, raise
            if not model_out.is_dir():
                raise ValueError(f"path to store model output should be a directory")
        else:
            df_path = self._df.paths.get_df_path()
            df_dir, relative = self._df.paths.split(df_path)
            os.makedirs(df_dir.joinpath("feature_extr_output"), exist_ok=True)
            model_out = df_dir.joinpath("feature_extr_output")
        if os.listdir(model_out):
            raise ValueError(f"directory to store model output should be empty")

        # validate only one experiment type being used in training
        if len(list(set(list(self._df["exp_type"])))) != 1:
            raise ValueError("Training can only be completed with experiments of same type. "
                             f"The current experiments in your dataframe are: {set(list(self._df['exp_type']))} "
                             "Take a subset of your dataframe to train with one kind of experiment.")
            # validate that an exp_type has been set
        if list(set(list(self._df["exp_type"])))[0] is None:
            raise ValueError("The experiment type for trials in your dataframe has not been set. Please"
                             "set the `exp_type` column in your dataframe before attempting training.")
        exp_type = list(self._df["exp_type"])[0]

        # check valid mode
        if mode not in TRAINING_OPTIONS.keys():
            raise ValueError(f"mode argument must be one of {TRAINING_OPTIONS.keys()}")

        # check gpu_id
        gpu_options = get_gpu_options()
        if gpu_id not in gpu_options.keys():
            raise ValueError(
                f"gpu_id: {gpu_id} not in {gpu_options}. " f"Please select a valid gpu."
            )

        # check batch_size
        if batch_size < MIN_BATCH_SIZE or batch_size > MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}"
            )

        # validate stop method
        if stop_method not in STOP_METHODS.keys():
            raise ValueError(
                f"stop_method argument must be one of {STOP_METHODS.keys()}"
            )

        # to train feature extractor must have ethograms in columns
        ethograms = list(self._df["ethograms"])
        for e in ethograms:
            if e is None:
                raise ValueError(
                    "In order to train the feature extractor you must have labels "
                    f"in the ethograms column. Row {ethograms.index(e)} does not have an ethogram. "
                    f"Either remove the row from the dataframe before attempting training or add "
                    f"labels for this trial."
                )
            if e.shape[0] != len(BEHAVIOR_NAMES):
                raise ValueError(
                    f"The ethogram in row {ethograms.index(e)} does not have the correct number of "
                    f"behaviors. Each ethogram should have {len(BEHAVIOR_NAMES)} rows. The current "
                    f"behaviors are: {BEHAVIOR_NAMES}"
                )

        # reload flow generator model
        flow_model, flow_model_in = _load_pretrained_flow_model(
            weight_path=flow_model_in, mode="slow", flow_window=flow_window, exp_type=exp_type
        )

        # create available dataset from items in df
        training_vids = list()
        parent_data_path = get_parent_raw_data_path()
        for ix, row in self._df.iterrows():
            training_vids.append(
                parent_data_path.joinpath(row["vid_path"])
            )

        # validate number of videos in training set
        if len(training_vids) < 3:
            raise ValueError(
                "You need at least 3 trials to train the feature extractor. Please "
                "add more trials to the current dataframe!"
            )

        # calculate norm augmentation values for given videos in dataframe
        print("Calculating normalization statistics based on trials in dataframe")
        normalization = get_normalization(training_vids)
        # update AUGS
        AUGS = DEFAULT_AUGS.copy()
        AUGS["normalization"] = normalization

        # set the convolution mode
        conv_mode = "2d"
        if mode == "slow":
            conv_mode = "3d"

        # generate torchvision.transforms object from augs
        transforms = get_cpu_transforms(AUGS)

        # create VideoDataset from available videos with augs
        datasets = VideoDataset(
            vid_paths=training_vids,
            transform=transforms,
            conv_mode=conv_mode,
            mean_by_channels=AUGS["normalization"]["mean"],
            frames_per_clip=flow_window,
            labels=ethograms
        )

        dataset_metadata = datasets.dataset_info

        # reload feature extractor model

        model_params = {
            "Model": TRAINING_OPTIONS[mode],
            "Mode": mode,
            "Parameters": {
                "initial_learning_rate": initial_lr,
                "batch_size": batch_size,
                "stop_method": stop_method,
                "flow_window": flow_window,
                "image_augmentations": AUGS,
            },
            "Flow Generator Weight Path": flow_model_in,
            "Feature Extractor Weight Path": feature_model_in,
            "Output Path": model_out,
        }

        print("Starting training")
        print("Model Parameters:")
        pprint.pprint(model_params)
        print("Data Info: ")
        pprint.pprint(dataset_metadata)

        if "model_params" not in self._df.columns:
            self._df.insert(
                loc=len(self._df.columns) - 1,
                column="model_params",
                value=[dict() for i in range(len(self._df.index))],
            )

        # add flow gen model params to df
        for ix in range(len(self._df.index)):
            self._df.loc[ix]["model_params"].update(
                {"feature_extr_train": f"{model_params}"}
            )
        # save df
        self._df.behavior.save_to_disk()

    def infer(
            self,
            mode: "fast",
            batch_size: int = 32,
            gpu_id: int = 0,
            initial_lr: float = 0.0001,
            stop_method: str = "learning_rate",
            flow_in: Union[str, Path] = None,
            model_in: Union[str, Path] = None,
            model_out: Union[str, Path] = None,
    ):
        """
        Inference using feature extractor.

        Parameters
        ----------
        mode
        batch_size
        gpu_id
        initial_lr
        stop_method
        flow_in
        model_in
        model_out

        Returns
        -------

        """
        pass


def _load_pretrained_feature_model():
    pass
