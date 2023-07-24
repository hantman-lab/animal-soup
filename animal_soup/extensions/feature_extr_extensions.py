from ..utils import *
from ..data import VideoDataset
import pprint
from typing import *

# map the mode of training to the appropriate model
TRAINING_OPTIONS = {
    "slow": "ResNet3D-34",
    "medium": "ResNet50",
    "fast": "ResNet18",
}

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
        flow_in: Union[str, Path] = None,
        model_in: Union[str, Path] = None,
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
            Method for stopping training. Argument must be one of ["early", "learning_rate", "num_epochs"]

            | stop method   | description                                                                |
            |---------------|----------------------------------------------------------------------------|
            | learning_rate | Stop training when learning rate drops below a given threshold, means loss |
            |               | has stopped improving                                                      |
            | num_epochs    | Stop training after a given number of epochs                               |

        flow_in: str or Path, default None
            Location of checkpoint used for flow generator. If None, then will use default checkpoint of flow
            generator.
        model_in: str or Path, default None
            If you want to retrain the model using different model weights than the default. User can
            provide a location to a different model checkpoint. For example, if you had retrained the feature extractor
            previously and wanted to use those weights instead.
        model_out: str or Path, default None
            User provided location of where to store model output such as model checkpoint with updated weights,
            hdf5 file with model results/metrics, etc. Should be a directory. By default, the model output will get
            stored in the same directory as the dataframe.
        """

        pass

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
