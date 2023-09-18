import h5py

from ..utils import *
import os
from ..data import SequenceDataset
from .feature_extr_extensions import BEHAVIOR_CLASSES
import pprint
from ..sequence_model import *
from ..viewers.ethogram_utils import get_ethogram_from_disk, get_features_from_disk

# default parameters for sequence
DEFAULT_DATA_PARAMS = {
    "sequence_length": 180,
    "nonoverlapping": True,
}

# default model params
DEFAULT_MODEL_PARAMS = {
    "c_in": 1,
    "c_out": 8,
    "dropout_p": 0.5,
    "input_dropout": 0.5,
    "filter_length": 15,
    "final_bn": True,
    "n_filters": 8,
    "nonlinear_classification": True,
    "num_features": 128,
    "num_layers": 3,
    "soft_attn": True,
    "tgm_reduction": "max"
}

# map the stop methods to default values
STOP_METHODS = {"learning_rate": 5e-7, "num_epochs": 15}

# max number of epochs before training will stop if stop method default value is never reached
MAX_EPOCHS = 1000
MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 512

# default classifier thresholds
DEFAULT_THRESHOLDS = {
    "fast": np.array([0.5628578, 0.8090643, 0.5678824, 0.5327101, 0.4121191,
                      0.42216834, 0.63822716], dtype=np.float32),
    "medium": np.array([0.5226608, 0.71359646, 0.7387196, 0.71862113, 0.5327101,
                        0.4121191, 0.5327101], dtype=np.float32),
    "slow": np.array([0.44729146, 0.8090643, 0.81408894, 0.7889658, 0.6332025,
                      0.5528085, 0.72364575], dtype=np.float32)
}

MIN_BOUT_LENGTH = 3


@pd.api.extensions.register_dataframe_accessor("sequence")
class SequenceModelDataframeExtension:
    """
    Pandas dataframe extensions for training the sequence model.
    """

    def __init__(self, df):
        self._df = df

    def train(
            self,
            mode: str = "fast",
            batch_size: int = 32,
            gpu_id: int = 0,
            initial_lr: float = 0.0001,
            stop_method: str = "learning_rate",
            model_in: Union[str, Path] = None,
            model_out: Union[str, Path] = None,
            ethogram_mode: str = "inference"
    ):
        """
        Method for training the sequence model with trials in the current dataframe.

        Note: In order to train the sequence model. You will have to done feature extractor inference.
        This can be done using a for-loop and doing feature extractor inference on each row. For example:

        .. code-block:: python

            for ix, row in df.iterrows():
                row.feature_extractor.infer()

        Parameters
        ----------
        mode: str, default "fast"
            One of ["slow", "medium", "fast"]. Indicates which pre-trained TGMJ model to use for training.
            Argument will be ignored if using non-default pre-trained model checkpoint for training.
        batch_size: int, default 32
            Batch size.
        gpu_id: int, default 0
            Specify which gpu to use for training the model.
        initial_lr: float, default 0.0001
            Initial learning rate.
        stop_method: str, default learning_rate
            Method for stopping training. Argument must be one of ["learning_rate", "num_epochs"]

            +---------------+----------------------------------------------------------------------------------------------------+
            | stop method   | description                                                                                        |
            +===============+====================================================================================================+
            | learning_rate | Stop training when learning rate drops below a given threshold, means loss has stopped improving   |
            +---------------+----------------------------------------------------------------------------------------------------+
            | num_epochs    | Stop training after a given number of epochs                                                       |
            +---------------+----------------------------------------------------------------------------------------------------+

        model_in: str or Path, default None
            If you want to retrain the model using different model weights than the default. User can
            provide a location to a different model checkpoint. For example, if you had retrained the sequence model
            previously and wanted to use those weights instead.
        model_out: str or Path, default None
            User provided location of where to store model output such as model checkpoint with updated weights,
            hdf5 file with model results/metrics, etc. Should be a directory. By default, the model output will get
            stored in the same directory as the dataframe.
        ethogram_mode: str, default 'inference'
            One of ["inference", "ground"]. Specifies the location of where to look for ground truth ethograms
            for training. If "inference", will try to use ethograms stored from running inference. If "ground",
            will look for a column in the dataframe called "ethograms" to use for training.
        """
        # check valid model_in if not None
        if model_in is not None:
            model_in = validate_checkpoint_path(model_in)

        # check if model_out is valid
        if model_out is not None:
            # validate path
            model_out = validate_path(model_out)
            # if model_out is not a directory, raise
            if not model_out.is_dir():
                raise ValueError(f"path to store model output should be a directory")
        else:
            df_path = self._df.paths.get_df_path()
            df_dir, relative = self._df.paths.split(df_path)
            os.makedirs(df_dir.joinpath("sequence_output"), exist_ok=True)
            model_out = df_dir.joinpath("sequence_output")
        if os.listdir(model_out):
            raise ValueError(f"directory to store model output should be empty")

        # check valid mode
        if mode not in ["slow", "medium", "fast"]:
            raise ValueError(f"mode argument must be one of ['slow', 'medium', 'fast'.")

        # validate experiment type
        exp_type = validate_exp_type(self._df)

        # check gpu_id
        gpu_options = get_gpu_options()
        if gpu_id not in gpu_options.keys():
            raise ValueError(
                f"gpu_id: {gpu_id} not in {gpu_options}. " f"Please select a valid gpu."
            )

        # check batch_size
        if not MIN_BATCH_SIZE < batch_size < MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}"
            )

        # validate stop method
        if stop_method not in STOP_METHODS.keys():
            raise ValueError(
                f"stop_method argument must be one of {STOP_METHODS.keys()}"
            )

        # to train feature extractor must have ethograms in columns
        # validate mode
        if ethogram_mode not in ["ground", "inference"]:
            raise ValueError("ethogram_mode must be one of ['inference', 'ground']")
        # if ethogram mode is ground, look for ethograms in 'ethograms' column
        if ethogram_mode == "ground":
            if "ethograms" not in self._df.columns:
                raise ValueError(f"If ethogram_mode is 'ground', the ethograms to be used for training the sequence "
                                 f"model must be stored in the current dataframe in a column called 'ethograms'")
            ethograms = list(self._df["ethograms"])
            for e in ethograms:
                if e is None:
                    raise ValueError(
                        "In order to train the feature extractor you must have labels "
                        f"in the ethograms column. Row {ethograms.index(e)} does not have an ethogram. "
                        f"Either remove the row from the dataframe before attempting training or add "
                        f"labels for this trial."
                    )
                if e.shape[0] != len(BEHAVIOR_CLASSES):
                    raise ValueError(
                        f"The ethogram in row {ethograms.index(e)} does not have the correct number of "
                        f"behaviors. Each ethogram should have {len(BEHAVIOR_CLASSES)} rows. The current "
                        f"behaviors are: {BEHAVIOR_CLASSES}"
                    )
        else:  # ethogram_mode must be inference
            ethograms = list()
            for ix, row in self._df.iterrows():
                # get a saved ethogram from disk and make sure it is the right shape
                ground = get_ethogram_from_disk(row)
                if ground.shape[0] != len(BEHAVIOR_CLASSES):
                    raise ValueError(
                        f"The ethogram in row {ix} does not have the correct number of "
                        f"behaviors. Each ethogram should have {len(BEHAVIOR_CLASSES)} rows. The current "
                        f"behaviors are: {BEHAVIOR_CLASSES}"
                    )
                ethograms.append(ground)

        # to train sequence model, must have already run feature extraction inference
        extracted_features = list()
        for ix, row in self._df.iterrows():
            f = get_features_from_disk(row)
            if f is None:
                raise ValueError(f"feature extraction has not been run for the trial in row {ix}. Please "
                                 f"remove the trial from the dataframe or run feature extraction before trying to "
                                 f"train the sequence model.")
            extracted_features.append(f)

        # create available dataset from items in df
        training_vids = list(self._df["vid_paths"].values)

        # validate number of videos in training set
        if len(training_vids) < 3:
            raise ValueError(
                "You need at least 3 trials to train the feature extractor. Please "
                "add more trials to the current dataframe!"
            )

        # create sequence datasets for training
        datasets = SequenceDataset(
            vid_paths=training_vids,
            labels=ethograms,
            features=extracted_features,
            nonoverlapping=DEFAULT_DATA_PARAMS["nonoverlapping"],
            sequence_length=DEFAULT_DATA_PARAMS["sequence_length"]
        )

        dataset_metadata = datasets.dataset_info

        # create TGMJ model, no pre-trained weights needed
        num_classes = len(BEHAVIOR_CLASSES) + 1  # account for background

        # reload weights from file, want to use pretrained weights
        model, model_in = _load_pretrained_sequence_model(
            weight_path=model_in,
            exp_type=exp_type,
            num_classes=num_classes,
            num_neg=datasets.num_neg,
            num_pos=datasets.num_pos,
            mode=mode
        )

        # lightning module
        lightning_module = SequenceLightningModule(
            sequence_model=model,
            gpu_id=gpu_id,
            datasets=datasets,
            initial_lr=initial_lr,
            batch_size=batch_size,
            model_in=model_in
        )

        # trainer
        trainer = get_sequence_trainer(
            gpu_id=gpu_id,
            model_out=model_out,
            stop_method=stop_method
        )

        model_params = {
            "Sequence Model": 'TGMJ',
            "Parameters": {
                "initial_learning_rate": initial_lr,
                "batch_size": batch_size,
                "stop_method": stop_method,
            },
            "Weight Path": model_in,
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

        # add sequence model params to df
        for ix in range(len(self._df.index)):
            self._df.loc[ix]["model_params"].update(
                {"sequence_train": f"{model_params}"}
            )
        # save df
        self._df.behavior.save_to_disk()

        trainer.fit(lightning_module)


@pd.api.extensions.register_series_accessor("sequence")
class SequenceModelSeriesExtensions:
    """Pandas series extensions for inference of the sequence model."""

    def __init__(self, s: pd.Series):
        self._series = s

    def infer(
            self,
            mode: str = "fast",
            model_in: Union[str, Path] = None,
            gpu_id: int = 0
    ):
        """
        Note: Every sequence model is a TGMJ model. However, depending on the architecture used for training
        the flow generator and feature extractor model the sequence model can have highly variable results. You
        should specify the ``mode`` to match the ``mode`` you used for inference with the feature extractor.

        Parameters
        ----------
        mode: str, default 'fast'
            One of ["slow", "medium", "fast"]. Indicates what sequence model and thresholds to use.
            Should match the ``mode`` that was used when doing feature extraction.

            +--------+-----------------+---------------+-----------------+
            | mode   | flow model      | feature model | sequence model  |
            +========+=================+===============+=================+
            | slow   | TinyMotionNet   | ResNet3D-34   | TGMJ            |
            +--------+-----------------+---------------+-----------------+
            | medium | MotionNet       | ResNet50      | TGMJ            |
            +--------+-----------------+---------------+-----------------+
            | fast   | TinyMotionNet3D | ResNet18      | TGMJ            |
            +--------+-----------------+---------------+-----------------+

        model_in: str or Path, default None
            If you want to use your own model instead of the default you can provide a location to a different model
            checkpoint. For example, if you retrained the sequence model for a new behavioral task or setup and want to
            use those weights for inference instead of the default models.
        gpu_id: int, default 0
            Specify which gpu to use for training the model.
        """
        # check valid model_in if not None
        if model_in is not None:
            model_in = validate_checkpoint_path(model_in)

        # validate mode
        if mode not in ["slow", "medium", "fast"]:
            raise ValueError("'mode' must be one of ['slow', 'medium', 'fast']")

        # in order to run sequence inference, you need to have previously done feature extractor inference
        output_path = get_parent_raw_data_path().joinpath(self._series["output_path"])
        # checking for output path to retrieve features
        if not os.path.exists(output_path):
            print("No outputs from feature extraction found. Running feature extraction now with default "
                  "mode = 'fast'.")

            self._series.feature_extractor.infer(mode='fast', gpu_id=gpu_id)
        else:
            curr_trial = str(self._series["trial_id"])
            # check if trial in keys
            with h5py.File(output_path, "r+") as f:

                # not in keys, feature extraction has not been run
                if curr_trial not in f.keys():
                    print(f"Feature extraction has not been run for this trial yet. Running feature extraction now"
                          f" with mode = {mode}")
                    self._series.feature_extractor.infer(mode=mode, gpu_id=gpu_id)
                elif mode not in f[curr_trial].keys():
                    print(f"Feature extraction has not been run for this trial with mode = {mode} yet. Running feature "
                          f"extraction now with mode = {mode}")
                    self._series.feature_extractor.infer(mode=mode, gpu_id=gpu_id)

                # load in features to pass to sequence model
                features = dict()

                features["logits"] = f[curr_trial][mode]["features"]["logits"][:]
                features["probabilities"] = f[curr_trial][mode]["features"]["probabilities"][:]
                features["spatial_features"] = f[curr_trial][mode]["features"]["spatial"][:]
                features["flow_features"] = f[curr_trial][mode]["features"]["flow"][:]

        # set experiment type
        exp_type = self._series["exp_type"]

        # reload model
        num_classes = len(BEHAVIOR_CLASSES) + 1

        model, model_in = _load_pretrained_sequence_model(
            weight_path=model_in,
            exp_type=exp_type,
            num_classes=num_classes,
            num_pos=None,
            num_neg=None,
            mode=mode
        )

        prediction_info = predict_single_video(
            gpu_id=gpu_id,
            vid_path=self._series["vid_paths"],
            sequence_model=model,
            nonoverlapping=DEFAULT_DATA_PARAMS["nonoverlapping"],
            sequence_length=DEFAULT_DATA_PARAMS["sequence_length"],
            features=features
        )

        # post processing
        final_ethogram = min_bout_post_process(prediction_info, DEFAULT_THRESHOLDS[mode], MIN_BOUT_LENGTH)

        # update h5 file with final_ethogram and pred_info
        with h5py.File(output_path, "r+") as f:

            # if exists, delete and regenerate, else just create
            if "sequence" in f[curr_trial][mode].keys():
                del f[curr_trial][mode]["sequence"]

            sequence_group = f[curr_trial][mode].create_group("sequence")

            sequence_group.create_dataset("logits",
                                          data=prediction_info["logits"])
            sequence_group.create_dataset("probabilities",
                                          data=prediction_info["probabilities"])

            # if exists, delete and regenerate, else just create
            if "ethograms" in f[curr_trial][mode].keys():
                del f[curr_trial][mode]["ethograms"]

            ethogram_group = f[curr_trial][mode].create_group("ethograms")

            ethogram_group.create_dataset("ethogram",
                                          data=final_ethogram)

        print("Successfully saved sequence outputs to disk!")


def _load_pretrained_sequence_model(
        weight_path: Path,
        exp_type: str,
        num_classes: int,
        num_pos: np.ndarray,
        num_neg: np.ndarray,
        mode: str
):
    """

    Parameters
    ----------
    mode: str
        One of ["slow", "medium", "fast"]. Indicates which pre-trained sequence model checkpoint to use if
        weight_path is None.
    weight_path: str or Path object
        Location of model checkpoint to use for reloading weights. If `None`, will use default pre-trained model
        provided for given mode.
    exp_type: str
            One of ["table", "pez"]. Indicates which pre-trained model checkpoint to load from if weight_path is None.
    num_classes: int
        Number of classes for sequence model to output for.
    num_pos: np.ndarray
        Number of positive labels in the current dataframe trials, used to weight the model.
    num_neg: np.ndarray
        Number of negative labels in the current dataframe trials, used to weight the model.

    Returns
    -------
    model: TGMJ
        sequence model with reloaded weights from checkpoint
    weight_path: Path
        location of model weights used to reload model

    """
    model = TGMJ(
        classes=num_classes,
        n_filters=DEFAULT_MODEL_PARAMS["n_filters"],
        filter_length=DEFAULT_MODEL_PARAMS["filter_length"],
        input_dropout=DEFAULT_MODEL_PARAMS["input_dropout"],
        num_features=DEFAULT_MODEL_PARAMS["num_features"],
        num_layers=DEFAULT_MODEL_PARAMS["num_layers"],
        reduction=DEFAULT_MODEL_PARAMS["tgm_reduction"],
        c_in=DEFAULT_MODEL_PARAMS["c_in"],
        c_out=DEFAULT_MODEL_PARAMS["c_out"],
        soft=DEFAULT_MODEL_PARAMS["soft_attn"],
        pos=num_pos,
        neg=num_neg,
        use_fe_logits=False,
        nonlinear_classification=DEFAULT_MODEL_PARAMS["nonlinear_classification"],
        final_bn=DEFAULT_MODEL_PARAMS["final_bn"]
    )

    # using default weight path
    if weight_path is None:
        weight_path = SEQUENCE_MODEL_PATHS[exp_type][mode]

    # load model weights from checkpoint
    pretrained_model_state = torch.load(weight_path)["state_dict"]

    # remove "model." prepend if exists
    new_state_dict = OrderedDict()
    for k, v in pretrained_model_state.items():
        if k[:6] == "model.":
            name = k[6:]
        else:
            name = k
        new_state_dict[name] = v
    pretrained_model_state = new_state_dict

    # update model dict with model state from checkpoint
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in pretrained_model_state.items():
        if "criterion" in k:
            # we might have parameters from the loss function in our loaded weights. we don't want to reload these;
            # we will specify them for whatever we are currently training.
            continue
        if k not in model_dict:
            raise ValueError(f"{k} not found in model dictionary")
        elif model_dict[k].size() != v.size():
            raise ValueError(
                f"{k} has different size: pretrained:{v.size()} model:{model_dict[k].size()}"
            )
        else:
            pretrained_dict[k] = v

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)

    print("Successfully loaded sequence model from checkpoint!")

    return model, weight_path
