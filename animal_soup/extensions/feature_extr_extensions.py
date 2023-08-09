import h5py

from ..utils import *
from ..data import VideoDataset
import pprint
from typing import *
from .flow_gen_extensions import _load_pretrained_flow_model
from ..feature_extractor import (get_cnn,
                                 remove_cnn_classifier_layer,
                                 Fusion, HiddenTwoStream,
                                 HiddenTwoStreamLightningModule,
                                 get_feature_trainer,
                                 predict_single_video)

# map the mode of training to the appropriate model
TRAINING_OPTIONS = {
    "slow": "ResNet3D-34",
    "medium": "ResNet50",
    "fast": "ResNet18",
}

# default augs for building flow and spatial classifier
DEFAULT_CLASSIFIER_AUGS = {
    "dropout_p": 0.25,
    "fusion": "average",
    "final_bn": False
}

# class names
BEHAVIOR_CLASSES = [
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
    Pandas dataframe extensions for training the feature extractor.
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
            flow_model_in: Union[str, Path] = None,
            flow_window: int = 11,
            feature_model_in: Union[str, Path] = None,
            model_out: Union[str, Path] = None,
    ):
        """
        Train feature extractor model.

        The flow generator used to reconstruct the feature extractor will be reconstructed based on the ``mode`` (
        default 'fast') by default unless a user-specified flow generator model checkpoint can be passed to
        ``flow_model_in``. If you specify your own model paths to the flow generator (perhaps you have re-trained
        it previously and want to use those model weights instead), in order to have the correct flow generator model
        reconstructed, the ``mode`` argument must match the flow generator model type you are trying to instantiate.
        For example, if the flow generator model checkpoint is for a TinyMotionNet model, then you should specify
        ``mode`` as 'slow`.

        Additionally, a user may also specify a ``feature_model_in`` which can be used to reconstruct the feature model
        being used for training. For example, you have already re-trained the feature extractor and want to use
        those model weights instead. However, if you are passing in both a ``flow_model_in`` and ``feature_model_in``,
        the specified ``mode`` must match the type of flow generator and feature extractor you are trying to
        instantiate.

        See the table below for details on the correct mode/model pairings.

        Parameters
        ----------
        mode: str, default 'slow'
            Argument must be one of ["slow", "medium", "fast"]. Determines the model used for training the feature
            extractor.

            +--------+-----------------+---------------+
            | mode   | flow model      | feature model |
            +========+=================+===============+
            | slow   | TinyMotionNet   | ResNet3D-34   |
            +--------+-----------------+---------------+
            | medium | MotionNet       | ResNet50      |
            +--------+-----------------+---------------+
            | fast   | TinyMotionNet3D | ResNet18      |
            +--------+-----------------+---------------+

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

        flow_model_in: str or Path, default None
            Location of checkpoint used for flow generator. If None, then will use default checkpoint of flow
            generator based on the mode.
        flow_window: int, default 11
            Flow window size. Used to infer optic flow features to pass to the feature extractor.
        feature_model_in: str or Path, default None
            If you want to train the model using different model weights than the default. User can
            provide a location to a different model checkpoint. For example, if you had retrained the feature extractor
            previously and wanted to use those weights instead. This should be a path to a hidden_two_stream model
            checkpoint that can be used to reconstruct the spatial and flow classifier. The model being
            reconstructed should align with the ``mode`` argument. See the table above for correct model/mode
            pairings.
        model_out: str or Path, default None
            User provided location of where to store model output such as model checkpoint with updated weights,
            hdf5 file with model results/metrics, etc. Should be a directory. By default, the model output will get
            stored in the same directory as the dataframe.
        """
        # validate feature_model_in
        if feature_model_in is not None:
            feature_model_in = validate_checkpoint_path(feature_model_in)

        # validate flow_model_in and flow_mode
        if flow_model_in is not None:
            flow_model_in = validate_checkpoint_path(flow_model_in)

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
            os.makedirs(df_dir.joinpath("feature_extr_output"), exist_ok=True)
            model_out = df_dir.joinpath("feature_extr_output")
        if os.listdir(model_out):
            raise ValueError(f"directory to store model output should be empty")

        # validate experiment type
        exp_type = validate_exp_type(self._df)

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

        # create available dataset from items in df
        training_vids = list(self._df["vid_paths"].values)

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

        # reload flow generator model
        flow_model, flow_model_in = _load_pretrained_flow_model(
            weight_path=flow_model_in, mode=mode, flow_window=flow_window, exp_type=exp_type
        )

        num_classes = len(BEHAVIOR_CLASSES) + 1  # account for background

        # build flow classifier
        if mode == "slow":
            in_channels = 2
        else:
            in_channels = (flow_window - 1) * 2
        flow_classifier, feature_model_in = _build_classifier(
            mode=mode,
            num_classes=num_classes,
            exp_type=exp_type,
            pos=datasets.num_pos,
            neg=datasets.num_neg,
            feature_model_in=feature_model_in,
            classifier_type="flow",
            final_bn=DEFAULT_CLASSIFIER_AUGS["final_bn"],
            in_channels=in_channels
        )

        # build spatial classifier model
        spatial_classifier, feature_model_in = _build_classifier(
            mode=mode,
            num_classes=num_classes,
            exp_type=exp_type,
            classifier_type="spatial",
            pos=datasets.num_pos,
            neg=datasets.num_neg,
            feature_model_in=feature_model_in,
            final_bn=DEFAULT_CLASSIFIER_AUGS["final_bn"],
            in_channels=3
        )

        # fuse spatial and flow classifiers into hidden two stream model
        spatial_classifier, flow_classifier, fused_model = _build_fusion(
            spatial_classifier=spatial_classifier,
            flow_classifier=flow_classifier,
            fusion_type=DEFAULT_CLASSIFIER_AUGS["fusion"],
            num_classes=num_classes
        )
        print("Successfully fused the flow classifier and spatial classifier models!")

        hidden_two_stream = HiddenTwoStream(
            flow_generator=flow_model,
            spatial_classifier=spatial_classifier,
            flow_classifier=flow_classifier,
            fusion=fused_model,
            classifier_name=TRAINING_OPTIONS[mode],
            num_images=flow_window
        )
        hidden_two_stream.set_mode("classifier")
        print("Successfully created hidden two stream model!")

        # lightning module
        lightning_module = HiddenTwoStreamLightningModule(
            hidden_two_stream=hidden_two_stream,
            datasets=datasets,
            initial_lr=initial_lr,
            batch_size=batch_size,
            augs=AUGS,
            gpu_id=gpu_id,
            classifier_name=TRAINING_OPTIONS[mode],
            model_in=feature_model_in
        )

        # trainer
        trainer = get_feature_trainer(
            gpu_id=gpu_id, model_out=model_out, stop_method=stop_method
        )

        model_params = {
            "Feature Model": TRAINING_OPTIONS[mode],
            "Flow Model": flow_model.__class__.__name__,
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

        # add feature extr model params to df
        for ix in range(len(self._df.index)):
            self._df.loc[ix]["model_params"].update(
                {"feature_extr_train": f"{model_params}"}
            )
        # save df
        self._df.behavior.save_to_disk()

        trainer.fit(lightning_module)


@pd.api.extensions.register_series_accessor("feature_extractor")
class FeatureExtractorSeriesExtensions:
    """Pandas series extensions for inference of the feature extractor."""

    def __init__(self, s: pd.Series):
        self._series = s

    def infer(
            self,
            mode: str = "fast",
            gpu_id: int = 0,
            feature_model_in: Union[str, Path] = None,
            flow_model_in: Union[str, Path] = None,
            flow_window: int = 11
    ):
        """
        Run feature extractor inference on a single trial.

        Parameters
        ----------
        mode: str, default 'fast'
            One of ["slow", "medium", "fast"]. Indicates what feature extractor to use for
            inference.

            +--------+-----------------+---------------+-----------------+
            | mode   | flow model      | feature model | inference speed |
            +========+=================+===============+=================+
            | slow   | TinyMotionNet   | ResNet3D-34   | ~13 fps         |
            +--------+-----------------+---------------+-----------------+
            | medium | MotionNet       | ResNet50      | ~80 fps         |
            +--------+-----------------+---------------+-----------------+
            | fast   | TinyMotionNet3D | ResNet18      | ~150 fps        |
            +--------+-----------------+---------------+-----------------+


        gpu_id: int, default 0
            Specify which gpu to use for training the model.
        flow_model_in: str or Path, default None
            Location of checkpoint used for flow generator. If None, then will use default checkpoint of flow
            generator based on the mode. Note: If using a non-default flow generator checkpoint, the flow
            generator instantiated must match the ``mode`` argument. For example, if the ``flow_model_in``
            checkpoint is to a TinyMotionNet model, then the ``mode`` argument MUST be 'fast'.
        flow_window: int, default 11
            Flow window size. Used to infer optic flow features to pass to the feature extractor.
        feature_model_in: str or Path, default None
            If you want to train the model using different model weights than the default. User can
            provide a location to a different model checkpoint. For example, if you had retrained the feature extractor
            previously and wanted to use those weights instead. This should be a path to a hidden_two_stream model
            checkpoint that can be used to reconstruct the spatial and flow classifier. Note: if also passing in
            a non-default checkpoint to reconstruct a re-trained flow generator, ``flow_model_in`` is not None, then
            the model checkpoints must be for the same ``mode``. See table above for mode/model pairings.

        """
        # validate feature_model_in
        if feature_model_in is not None:
            feature_model_in = validate_checkpoint_path(feature_model_in)

        # validate flow_model_in and flow_mode
        if flow_model_in is not None:
            flow_model_in = validate_checkpoint_path(flow_model_in)

        # check valid mode
        if mode not in TRAINING_OPTIONS.keys():
            raise ValueError(f"mode argument must be one of {TRAINING_OPTIONS.keys()}")

        # check gpu_id
        gpu_options = get_gpu_options()
        if gpu_id not in gpu_options.keys():
            raise ValueError(
                f"gpu_id: {gpu_id} not in {gpu_options}. " f"Please select a valid gpu."
            )

        # set experiment type
        exp_type = self._series["exp_type"]

        # reconstruct hidden two stream
        # reload flow generator model
        flow_model, flow_model_in = _load_pretrained_flow_model(
            weight_path=flow_model_in, mode=mode, flow_window=flow_window, exp_type=exp_type
        )

        num_classes = len(BEHAVIOR_CLASSES) + 1  # account for background

        # build flow classifier
        if mode == "slow":
            in_channels = 2
        else:
            in_channels = (flow_window - 1) * 2
        flow_classifier, feature_model_in = _build_classifier(
            mode=mode,
            num_classes=num_classes,
            exp_type=exp_type,
            pos=None,
            neg=None,
            feature_model_in=feature_model_in,
            classifier_type="flow",
            final_bn=DEFAULT_CLASSIFIER_AUGS["final_bn"],
            in_channels=in_channels
        )

        # build spatial classifier model
        spatial_classifier, feature_model_in = _build_classifier(
            mode=mode,
            num_classes=num_classes,
            exp_type=exp_type,
            classifier_type="spatial",
            pos=None,
            neg=None,
            feature_model_in=feature_model_in,
            final_bn=DEFAULT_CLASSIFIER_AUGS["final_bn"],
            in_channels=3
        )

        # fuse spatial and flow classifiers into hidden two stream model
        spatial_classifier, flow_classifier, fused_model = _build_fusion(
            spatial_classifier=spatial_classifier,
            flow_classifier=flow_classifier,
            fusion_type=DEFAULT_CLASSIFIER_AUGS["fusion"],
            num_classes=num_classes
        )
        print("Successfully fused the flow classifier and spatial classifier models!")

        hidden_two_stream = HiddenTwoStream(
            flow_generator=flow_model,
            spatial_classifier=spatial_classifier,
            flow_classifier=flow_classifier,
            fusion=fused_model,
            classifier_name=TRAINING_OPTIONS[mode],
            num_images=flow_window
        )
        hidden_two_stream.set_mode("inference")
        print("Successfully reloaded hidden two stream model!")

        # calculate norm augmentation values for given videos in dataframe
        print("Calculating vid normalization statistics")
        normalization = get_normalization([self._series["vid_paths"]])

        # update AUGS
        AUGS = DEFAULT_AUGS.copy()
        AUGS["normalization"] = normalization

        # set the convolution mode
        conv_mode = "2d"
        if mode == "slow":
            conv_mode = "3d"

        prediction_info = predict_single_video(
            vid_path=self._series["vid_paths"],
            hidden_two_stream=hidden_two_stream,
            mean_by_channels=AUGS["normalization"]["mean"],
            gpu_id=gpu_id,
            flow_window=flow_window,
            cpu_transform=get_cpu_transforms(AUGS),
            gpu_transform=get_gpu_inference_transforms(AUGS, conv_mode=conv_mode)
        )

        output_path = get_parent_raw_data_path().joinpath(self._series["output_path"])
        # write output to hdf5 file per session

        curr_trial = str(self._series["trial_id"])

        # if file does not exist, write mode
        if not output_path.is_file():
            with h5py.File(output_path, "w") as f:
                # create group for trial
                trial = f.create_group(curr_trial)
                # create feature group and add relevant datasets
                feature_group = trial.create_group("features")
                feature_group.create_dataset("spatial",
                                             data=prediction_info["spatial_features"].numpy())
                feature_group.create_dataset("flow",
                                             data=prediction_info["flow_features"].numpy())
                feature_group.create_dataset("logits",
                                             data=prediction_info["logits"].numpy())
                feature_group.create_dataset("probabilities",
                                             data=prediction_info["probabilities"].numpy())
        else:
            # file already exists, del group and recreate if exists otherwise just create
            with h5py.File(output_path, "r+") as f:

                if curr_trial in f.keys():
                    # delete and remake
                    del f[curr_trial]

                trial = f.create_group(curr_trial)

                # create feature group and add relevant datasets
                feature_group = trial.create_group("features")
                feature_group.create_dataset("spatial",
                                             data=prediction_info["spatial_features"].numpy())
                feature_group.create_dataset("flow",
                                             data=prediction_info["flow_features"].numpy())
                feature_group.create_dataset("logits",
                                             data=prediction_info["logits"].numpy())
                feature_group.create_dataset("probabilities",
                                             data=prediction_info["probabilities"].numpy())

        print("Successfully saved feature extraction output to disk!")


def _build_classifier(
        mode: str,
        num_classes: int,
        exp_type: str,
        classifier_type: str,
        pos: np.ndarray,
        neg: np.ndarray,
        feature_model_in: Path,
        in_channels: int,
        final_bn: bool = False,
):
    """
    Build a flow classifier model.

    Parameters
    ----------
    mode: str
        One of ["slow", "medium", "fast"]. Determines the ResNet architecture to use for the flow_classifier.
    num_classes: int
        Number of behaviors being classified.
    exp_type: str
        One of ["table", "pez"]. Reload from pretrained checkpoint if user did not provide weight path.
    pos: np.ndarray
        Number of positive examples in training set. Used for custom bias initialization in the final layer.
    neg: np.ndarray
        Number of negative examples in training set. Used for custom bias initialization in the final layer.
    final_bn: bool, default False
        Indicates whether there should be batch normalization at the end.
    classifier_type: str
        One of ["spatial", "flow"]. Type of classifier to instantiate.
    in_channels: int
        Number of input channels.

    Returns
    -------
    model
        One of [ResNet18, ResNet50, ResNet34-3D]. Depends on ``mode``. Either a flow classifier or a spatial classifier.
    flow_model_in: Path
        The path used for loading in the spatial and flow classifier weights.
    """
    # validate classifier type
    if classifier_type not in ["spatial", "flow"]:
        raise ValueError("classifier_type must be one of ['spatial', 'flow']")

    # construct CNN based on mode
    if mode == "slow":
        model = get_cnn(
            model_name=TRAINING_OPTIONS["slow"],
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_p=DEFAULT_CLASSIFIER_AUGS["dropout_p"],
            pos=pos,
            neg=neg,
            final_bn=final_bn
        )
    elif mode == "medium":
        model = get_cnn(
            model_name=TRAINING_OPTIONS["medium"],
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_p=DEFAULT_CLASSIFIER_AUGS["dropout_p"],
            pos=pos,
            neg=neg,
            final_bn=final_bn
        )
    else:  # mode must be fast
        model = get_cnn(
            model_name=TRAINING_OPTIONS["fast"],
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_p=DEFAULT_CLASSIFIER_AUGS["dropout_p"],
            pos=pos,
            neg=neg,
            final_bn=final_bn
        )

    # load weight into CNN from pretrained checkpoint
    if classifier_type == "spatial":
        key = "spatial_classifier."
    else:  # classifier must be "flow"
        key = "flow_classifier."

    if feature_model_in is None:
        feature_model_in = FEATURE_EXTRACTOR_MODEL_PATHS[exp_type][mode]

    pretrained_model_state = torch.load(feature_model_in)["state_dict"]

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

    # only update classifier model with keys pertaining to classifier
    params = {k.replace(key, ''): v for k, v in pretrained_model_state.items() if k.startswith(key)}

    pretrained_dict = {}
    for k, v in params.items():
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

    print(f"Successfully loaded {classifier_type} classifier from checkpoint!")

    return model, feature_model_in


def _build_fusion(
        spatial_classifier,
        flow_classifier,
        fusion_type: str,
        num_classes: int,
        fusion_weight: float = 1.5
):
    """
    Fuses the flow classifier and spatial classifier together based on a fusion type.

    Parameters
    ----------
    spatial_classifier
        Spatial classifier model with pre-trained weights already loaded.
        One of [resnet18, resnet50, resnet34-3D] depending on the mode.
    flow_classifier
        Flow classifier model with pre-trained weights already loaded.
        One of [resnet18, resnet50, resnet34-3D] depending on the mode.
    fusion_type: str
        One of ["average", "weighted_average", "concatenate"]. Indicates the way that fusing the classifier
        models together should occur.
    num_classes: int
        Number of behaviors being classified for plus background.
    fusion_weight: float, default 1.5
        How much to up-weight the flow fusion.

    Returns
    -------
    spatial_classifier, flow_classifier, fusion_model
    """
    if fusion_type in ["average", "weighted_average"]:
        num_spatial_features = None
        num_flow_features = None
    elif fusion_type == "concatenate":
        spatial_classifier, num_spatial_features = remove_cnn_classifier_layer(spatial_classifier)
        flow_classifier, num_flow_features = remove_cnn_classifier_layer(flow_classifier)
    else:
        raise ValueError("fusion_type must be one of ['average', 'weighted_average', 'concatenate']")

    fusion_model = Fusion(fusion_type=fusion_type,
                          num_spatial_features=num_spatial_features,
                          num_flow_features=num_flow_features,
                          num_classes=num_classes,
                          flow_fusion_weight=fusion_weight)

    return spatial_classifier, flow_classifier, fusion_model
