import pytorch_lightning as pl
import torch
from typing import *
from ..utils import get_gpu_transforms
from .models import *
from .loss import *
from .reconstructor import Reconstructor
from ..utils import L2, L2_SP
from pathlib import Path
from ..utils import PrintCallback, PlotLossCallback, CheckpointCallback, CheckStopCallback

DEFAULT_TRAINING_PARAMS = {
    "min_lr": 5e-07,
    "num_epochs": 10,
    "patience": 3,
    "reduction_factor": 0.1,
    "steps_per_epoch": 1000,
    "regularization": {
        "alpha": 1.0e-05,
        "beta": 0.001,
        "style": "l2_sp"
    },
    "smooth_weight_multiplier": 1.0,
    "sparsity_weight": 0.0,
    "flow_sparsity": False
}

MODEL_MAP = {
    TinyMotionNet3D: "TinyMotionNet3D",
    TinyMotionNet: "TinyMotionNet",
    MotionNet: "MotionNet"
}


class FlowLightningModule(pl.LightningModule):
    def __init__(
            self,
            model: Union[TinyMotionNet3D, TinyMotionNet, MotionNet],
            gpu_id: int,
            datasets: dict,
            initial_lr: float = 0.0001,
            batch_size: int = 32,
            augs: dict = None,
            model_in: Union[str, Path] = None
    ):
        """
        Class for training flow generator using lightning module.

        Parameters
        ----------
        model: torch.nn.Module
            Model for training flow generator. Will be one of [TinyMotionNet3D, TinyMotionNet, MotionNet].
        datasets: dict
            Dictionary of datasets created from available trials in dataframe.
        initial_lr: float, default 0.0001
            Learning rate to begin training with.
        batch_size: int, default 32
            Batch size.
        augs: Dict, default None
            Dictionary of augmentations, used to get gpu augs.
        gpu_id: int
            GPU to be used for training.
        """
        super().__init__()

        self.model = model
        self.datasets = datasets
        self.batch_size = batch_size
        self.lr = initial_lr
        self.augs = augs
        self.gpu_id = gpu_id

        if isinstance(self.model, TinyMotionNet3D):
            self.gpu_transforms = get_gpu_transforms(augs=self.augs, conv_mode='3d')['train']
        else:
            self.gpu_transforms = get_gpu_transforms(augs=self.augs)['train']

        if model_in is None:
            self.model_in = Path('/home/clewis7/repos/animal-soup/pretrained_models/flow_generator').joinpath(
                MODEL_MAP[type(self.model)]).with_suffix('.ckpt')
        else:  # no need to validate because model weights will have already been loaded
            self.model_in = model_in

        # configure optimizer and criterion
        self.optimizer = None
        self.criterion = None
        self.configure_criterion()
        self.epoch_losses = list()

        self.reconstructor = Reconstructor(gpu_id=gpu_id, augs=self.augs)

    def get_dataloader(self):
        """Returns a dataloader."""
        dataloader = torch.utils.data.DataLoader(
            dataset=self.datasets,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )

        return dataloader

    def _validate_batch_size(self, batch: dict):
        """Validate a batch size to make sure it has the right shape."""
        if "images" in batch.keys():
            if batch["images"].ndim != 5:
                batch["images"] = batch["images"].unsqueeze(0)
        return batch

    def apply_gpu_transforms(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the flow generator GPU transformations to a video."""
        with torch.no_grad():
            images = self.gpu_transforms(images).detach()
        return images

    def configure_optimizers(self):
        """Configure the optimizer to be used in training the flow generator."""

        weight_decay = 0  # if self.hparams.weight_decay is None else self.hparams.weight_decay

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=self.lr,
                                     weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # mode will always be min because metrics are loss or SSIM
            factor=DEFAULT_TRAINING_PARAMS["reduction_factor"],
            patience=DEFAULT_TRAINING_PARAMS["patience"],
            verbose=True,
            min_lr=DEFAULT_TRAINING_PARAMS["min_lr"]
        )

        self.optimizer = optimizer

        return {'optimizer': optimizer,
                'lr_scheduler': {
                                "scheduler": scheduler,
                                "monitor": "loss"
                        }
                }

    def configure_criterion(self):
        """Configure the loss function to be used in training the flow generator."""
        if DEFAULT_TRAINING_PARAMS["regularization"]["style"] == "l2":
            regularization_criterion = L2(
                model=self.model,
                alpha=DEFAULT_TRAINING_PARAMS["regularization"]["alpha"])
        else:  # regularization criterion must be "l2_sp"
            regularization_criterion = L2_SP(
                model=self.model,
                path_to_pretrained_weights=self.model_in,
                alpha=DEFAULT_TRAINING_PARAMS["regularization"]["alpha"],
                beta=DEFAULT_TRAINING_PARAMS["regularization"]["beta"])
        # # criterion, loss func
        criterion = MotionNetLoss(
            regularization_criterion,
            flow_sparsity=DEFAULT_TRAINING_PARAMS["flow_sparsity"],
            sparsity_weight=DEFAULT_TRAINING_PARAMS["sparsity_weight"],
            smooth_weight_multiplier=DEFAULT_TRAINING_PARAMS["smooth_weight_multiplier"],
        )

        self.criterion = criterion

    def forward(self, batch: dict) -> Tuple[torch.Tensor, List]:
        """
        Compute the optic flow through the forward pass.

        Parameters
        ----------
        batch: dict
            Images for the batch

        Returns
        -------
        images: torch.Tensor
            Input images after gpu augmentations have been applied
        outputs: list
            List of optic flows at multiple scales
        """
        batch = self._validate_batch_size(batch)

        images = batch["images"]

        images = self.apply_gpu_transforms(images=images)

        outputs = self.model(images)

        return images, outputs

    def common_step(self, batch: dict) -> torch.Tensor:
        """
        Method for doing forward pass, reconstructing images, and computing loss.

        Parameters
        ----------
        batch: dict
            Current batch of videos.

        Returns
        -------
        loss: torch.Tensor
            Mean loss of the input batch for the backward pass
        """
        # TODO: forward pass, reconstruct images and visualize with fpl in gridplot?, compute loss,
        #  print/update loss, plot loss, compute metrics, print metrics
        pass
        # # forward pass. images are returned because the forward pass runs augmentations on the gpu as well
        images, outputs = self.forward(batch)
        # actually reconstruct t0 using t1 and estimated optic flow
        downsampled_t0, estimated_t0, flows_reshaped = self.reconstructor(images, outputs)
        loss, loss_components = self.criterion(batch, downsampled_t0, estimated_t0, flows_reshaped, self.model)

        to_log = loss_components
        to_log['loss'] = loss.detach()

        self.epoch_losses.append(to_log["loss"].cpu())

        self.log("loss", to_log['loss'], prog_bar=True)

        return loss

    def training_step(self, batch: dict):
        return self.common_step(batch)

    def train_dataloader(self):
        return self.get_dataloader()


def get_flow_trainer(
        gpu_id: int,
        model_out: Path,
        stop_method: str = "learning_rate",
        profiler: str = None,
):
    """
    Returns a Pytorch Lightning trainer to be used in training the flow generator.

    Parameters
    ----------
    gpu_id: int
        Integer id of gpu to complete training on
    lightning_module: pl.LightningModule
        Pytorch Lightning module that has the model, datasets, criterion, etc.
        needed for training a model
    stop_method: str, default learning_rate
        Stop method for ending training, one of ["learning_rate", "num_epochs"]
    profiler: str, default None
        Can be a string (ex: "simple", "advanced") or a Pytorch Lightning Profiler instance. Gives metrics
        during training.

    Returns
    -------
    trainer: pl.Trainer
        A trainer to be used to manage training the flow generator.
    """

    tensorboard_logger = pl.loggers.tensorboard.TensorBoardLogger(
                                                        save_dir=model_out,
                                                        name="flow_gen_train"
                                                        )

    callbacks = list()
    callbacks.append(PrintCallback())
    callbacks.append(PlotLossCallback())
    callbacks.append(pl.callbacks.LearningRateMonitor())
    callbacks.append(CheckpointCallback(model_out=model_out))
    callbacks.append(CheckStopCallback(model_out=model_out, stop_method=stop_method))

    # tuning messes with the callbacks
    trainer = pl.Trainer(devices=[gpu_id],
                         precision=32,
                         limit_train_batches=DEFAULT_TRAINING_PARAMS["steps_per_epoch"],
                         logger=tensorboard_logger,
                         max_epochs=DEFAULT_TRAINING_PARAMS["num_epochs"],
                         num_sanity_val_steps=0,
                         callbacks=callbacks,
                         profiler=profiler)
    torch.cuda.empty_cache()

    return trainer
