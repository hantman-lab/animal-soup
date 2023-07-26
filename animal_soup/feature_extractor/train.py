import pytorch_lightning as pl
import torch.nn
from ..utils import get_gpu_transforms
import numpy as np
from .models import HiddenTwoStream
from .loss import *
from ..utils import L2, L2_SP
from typing import *
from ..utils import (
    PrintCallback,
    PlotLossCallback,
    CheckpointCallback,
    CheckStopCallback,
)
from pathlib import Path

DEFAULT_TRAINING_PARAMS = {
    "min_lr": 5e-07,
    "num_epochs": 20,
    "patience": 3,
    "reduction_factor": 0.1,
    "steps_per_epoch": 1000,
    "regularization": {"alpha": 1.0e-05, "beta": 0.001, "style": "l2_sp"},
    "label_smoothing": 0.05,
    "loss_gamma": 1.0
}


class HiddenTwoStreamLightningModule(pl.LightningModule):
    def __init__(
            self,
            hidden_two_stream: HiddenTwoStream,
            model_in: Union[str, Path],
            gpu_id: int,
            datasets: dict,
            classifier_name: str,
            initial_lr: float = 0.0001,
            batch_size: int = 32,
            augs: dict = None,
    ):
        """
        Class for training hidden two stream model for feature extractor using a lightning module.

        Parameters
        ----------
        hidden_two_stream: HiddenTwoStream
            Hidden two stream model for feature extraction.
        model_in: str or Path
            Location of model weights used to reload the model previously. Needed for L2_SP regularization.
        gpu_id: int
            GPU to be used for training.
        datasets: dict
            Dictionary of datasets created from available trials in the dataframe.
        classifier_name: str
            One of ['resnet18', 'resnet50', 'resnet34_3d'].
        initial_lr: float, default 0.0001
            Learning rate to begin training with/
        batch_size: int, default 32
            Batch size.
        augs: dict, default None
            Dictionary of augmentations, used to get gpu augs.
        """
        super().__init__()

        self.model = hidden_two_stream
        self.datasets = datasets
        self.batch_size = batch_size
        self.lr = initial_lr
        self.augs = augs
        self.gpu_id = gpu_id
        self.model_in = Path(model_in)

        self.final_activation = torch.nn.Sigmoid()

        if '3d' in classifier_name.lower():
            self.gpu_transforms = get_gpu_transforms(augs=self.augs, conv_mode="3d")["train"]
        else:
            self.gpu_transforms = get_gpu_transforms(augs=self.augs)["train"]

        # configure optimizer and criterion
        self.optimizer = None
        self.criterion = None
        self.configure_criterion()
        self.epoch_losses = list()

    def get_dataloader(self):
        """Returns a dataloader."""
        dataloader = torch.utils.data.DataLoader(
            dataset=self.datasets,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

        return dataloader

    def _validate_batch_size(self, batch: dict):
        """Validate a batch size to make sure it has the right shape."""
        if 'images' in batch.keys():
            if batch['images'].ndim != 5:
                batch['images'] = batch['images'].unsqueeze(0)
        if 'labels' in batch.keys():
            if self.final_activation == 'sigmoid' and batch['labels'].ndim == 1:
                batch['labels'] = batch['labels'].unsqueeze(0)
        return batch

    def apply_gpu_transforms(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the flow generator GPU transformations to a video."""
        with torch.no_grad():
            images = self.gpu_transforms(images).detach()
        return images

    def configure_optimizers(self):
        """Configure the optimizer to be used in training the feature extractor."""

        weight_decay = 0

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # mode will always be min because metrics are loss or SSIM
            factor=DEFAULT_TRAINING_PARAMS["reduction_factor"],
            patience=DEFAULT_TRAINING_PARAMS["patience"],
            verbose=True,
            min_lr=DEFAULT_TRAINING_PARAMS["min_lr"],
        )

        self.optimizer = optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "loss"},
        }

    def configure_criterion(self):
        """Configure loss function to be used in training the feature extractor"""
        pos_weight = self.datasets.pos_weight

        if type(pos_weight) == np.ndarray:
            pos_weight = torch.from_numpy(pos_weight)

        data_criterion = BinaryFocalLoss(pos_weight=pos_weight,
                                         gamma=DEFAULT_TRAINING_PARAMS["loss_gamma"],
                                         label_smoothing=DEFAULT_TRAINING_PARAMS["label_smoothing"])

        if DEFAULT_TRAINING_PARAMS["regularization"]["style"] == "l2":
            regularization_criterion = L2(
                model=self.model,
                alpha=DEFAULT_TRAINING_PARAMS["regularization"]["alpha"],
            )
        else:  # regularization criterion must be "l2_sp"
            regularization_criterion = L2_SP(
                model=self.model,
                path_to_pretrained_weights=self.model_in,
                alpha=DEFAULT_TRAINING_PARAMS["regularization"]["alpha"],
                beta=DEFAULT_TRAINING_PARAMS["regularization"]["beta"],
            )

        criterion = ClassificationLoss(data_criterion, regularization_criterion)

        self.criterion = criterion

    def forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through hidden two stream.

        Parameters
        ----------
        batch: dict
            Images for the batch

        Returns
        -------
        images: torch.Tensor
            Input images after gpu augmentations have been applied
        outputs: list
            List of fused spatial and flow features
        """
        batch = self._validate_batch_size(batch)

        images = batch['images']

        images = self.apply_gpu_transforms(images)

        outputs = self.model(images)

        return images, outputs

    def training_step(self, batch: dict) -> torch.Tensor:
        """
        Method for forward pass, loss calculation, backward pass, and parameter update.

        Parameters
        ----------
        batch : dict
            contains images and other information

        Returns
        -------
        loss : torch.Tensor
            mean loss for batch for Lightning's backward + update hooks
        """
        images, outputs = self.forward(batch)

        probabilities = self.final_activation(outputs)

        loss, loss_dict = self.criterion(outputs, batch['labels'], self.model)

        to_log = loss_dict
        to_log["loss"] = loss.detach()
        to_log["probabilities"] = probabilities.detach()

        self.epoch_losses.append(to_log["loss"].cpu())

        self.log("loss", to_log["loss"], prog_bar=True)

        return loss

    def train_dataloader(self):
        return self.get_dataloader()


def get_feature_trainer(
        gpu_id: int,
        model_out: Path,
        stop_method: str = "learning rate",
        profiler: str = None,
):
    """
    Returns a Pytorch Lightning trainer to be used in training the flow generator.

    Parameters
    ----------
    gpu_id: int
        Integer id of gpu to complete training on
    stop_method: str, default learning_rate
        Stop method for ending training, one of ["learning_rate", "num_epochs"]
    profiler: str, default None
        Can be a string (ex: "simple", "advanced") or a Pytorch Lightning Profiler instance. Gives metrics
        during training.

    Returns
    -------
    trainer: pl.Trainer
        A trainer to be used to manage training the feature extractor.
    """
    tensorboard_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=model_out, name="flow_gen_train_logs"
    )

    callbacks = list()
    callbacks.append(PrintCallback())
    callbacks.append(PlotLossCallback())
    callbacks.append(pl.callbacks.LearningRateMonitor())
    callbacks.append(CheckpointCallback(model_out=model_out))
    callbacks.append(CheckStopCallback(model_out=model_out, stop_method=stop_method))

    # tuning messes with the callbacks
    trainer = pl.Trainer(
        devices=[gpu_id],
        precision=32,
        limit_train_batches=DEFAULT_TRAINING_PARAMS["steps_per_epoch"],
        logger=tensorboard_logger,
        max_epochs=DEFAULT_TRAINING_PARAMS["num_epochs"],
        num_sanity_val_steps=0,
        callbacks=callbacks,
        profiler=profiler,
    )
    torch.cuda.empty_cache()

    return trainer
