import pytorch_lightning as pl
import torch.nn
import numpy as np
from ..feature_extractor import BinaryFocalLoss, ClassificationLoss
from ..utils import L2, L2_SP, PlotLossCallback, PrintCallback, CheckpointCallback, CheckStopCallback
from pathlib import Path
from typing import *

from .models import TGMJ

DEFAULT_TRAINING_PARAMS = {
    "min_lr": 5e-07,
    "num_epochs": 100,
    "patience": 5,
    "reduction_factor": 0.1,
    "steps_per_epoch": 1000,
    "regularization": {"alpha": 1.0e-05, "beta": 0.001, "style": "l2"},
    "label_smoothing": 0.05,
    "loss_gamma": 1.0
}


class SequenceLightningModule(pl.LightningModule):
    def __init__(
            self,
            sequence_model: TGMJ,
            model_in: Union[str, Path],
            datasets: dict,
            gpu_id: int = 0,
            initial_lr: float = 0.0001,
            batch_size: int = 32,
    ):
        """
        PyTorch Lightning module for sequence model training.

        Parameters
        ----------
        sequence_model: TGMJ
            Sequence model.
        datasets: dict
            Datasets for training based on the current trials in the dataframe.
        gpu_id: int, default 0
            Integer id of gpu to be used for training, assumes only 1 option.
        initial_lr: float, default 0.0001
            Initial learning rate.
        batch_size: int, default 32
            Batch size.
        model_in: Path or str
            Used in L2_SP regularization. Location of weights used to load sequence model.
        """
        super().__init__()

        self.model = sequence_model
        self.datasets = datasets
        self.batch_size = batch_size
        self.lr = initial_lr
        self.gpu_id = gpu_id
        self.model_in = model_in

        self.final_activation = torch.nn.Sigmoid()

        self.optimizer = None
        self.criterion = None
        self.configure_criterion()
        self.epoch_losses = list()

    def get_dataloader(self):
        """Returns dataloader."""
        dataloader = torch.utils.data.DataLoader(
            dataset=self.datasets,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )

        return dataloader

    def configure_optimizers(self):
        """Configure optimizer used in training the sequence model."""

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
        """Configure loss to be used in training the sequence model."""

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

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass of sequence model."""
        outputs = self.model(batch['features'])
        return outputs

    def training_step(self, batch: dict):
        """Completes forward pass, backward pass, updates model weights, and logs loss."""
        outputs = self.forward(batch)

        probabilities = self.final_activation(outputs)

        loss, loss_dict = self.criterion(outputs, batch['labels'], self.model)

        to_log = loss_dict
        to_log["loss"] = loss.detach()
        to_log["probabilities"] = probabilities.detach()

        self.epoch_losses.append(to_log["loss"].cpu())

        self.log("loss", to_log["loss"], prog_bar=True)

        return loss

    def train_dataloader(self):
        """Get train dataloader."""
        return self.get_dataloader()


def get_sequence_trainer(
        gpu_id: int,
        model_out: Path,
        stop_method: str = "learning rate",
        profiler: str = None,
):
    """
    Return PyTorch Lightning trainer instance for training the sequence model.

    Parameters
    ----------
    gpu_id: int
        Integer id of GPU to use for training.
    model_out: Path
        Location to store model output.
    stop_method: str, default 'learning_rate'
        Early stopping method used for training. One of ['num_epochs', 'learning_rate'].
    profiler: str, default None
        Can be a string (ex: "simple", "advanced") or a Pytorch Lightning Profiler instance. Gives metrics
        during training.

    Returns
    -------
    trainer: pl.Trainer
        A trainer to be used to manage training the feature extractor.
    """
    tensorboard_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=model_out, name="sequence_train_logs"
    )

    callbacks = list()
    callbacks.append(PrintCallback())
    callbacks.append(PlotLossCallback())
    callbacks.append(pl.callbacks.LearningRateMonitor())
    callbacks.append(CheckpointCallback(model_out=model_out, train_type="sequence"))
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
