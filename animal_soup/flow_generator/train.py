import pytorch_lightning as pl
import torch
from typing import *
from ..utils import get_gpu_transforms
from .models import *
from .loss import *

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
    }
}


class FlowLightningModule(pl.LightningModule):
    def __init__(
            self,
            model: Union[TinyMotionNet3D, TinyMotionNet, MotionNet],
            datasets: dict,
            initial_lr: float = 0.0001,
            batch_size: int = 32,
            augs: dict = None
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
            Default learning rate to begin training with.
        batch_size: int, default 32
            Default batch_size
        augs: Dict
            Dictionary of augmentations, used to get gpu augs.
        """
        super().__init__()

        self.model = model
        self.datasets = datasets
        self.batch_size = batch_size
        self.lr = initial_lr

        # what to do about calculating and saving metrics

        if isinstance(self.model, TinyMotionNet3D):
            self.gpu_transforms = get_gpu_transforms(augs=augs, conv_mode='3d')
        else:
            self.gpu_transforms = get_gpu_transforms(augs=augs)

        self.optimizer = None
        self.reconstructor = None

    def get_dataloader(self):
        """Returns a dataloader."""
        dataloader = torch.utils.data.DataLoader(
            dataset=self.datasets,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=torch.cuda.is_available()
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

    def configure_optimizer(self):
        """Configure the optimizer to be used in training the flow generator."""

        weight_decay = 0  # if self.hparams.weight_decay is None else self.hparams.weight_decay

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=self.lr,
                                     weight_decay=weight_decay)
        self.optimizer = optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min", # mode will always be min because metrics are loss or SSIM
            factor=DEFAULT_TRAINING_PARAMS["reduction_factor"],
            patience=DEFAULT_TRAINING_PARAMS["patience"],
            verbose=True,
            min_lr=DEFAULT_TRAINING_PARAMS["min_lr"]
        )

        monitor_key = 'val/' + self.metrics.key_metric
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': monitor_key}

    def configure_criterion(self):
        """Configure the loss function to be used in training the flow generator."""
        if DEFAULT_TRAINING_PARAMS["regularization"]["style"] == "l2":
            regularization_criterion = L2(
                                        model=self.model,
                                        alpha=DEFAULT_TRAINING_PARAMS["regularization"]["alpha"])
        else: # regularization criterion must be "l2_sp"
            pass
        # regularization_criterion = L2_SP(model, pretrained_file, cfg.train.regularization.alpha,
        #                                  cfg.train.regularization.beta)
        # # criterion, loss func
        # criterion = MotionNetLoss(
        #     regularization_criterion,
        #     flow_sparsity=cfg.flow_generator.flow_sparsity,
        #     sparsity_weight=cfg.flow_generator.sparsity_weight,
        #     smooth_weight_multiplier=cfg.flow_generator.smooth_weight_multiplier,
        # )

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
        self.log_image_stats(images)

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
        pass
        # # forward pass. images are returned because the forward pass runs augmentations on the gpu as well
        images, outputs = self.forward(batch)
        # # actually reconstruct t0 using t1 and estimated optic flow
        # downsampled_t0, estimated_t0, flows_reshaped = self.reconstructor(images, outputs)
        # loss, loss_components = self.criterion(batch, downsampled_t0, estimated_t0, flows_reshaped, self.model)
        # self.visualize_batch(images, downsampled_t0, estimated_t0, flows_reshaped, split)
        #
        # to_log = loss_components
        # to_log['loss'] = loss.detach()
        #
        # self.metrics.buffer.append(split, to_log)
        # # need to use the native logger for lr scheduling, etc.
        # key_metric = self.metrics.key_metric
        # self.log(f'{split}_loss', loss)
        # if split == 'val':
        #     self.log(f'{split}_{key_metric}', loss_components[key_metric].mean())
        #
        # return loss

    def get_trainer(self):
        pass
