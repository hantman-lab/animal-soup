"""Module for defining callbacks classes for Pytorch Lightning Trainer"""

import pytorch_lightning as pl
from matplotlib import pyplot as plt
import numpy as np
import torch
from pathlib import Path

DEFAULT_STOP_METHODS = {"learning_rate": 5e-7,
                        "num_epochs": 15
                        }


class PrintCallback(pl.Callback):
    """Callback to print the epoch loss after an epoch ends."""
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        metrics = trainer.callback_metrics
        print(f"Epoch: {trainer.current_epoch} \n"
              f"Train Loss: {metrics['loss']:.4f}")


class PlotLossCallback(pl.Callback):
    def __init__(self):
        super().__init__()

        self.total_losses = list()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """After each training epoch, plot the batch losses."""

        plt.plot(np.array(pl_module.epoch_losses))
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"Epoch {trainer.current_epoch}")
        plt.show()

        self.total_losses.append(np.array(pl_module.epoch_losses))

        pl_module.epoch_losses.clear()

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """At the end of training, plot the average epoch loss."""
        average_loss = list()
        for epoch in self.total_losses:
            average_loss.append(epoch.mean())

        plt.plot(np.array(average_loss))
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title(f"Average Epoch Loss")
        plt.show()


class CheckpointCallback(pl.Callback):
    def __init__(self,
                 model_out: Path):
        """
        Callback to save model after training has finished.

        Parameters
        ----------
        model_out: Path
            location of where to store model checkpoint after training has finished

        """
        super().__init__()

        self.model_out = model_out

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save model after training finishes at specified location."""
        epoch = trainer.current_epoch
        state_dict = pl_module.model.state_dict()

        save_path = self.model_out.joinpath('flow_generator_train.ckpt')

        torch.save({'epoch': epoch, 'state_dict': state_dict}, save_path)


class CheckStopCallback(pl.Callback):
    def __init__(
            self,
            stop_method: str,
            model_out: Path
    ):
        """
        Callback method added to training to check at the end of every epoch if training should be stopped.

        Parameters
        ----------
        stop_method: str
            One of ["learning_rate", "num_epochs"].
        model_out: Path
            Location of where to put model checkpoint if training ends early.
        """
        super().__init__()

        self.stop_method = stop_method
        self.model_out = model_out

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Parse current values and compare to default stopping methods to see if training should end."""
        if self.stop_method == "learning_rate":
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            if current_lr <= DEFAULT_STOP_METHODS["learning_rate"]:
                print("Stopping criterion reached! Stopping training.")
                trainer.should_stop = True
        else: # stop method is num epochs
            if trainer.current_epoch == DEFAULT_STOP_METHODS["num_epochs"]:
                print("Stopping criterion reached! Stopping training.")
                trainer.should_stop = True

        epoch = trainer.current_epoch
        state_dict = pl_module.model.state_dict()

        save_path = self.model_out.joinpath('flow_generator_train.ckpt')

        torch.save({'epoch': epoch, 'state_dict': state_dict}, save_path)


