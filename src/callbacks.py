import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.utils import to_numpy, visualize_prediction


class ImageLoggerCallback(pl.Callback):
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, log_batches: int = 2, threshold: float = 0.0):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_batches = log_batches
        self.threshold = threshold

    def make_predictions(self, pl_module, loader):
        images, pred_masks, true_masks = [], [], []

        for ii, batch in enumerate(loader):

            if ii >= self.log_batches:
                break

            x, y, label = batch

            with torch.no_grad():
                x = x.cuda()
                output = pl_module(x)
                output = output.cpu()

            image = torch.squeeze(x)
            pred_mask = (output >= self.threshold).int().reshape(-1, 512, 512)
            true_mask = y.reshape(-1, 512, 512)

            images.extend(list(to_numpy(image)))
            pred_masks.extend(list(to_numpy(pred_mask)))
            true_masks.extend(list(to_numpy(true_mask)))

        return images, pred_masks, true_masks

    @staticmethod
    def make_grid(images, pred_masks, true_masks):
        grid = []
        for image, pred, true in zip(images, pred_masks, true_masks):
            visualization = visualize_prediction(image, pred, true)

            grid.append(visualization)

        grid = np.concatenate(grid, axis=0)
        return grid

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        images, pred_masks, true_masks = self.make_predictions(pl_module, self.val_loader)
        grid = self.make_grid(images, pred_masks, true_masks)

        trainer.logger.experiment.log({
            "val/predictions": wandb.Image(grid, caption="Red are ground truth, Green are predictions"),
            "global_step": trainer.global_step
        })

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        images, pred_masks, true_masks = self.make_predictions(pl_module, self.train_loader)
        grid = self.make_grid(images, pred_masks, true_masks)

        trainer.logger.experiment.log({
            "train/predictions": wandb.Image(grid, caption="Red are ground truth, Green are predictions"),
            "global_step": trainer.global_step
        })
