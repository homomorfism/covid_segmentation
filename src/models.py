import sys
from functools import reduce
from typing import Optional, Any

import torch
from torch.optim.lr_scheduler import StepLR

from src.select_hparam import get_loss, get_optimizer
from src.utils import to_numpy
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_toolbelt.losses import WeightedLoss

class CTSemanticSegmentation(pl.LightningModule):

    def __init__(self, cfg: dict, threshold: float = 0.0):
        super(CTSemanticSegmentation, self).__init__()
        self.model = smp.Unet('mobilenet_v2', encoder_weights="imagenet", classes=1, activation=None,
                              encoder_depth=5,
                              decoder_channels=[256, 128, 64, 32, 16], in_channels=1)

        self.cfg = cfg
        self.loss_fn1 = WeightedLoss(smp.losses.DiceLoss(mode='binary'), cfg['dice_weight'])
        self.loss_fn1 = WeightedLoss(smp.losses.DiceLoss(mode='binary'), cfg['dice_weight'])
        self.eval_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=False)
        self.threshold = threshold

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.cfg, self.model.parameters(), lr=self.cfg['lr'])
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y, label = batch
        output = self(x)
        loss = self.loss_fn(output, y)

        self.log("train/loss_step", loss.item())

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, label = batch
        output = self(x)
        output = (output > self.threshold).int()
        loss = self.eval_loss_fn(output, y)

        self.log("val/loss_step", loss.item())

        return {"loss": loss}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> dict:
        x, label = batch
        output = self.model(x)
        output = (output > self.threshold).int().squeeze()
        output = list(to_numpy(output))

        return {"mask": output, "label": label}

    def training_epoch_end(self, outputs) -> None:
        avg_losses = torch.hstack([loss["loss"] for loss in outputs]).mean()
        self.log("train/loss_epoch", avg_losses)

    def validation_epoch_end(self, outputs) -> None:
        avg_losses = torch.hstack([loss["loss"] for loss in outputs]).mean()
        self.log("val/loss_epoch", avg_losses)
