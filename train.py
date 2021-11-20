from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src import conf
from src.callbacks import ImageLoggerCallback
from src.dataloader import CTDataLoader
from src.models import CTSemanticSegmentation
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def train():
    full_config = dict(conf, lr=0.001, train_aug=False, max_epochs=30, loss_fn='focal_loss', optim='Adam')
    logger = WandbLogger(name="lr=0.001,epoch=30, focal_loss, Adam, w/o aug, StepLR{step=20, gamma=0.1} ",
                         project="CTFinalModel", config=full_config, log_model='all')

    checkpoint = ModelCheckpoint(monitor="val/loss_epoch", save_top_k=1, mode='min', save_weights_only=True)
    model = CTSemanticSegmentation(full_config)
    dataloader = CTDataLoader(full_config['train_aug'])
    train_dataloader, val_dataloader = dataloader.train_dataloader(), dataloader.val_dataloader()

    early_stopping = EarlyStopping(monitor="val/loss_epoch", patience=5, mode="min")
    # image_callback = ImageLoggerCallback(train_dataloader, val_dataloader)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            early_stopping,
            # image_callback,
            checkpoint
        ],
        max_epochs=full_config['max_epochs'],
        accumulate_grad_batches=2,
        limit_train_batches=2,
        limit_val_batches=2,
        gpus=0
    )

    trainer.fit(model, dataloader.full_dataset(), val_dataloader)


train()
