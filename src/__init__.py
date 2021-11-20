from pathlib import Path

from easydict import EasyDict

conf = EasyDict(
    dataset_folder=Path("data/preprocessed/"),
    batch_size=2,
    val_size=0.1
)
