from sklearn.model_selection import train_test_split
from torch import Generator
from torch.utils.data import random_split, DataLoader, SubsetRandomSampler, Subset, ChainDataset

from src import conf
from src.augmentations import train_transform, valid_transform
from src.dataset import CTDataset
import pytorch_lightning as pl
from src.utils import extract_image_names, get_indices_of_frame_names


class CTDataLoader(pl.LightningDataModule):
    def __init__(self, train_aug: bool):
        super(CTDataLoader, self).__init__()

        dataset = CTDataset(train=True, transform=train_transform if train_aug else valid_transform)
        self.test_dataset = CTDataset(train=False, transform=valid_transform)

        train_names = extract_image_names(dataset.frame_names)
        train_names, val_names = train_test_split(train_names, test_size=0.2)

        train_indices = get_indices_of_frame_names(dataset.frame_names, train_names)
        self.train_dataset = Subset(dataset, train_indices)

        val_indices = get_indices_of_frame_names(dataset.frame_names, val_names)
        self.val_dataset = Subset(dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=conf.batch_size,
                          num_workers=2,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=conf.batch_size,
                          shuffle=False,
                          num_workers=2)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=conf.batch_size,
                          shuffle=False,
                          num_workers=2)

    def full_dataset(self):
        return DataLoader(dataset=ChainDataset([self.train_dataset, self.val_dataset]),
                          batch_size=conf.batch_size,
                          shuffle=True,
                          num_workers=2)
