from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import Dataset
import albumentations as A

from src import conf


class CTDataset(Dataset):
    """
    Returns
        - train - images with shape (1, 512, 512), labels with shape (512, 512) and frame name
        - test - images with shpae (1, 512, 512) and frame name
    """

    def __init__(self, transform: A.Compose, train: bool):
        self.train = train
        self.transform = transform

        if train:
            self.images = np.load(conf.dataset_folder / "training_images.npy")
            self.labels = np.load(conf.dataset_folder / "training_labels.npy")
            self.frame_names = (conf.dataset_folder / "training_frame_names.txt").read_text().split()

        else:
            self.images = np.load(conf.dataset_folder / "testing_images.npy")
            self.frame_names = (conf.dataset_folder / "testing_frame_names.txt").read_text().split()

    def __getitem__(self, item):
        image = self.images[item]
        image = np.expand_dims(image, axis=2)
        assert image.shape == (512, 512, 1)

        frame_name = self.frame_names[item]

        if self.train:
            label = self.labels[item]
            transformed = self.transform(image=image, mask=label)
            mask = torch.unsqueeze(transformed['mask'], dim=0).float()
            return transformed['image'], mask, frame_name

        else:
            transformed = self.transform(image=image)
            return transformed['image'], frame_name

    def __len__(self):
        return len(self.images)
