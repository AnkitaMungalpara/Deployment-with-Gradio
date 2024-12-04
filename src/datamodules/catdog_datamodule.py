import os
import shutil
from pathlib import Path
from typing import Union
from zipfile import ZipFile

from torch.utils.data import DataLoader, random_split
import lightning as L
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


class CatDogDataModule(L.LightningModule):
    """
    A PyTorch Lightning DataModule for loading and preparing dog breed images
    for training, validation, and testing. This module manages the datasets
    and handles the dataloader configurations like batch size and number of workers.

    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        num_workers: int = 4,
        batch_size: int = 32,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._dataset = None

    def prepare_data(self):
        dataset_path = self.data_dir 
        if not dataset_path.exists():
            download_and_extract_archive(
                url="https://drive.google.com/uc?export=download&id=1Bu3HQmZ6_XP-qnEuCVJ4Bg4641JuoPbx",
                download_root=self.data_dir,
                remove_finished=True,
            )

    def setup(self, stage: str):
        if self._dataset is None:
            self._dataset = ImageFolder(
                root=self.data_dir / "train",
                transform=self.train_transform,
            )
            train_size = int(self._splits[0] * len(self._dataset))
            val_size = int(self._splits[1] * len(self._dataset))
            test_size = len(self._dataset) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self._dataset, [train_size, val_size, test_size]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    @property
    def normalize_transform(self):
        transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return transform

    @property
    def train_transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )
        return transform

    @property
    def val_transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

        return transform
