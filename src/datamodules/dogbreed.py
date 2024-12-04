import os
import shutil
from pathlib import Path
from typing import Union
from zipfile import ZipFile
from torchvision import transforms, datasets
import gdown
import lightning as L
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from PIL import Image


class CatDogDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for loading and preparing dog breed images
    for training, validation, and testing. This module manages the datasets
    and handles the dataloader configurations like batch size and number of workers.

    """

    def __init__(
        self,
        data_dir: str = "data/catdog",
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.pin_memory = pin_memory

    def setup(self, stage: str):
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Create full dataset using CustomImageFolder
        full_dataset = CustomImageFolder(root=self.data_dir, transform=transform)
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        self.class_names = full_dataset.classes
        print("+"*50)
        print(self.class_names)
        print("+"*50)

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

    def prepare_data(self):
        # Download data if needed (not required if data is already present)
        # self.download_data()
        self.clean_data()

    def clean_data(self):
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    file_path = os.path.join(root, file)
                    try:
                        with Image.open(file_path) as img:
                            img.verify()
                    except (IOError, SyntaxError) as e:
                        print(f'Bad file {file_path}: {e}')
                        # Optionally, remove the file
                        # os.remove(file_path)

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            # Return a placeholder image or skip to the next valid image
            return self.__getitem__((index + 1) % len(self))
