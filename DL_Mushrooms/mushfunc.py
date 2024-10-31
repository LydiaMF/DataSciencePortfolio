# functions for
# Mushroom Classification (Computer Vision)


# import torch.utils.data as data

import pandas as pd
import numpy as np
# import glob

import torch

# import pytorch


# import torchvision.models as models
import lightning as L

#import pytorch_lightning as L

import torch.nn as nn
import torchmetrics


# from torchsummary import summary
# from torchvision import datasets, transforms, models
from torchvision import models

# from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

import cv2
import albumentations as A

# import albumentations.pytorch
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
#from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from lightning.pytorch import Trainer


# from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import matplotlib.pyplot as plt
import copy


dev = (
    torch.device("cuda:0")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# https://albumentations.ai/docs/examples/pytorch_classification/
# works with albumentation only !


class MushroomsDataset(Dataset):
    def __init__(self, images_filepaths=None, labels=None, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform
        self.labels = labels
        # self.classes, self.class_to_idx = self.find_classes(self.images_filepaths)

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]

        # im_class = image_filepath.split("/")[-2]
        label = self.labels[idx]  # class_to_idx[im_class]

        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label


def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose(
        [
            t
            for t in dataset.transform
            if not isinstance(t, (A.Normalize, ToTensorV2))
        ]
    )
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def split_data_from_path_df(df, x_col="filename", y_col="class_label"):
    X = df[x_col].to_frame()
    y = df[y_col].to_frame()



    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=1
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        stratify=y_train_val,
        test_size=0.2,
        random_state=1,
    )

    train_files = [X_train[x_col].tolist(), y_train[y_col].tolist()]
    val_files =   [X_val[x_col].tolist(), y_val[y_col].tolist()]
    test_files =  [X_test[x_col].tolist(), y_test[y_col].tolist()]

    return train_files, val_files, test_files


# train_files, val_files, test_files = split_data_from_path_df(mushroom_df)


class MushroomDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_df=None,
        filename_col = 'filename',
        class_col = 'class',
        batch_size=32,
        augmentations=None,
        basic_transforms=None,
        weighted_sample=True,
        num_workers = 2
    ):
        super().__init__()
        """Expects augmentations/transforms from albumentation module
        """

        self.data_df = data_df
        self.filename_col = filename_col
        self.class_col = class_col 

        self.transform = basic_transforms
        self.augmentations = augmentations

        self.weighted_sample = weighted_sample
        self.batch_size = batch_size
        self.dims = (3, 224, 224)
        self.num_classes = 9
        self.seed = torch.Generator().manual_seed(42)

        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Separate train, val, and test dataset for use in dataloaders

        train_files, val_files, test_files = split_data_from_path_df(
            self.data_df, x_col = self.filename_col, y_col = self.class_col
        )

        if self.augmentations:
            transform = self.augmentations
        else:
            transform = self.transform

        train_dataset = MushroomsDataset(
            images_filepaths=train_files[0],
            labels=train_files[1],
            transform=transform,
        )

        val_dataset = MushroomsDataset(
            images_filepaths=val_files[0],
            labels=val_files[1],
            transform=self.transform,
        )

        test_dataset = MushroomsDataset(
            images_filepaths=test_files[0],
            labels=test_files[1],
            transform=self.transform,
        )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = test_dataset

    def weighted_sampler(self):
        # setup sampler
        class_counts = self.num_classes * [
            0
        ]  # Initialize counts for each class

        # Count the number of samples in each class
        train_targets = self.train_dataset.labels
        for sample in train_targets:
            class_counts[sample] += 1

        # Calculate the weight for each sample
        weights = [1.0 / class_counts[sample] for sample in train_targets]

        # Create a sampler with the calculated weights
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_targets), replacement=True
        )

        return sampler

    def train_dataloader(self):
        if self.weighted_sample:
            sampler = self.weighted_sampler()
        else:
            sampler = None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,  # https://lightning.ai/docs/pytorch/stable/advanced/speed.html
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# https://www.kaggle.com/code/shreydan/resnet50-pytorch-lightning-kfolds#Datasets

from torchvision.models import resnet18, ResNet18_Weights

class ResNet18TransferModel(L.LightningModule):
    def __init__(
        self,
        #pretrained=True,
        in_channels=3,
        num_classes=9,
        lr=3e-4,
        freeze=True,
    ):
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr

        self.model = self.backbone #models.resnet18(pretrained=pretrained)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )

        self.usecuda = 0
        if torch.cuda.is_available():
            self.usecuda = 1

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        target_tensor=torch.from_numpy(np.array(y.cpu())).long()

        if self.usecuda:
            target_tensor = target_tensor.cuda()

        preds = self.model(x)

        loss = self.loss_fn(preds, target_tensor)
        self.train_acc(torch.argmax(preds, dim=1), y)

        self.log("train_loss", loss.item(), on_epoch=True)
        self.log("train_acc", self.train_acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        target_tensor=torch.from_numpy(np.array(y.cpu())).long()

        if self.usecuda:
            target_tensor = target_tensor.cuda()

        preds = self.model(x)

        loss = self.loss_fn(preds, target_tensor)
        self.val_acc(torch.argmax(preds, dim=1), y)

        self.log("val_loss", loss.item(), on_epoch=True)
        self.log("val_acc", self.val_acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        self.test_acc(torch.argmax(preds, dim=1), y)

        self.log("test_acc", self.test_acc, on_epoch=True)





def get_lr(model, datamodule=None, 
           min_lr=1e-6, 
           max_lr=1e-1, 
           early_stop_threshold=None,
           train_max_epochs = 5, 
           result_path = "lr_finder.csv"
           ):

    if torch.cuda.is_available():
        accelerator="gpu"
    else:
        accelerator="cpu"
        
    trainer = Trainer(max_epochs=5, accelerator=accelerator) #, devices=[0])

    tuner = Tuner(trainer)

    lr_finder_aug = tuner.lr_find(model, 
                                  datamodule=datamodule, 
                                  min_lr=min_lr, 
                                  max_lr=max_lr, 
                                  early_stop_threshold=early_stop_threshold)

    pd.DataFrame(lr_finder_aug.results).to_csv(result_path)

    fig = lr_finder_aug.plot(suggest=True)
    new_lr = lr_finder_aug.suggestion()
    print(f"Suggested learning rate: {new_lr:.6f}")

    return new_lr